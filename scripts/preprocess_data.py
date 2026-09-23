"""
preprocess_data.py — Tokenise code files into memory-mapped binary shards.

Differences from the generic version:
  - Walks directory trees, not just flat .txt files
  - Filters by code extension (from train_config.yaml)
  - Skips files that are too small / too large / non-UTF-8
  - Prepends a language tag token (<|python|> etc.) to each file
  - Inserts <|endoftext|> between documents (document-level causal LM)
  - Parallel tokenisation via multiprocessing

Output:
    data/processed/
        train.bin    — uint16 token IDs (uint32 if vocab > 65535)
        val.bin      — uint16 token IDs
        meta.json    — vocab size, split sizes, dtype, file stats

Usage:
    python scripts/preprocess_data.py
    python scripts/preprocess_data.py --raw_dir data/raw --num_workers 8
    python scripts/preprocess_data.py --num_workers 0   # Windows safe mode
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Language tag mapping  (extension → special token)
# ---------------------------------------------------------------------------

EXT_TO_LANG_TAG = {
    ".py":    "<|python|>",
    ".js":    "<|javascript|>",
    ".jsx":   "<|javascript|>",
    ".ts":    "<|typescript|>",
    ".tsx":   "<|typescript|>",
    ".cpp":   "<|cpp|>",
    ".cc":    "<|cpp|>",
    ".cxx":   "<|cpp|>",
    ".c":     "<|cpp|>",
    ".h":     "<|cpp|>",
    ".hpp":   "<|cpp|>",
    ".java":  "<|java|>",
    ".go":    "<|go|>",
    ".rs":    "<|rust|>",
    ".rb":    "<|rust|>",   # fallback — no ruby tag defined
    ".cs":    "<|cpp|>",    # fallback
    ".php":   "<|javascript|>",  # fallback
    ".swift": "<|cpp|>",         # fallback
    ".kt":    "<|java|>",        # fallback
    ".sh":    "<|python|>",      # fallback
    ".md":    None,              # no language tag for markdown
    ".txt":   None,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Tokenise code files into binary shards.")
    p.add_argument("--config",         type=str, default="configs/train_config.yaml")
    p.add_argument("--raw_dir",        type=str, default=None,
                   help="Override data.raw_dir from config")
    p.add_argument("--out_dir",        type=str, default=None,
                   help="Override data.processed_dir from config")
    p.add_argument("--tokenizer_path", type=str, default=None,
                   help="Override data.tokenizer_path from config")
    p.add_argument("--val_ratio",      type=float, default=0.005)
    p.add_argument("--num_workers",    type=int,   default=4,
                   help="Worker processes (use 0 on Windows if you hit issues)")
    p.add_argument("--chunk_files",    type=int,   default=500,
                   help="Files per parallel chunk")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_files(raw_dir: str, extensions: set, exclude_patterns: list,
                   min_bytes: int, max_bytes: int) -> list[Path]:
    """
    Walk raw_dir recursively and return all code files that pass filters.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        print(f"[ERROR] raw_dir does not exist: {raw_dir}")
        sys.exit(1)

    exclude_lower = [p.lower() for p in exclude_patterns]
    paths = []

    for f in raw_dir.rglob("*"):
        if not f.is_file():
            continue

        # Extension filter
        if f.suffix.lower() not in extensions:
            continue

        # Exclusion pattern filter (simple substring match on path parts)
        parts_lower = [p.lower() for p in f.parts]
        if any(excl in part for excl in exclude_lower for part in parts_lower):
            continue

        # Size filter
        try:
            size = f.stat().st_size
        except OSError:
            continue
        if size < min_bytes or size > max_bytes:
            continue

        paths.append(f)

    return sorted(paths)


# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------

def _tokenize_files(args_tuple):
    """
    Tokenise a list of file paths.
    Returns a flat list of token IDs with <|endoftext|> between documents
    and optional language tags at the start of each file.
    """
    file_paths, tokenizer_path, eot_token, lang_tag_map = args_tuple

    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    eot_id = tok.convert_tokens_to_ids(eot_token)
    if eot_id == tok.unk_token_id:
        eot_id = tok.eos_token_id  # fallback

    all_ids: list[int] = []
    skipped = 0

    for fpath in file_paths:
        try:
            text = fpath.read_text(encoding="utf-8", errors="strict")
        except (UnicodeDecodeError, OSError):
            skipped += 1
            continue

        text = text.strip()
        if not text:
            skipped += 1
            continue

        # Prepend language tag if available
        ext = fpath.suffix.lower()
        lang_tag = lang_tag_map.get(ext)
        if lang_tag:
            lang_id = tok.convert_tokens_to_ids(lang_tag)
            # If the tag isn't in the tokenizer yet, skip the tag (don't inject unk)
            if lang_id != tok.unk_token_id:
                all_ids.append(lang_id)

        # Tokenise the file content (no BOS/EOS — we use EOT between docs instead)
        encoded = tok.encode(text, add_special_tokens=False)
        all_ids.extend(encoded)

        # Document separator
        all_ids.append(eot_id)

    return all_ids, skipped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    tok_cfg  = cfg.get("tokenizer", {})

    raw_dir        = args.raw_dir        or data_cfg.get("raw_dir",        "data/raw")
    out_dir        = args.out_dir        or data_cfg.get("processed_dir",  "data/processed")
    tokenizer_path = args.tokenizer_path or data_cfg.get("tokenizer_path", "models/tokenizer")

    extensions = set(data_cfg.get("code_extensions", [
        ".py", ".js", ".ts", ".jsx", ".tsx",
        ".cpp", ".c", ".h", ".hpp", ".java",
        ".go", ".rs", ".rb", ".cs", ".php",
        ".swift", ".kt", ".sh", ".md", ".txt",
    ]))
    exclude_patterns = data_cfg.get("exclude_patterns", [
        "node_modules", ".git", "__pycache__", "dist", "build", "vendor",
    ])
    min_bytes = data_cfg.get("min_file_bytes", 64)
    max_bytes = data_cfg.get("max_file_bytes", 1_048_576)

    # ------------------------------------------------------------------
    # Validate tokenizer
    # ------------------------------------------------------------------
    if not os.path.isdir(tokenizer_path):
        print(f"[ERROR] Tokenizer not found at '{tokenizer_path}'.")
        print("        Run scripts/train_tokenizer.py first.")
        sys.exit(1)

    from transformers import PreTrainedTokenizerFast
    tokenizer  = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    vocab_size = tokenizer.vocab_size
    print(f"Loaded tokenizer  vocab_size={vocab_size}")

    eot_token = "<|endoftext|>"

    # ------------------------------------------------------------------
    # Discover files
    # ------------------------------------------------------------------
    print(f"\nScanning '{raw_dir}' for code files...")
    all_files = discover_files(raw_dir, extensions, exclude_patterns, min_bytes, max_bytes)

    if not all_files:
        print(f"[ERROR] No code files found under '{raw_dir}' with extensions {extensions}.")
        print("        Run scripts/download_data.py first, or add your own code files.")
        sys.exit(1)

    total_size_mb = sum(f.stat().st_size for f in all_files) / 1024 / 1024
    print(f"Found {len(all_files):,} files  ({total_size_mb:.1f} MB)")

    # Print breakdown by extension
    from collections import Counter
    ext_counts = Counter(f.suffix.lower() for f in all_files)
    for ext, count in ext_counts.most_common(15):
        print(f"  {ext:10s}  {count:,}")

    # ------------------------------------------------------------------
    # Build work chunks
    # ------------------------------------------------------------------
    chunk_size = args.chunk_files
    chunks = [all_files[i : i + chunk_size] for i in range(0, len(all_files), chunk_size)]

    # Build lang_tag_map for workers (only pass serialisable data)
    lang_tag_map = {ext: EXT_TO_LANG_TAG.get(ext) for ext in extensions}

    work_items = [(chunk, tokenizer_path, eot_token, lang_tag_map) for chunk in chunks]

    # ------------------------------------------------------------------
    # Tokenise in parallel
    # ------------------------------------------------------------------
    print(f"\nTokenising {len(chunks)} chunk(s) using {args.num_workers} worker(s)...")

    all_ids:    list[int] = []
    total_skip: int       = 0

    if args.num_workers > 0:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.num_workers) as pool:
            for ids, skipped in tqdm(
                pool.imap(_tokenize_files, work_items),
                total=len(work_items),
                desc="Tokenising",
                unit="chunk",
            ):
                all_ids.extend(ids)
                total_skip += skipped
    else:
        # Single-process (safe on Windows)
        for item in tqdm(work_items, desc="Tokenising", unit="chunk"):
            ids, skipped = _tokenize_files(item)
            all_ids.extend(ids)
            total_skip += skipped

    total_tokens = len(all_ids)
    print(f"\nTotal tokens  : {total_tokens:,}")
    print(f"Skipped files : {total_skip:,}")

    if total_tokens == 0:
        print("[ERROR] Zero tokens produced — check your files and tokenizer.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Train / val split
    # ------------------------------------------------------------------
    val_ratio = max(0.0, min(args.val_ratio, 0.5))
    n_val     = max(1, int(total_tokens * val_ratio))
    n_train   = total_tokens - n_val
    print(f"Train tokens  : {n_train:,}")
    print(f"Val tokens    : {n_val:,}  ({val_ratio*100:.2f}%)")

    dtype = np.uint16 if vocab_size < 65535 else np.uint32

    arr = np.array(all_ids, dtype=dtype)
    del all_ids

    # ------------------------------------------------------------------
    # Write shards
    # ------------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    out_dir = Path(out_dir)

    train_path = out_dir / "train.bin"
    val_path   = out_dir / "val.bin"

    arr[:n_train].tofile(train_path)
    arr[n_train:].tofile(val_path)

    print(f"\nWritten: {train_path}  ({train_path.stat().st_size / 1024**2:.1f} MB)")
    print(f"Written: {val_path}   ({val_path.stat().st_size / 1024**2:.1f} MB)")

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    meta = {
        "vocab_size":    vocab_size,
        "total_tokens":  total_tokens,
        "train_tokens":  n_train,
        "val_tokens":    n_val,
        "val_ratio":     val_ratio,
        "dtype":         dtype.__name__,   # always "uint16" or "uint32", never "numpy.uint16"
        "tokenizer":     str(tokenizer_path),
        "total_files":   len(all_files),
        "skipped_files": total_skip,
        "ext_counts":    dict(ext_counts.most_common()),
    }
    meta_path = out_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata: {meta_path}")
    print("\nPreprocessing complete. Next: python scripts/train.py")


if __name__ == "__main__":
    main()
