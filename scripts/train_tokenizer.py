"""
train_tokenizer.py — Train a BPE tokenizer on your raw code data.

Reads code files (filtered by extension) under data/raw/, trains a
Byte-Pair Encoding (BPE) tokenizer, injects FIM and language-tag special
tokens from train_config.yaml, and saves as a HuggingFace
PreTrainedTokenizerFast to models/tokenizer/.

Usage:
    python scripts/train_tokenizer.py
    python scripts/train_tokenizer.py --vocab_size 49152 --raw_dir data/raw

The saved tokenizer can then be loaded anywhere with:
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast.from_pretrained("models/tokenizer")
"""

import argparse
import glob
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train a BPE tokenizer on raw code files.")
    p.add_argument("--config",     type=str, default="configs/train_config.yaml")
    p.add_argument("--raw_dir",    type=str, default=None,          help="Override data.raw_dir")
    p.add_argument("--out_dir",    type=str, default=None,          help="Override data.tokenizer_path")
    p.add_argument("--vocab_size", type=int, default=None,          help="Override tokenizer.vocab_size")
    p.add_argument("--min_freq",   type=int, default=None,          help="Override tokenizer.min_freq")
    p.add_argument("--batch_size", type=int, default=1000,          help="Lines per batch")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def iter_text_files(raw_dir: str):
    """Yield paths to every code file found recursively under raw_dir."""
    import glob as _glob
    # Accept all common code extensions
    exts = {
        ".py", ".js", ".ts", ".jsx", ".tsx",
        ".cpp", ".c", ".h", ".hpp", ".java",
        ".go", ".rs", ".rb", ".cs", ".php",
        ".swift", ".kt", ".sh", ".md", ".txt",
    }
    paths = []
    for f in Path(raw_dir).rglob("*"):
        if f.is_file() and f.suffix.lower() in exts:
            # Skip auto-generated / minified files by size heuristic
            try:
                if f.stat().st_size > 1_048_576:  # > 1 MB
                    continue
            except OSError:
                continue
            paths.append(f)

    paths = sorted(paths)
    if not paths:
        print(f"[ERROR] No code files found under '{raw_dir}'.")
        print("        Run scripts/download_data.py first.")
        sys.exit(1)
    total_mb = sum(p.stat().st_size for p in paths) / 1024 / 1024
    print(f"Found {len(paths):,} file(s)  ({total_mb:.1f} MB) under '{raw_dir}'")
    return paths


def batch_iterator(paths: list, batch_size: int):
    """Yield batches of lines from the file list — memory-efficient."""
    batch = []
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    batch.append(line)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
    if batch:
        yield batch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Load config — explicit UTF-8 to avoid Windows cp1252 codec errors
    cfg = {}
    if os.path.exists(args.config):
        import yaml
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    tok_cfg  = cfg.get("tokenizer", {})
    data_cfg = cfg.get("data", {})

    raw_dir    = args.raw_dir    or data_cfg.get("raw_dir",        "data/raw")
    out_dir    = args.out_dir    or data_cfg.get("tokenizer_path", "models/tokenizer")
    vocab_size = args.vocab_size or tok_cfg.get("vocab_size",      49152)
    min_freq   = args.min_freq   or tok_cfg.get("min_freq",        2)

    # Extra special tokens from config (FIM tokens + language tags)
    extra_special = tok_cfg.get("extra_special_tokens", [
        "<fim_prefix>", "<fim_suffix>", "<fim_middle>",
        "<|endoftext|>",
        "<|python|>", "<|javascript|>", "<|typescript|>",
        "<|cpp|>", "<|java|>", "<|go|>", "<|rust|>",
    ])

    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.processors import TemplateProcessing
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from transformers import PreTrainedTokenizerFast
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("        Run: pip install tokenizers transformers")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Discover training files
    # ------------------------------------------------------------------
    paths = iter_text_files(raw_dir)

    # ------------------------------------------------------------------
    # 2. Build the tokenizer
    # ------------------------------------------------------------------
    print(f"\nTraining BPE tokenizer  vocab_size={vocab_size}  min_freq={min_freq}")
    print(f"Extra special tokens: {extra_special}")

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder        = ByteLevelDecoder()

    # Core special tokens first; extras appended after training
    core_special_tokens = ["<unk>", "<pad>", "<bos>", "<eos>"]
    all_special_tokens  = core_special_tokens + [
        t for t in extra_special if t not in core_special_tokens
    ]

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=all_special_tokens,
        show_progress=True,
    )

    tokenizer.train_from_iterator(
        batch_iterator(paths, args.batch_size),
        trainer=trainer,
    )

    # ------------------------------------------------------------------
    # 3. Post-processor
    # ------------------------------------------------------------------
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        pair="<bos> $A <eos> $B:1 <eos>:1",
        special_tokens=[("<bos>", bos_id), ("<eos>", eos_id)],
    )

    # ------------------------------------------------------------------
    # 4. Save
    # ------------------------------------------------------------------
    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        additional_special_tokens=[t for t in extra_special
                                    if t not in ("<unk>","<pad>","<bos>","<eos>")],
    )
    fast_tok.save_pretrained(out_dir)
    print(f"\nTokenizer saved to '{out_dir}'")
    print(f"  Vocabulary size  : {fast_tok.vocab_size}")
    print(f"  Special tokens   : {fast_tok.all_special_tokens}")

    # ------------------------------------------------------------------
    # 5. Smoke test
    # ------------------------------------------------------------------
    sample = "def hello_world():\n    print('Hello, world!')\n"
    ids = fast_tok.encode(sample)
    decoded = fast_tok.decode(ids, skip_special_tokens=True)
    print(f"\nSmoke test (Python):")
    print(f"  Input    : {sample!r}")
    print(f"  Tokens   : {len(ids)}  IDs: {ids[:20]}{'...' if len(ids)>20 else ''}")
    print(f"  Decoded  : {decoded!r}")

    # Verify FIM tokens are in vocab
    for tok_name in ["<fim_prefix>", "<fim_suffix>", "<fim_middle>"]:
        tid = fast_tok.convert_tokens_to_ids(tok_name)
        if tid != fast_tok.unk_token_id:
            print(f"  {tok_name} → ID {tid}  OK")
        else:
            print(f"  [WARN] {tok_name} not found in vocabulary")


if __name__ == "__main__":
    main()
