"""
download_data.py — Fetch public code datasets for coder LLM training.

Available datasets (all FREE, no login required):
  --dataset codeparrot-small   cleaned Python subset ~400 MB, no login needed
  --dataset codeparrot         full ~50 GB Python from GitHub (streamed, stop anytime)
  --dataset codessearchnet     Python/JS/Java/PHP/Ruby/Go function+docstring pairs ~2 GB
  --dataset humaneval          164 Python eval problems (for benchmarking only)
  --dataset github-code        github-code dataset (Python subset), no login needed
  --dataset all-small          codeparrot-small + codessearchnet + humaneval (RECOMMENDED)

NOTE: the-stack requires a HuggingFace login and dataset license acceptance.
      Use --dataset stack-python if you have done that.

Usage:
    python scripts/download_data.py --dataset all-small
    python scripts/download_data.py --dataset codeparrot --max_gb 10
    python scripts/download_data.py --dataset humaneval
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Download code datasets for LLM training.")
    p.add_argument("--dataset",   type=str, default="all-small",
                   choices=["codeparrot-small", "codeparrot", "stack-python",
                            "stack-small", "codessearchnet", "github-code",
                            "humaneval", "all-small"],
                   help="Which dataset to download")
    p.add_argument("--out_dir",   type=str, default="data/raw",
                   help="Root output directory")
    p.add_argument("--max_gb",    type=float, default=None,
                   help="Stop after downloading this many GB (approximate)")
    p.add_argument("--max_files", type=int,   default=None,
                   help="Stop after saving this many files")
    p.add_argument("--num_proc",  type=int,   default=4,
                   help="Parallel workers for dataset processing")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def require_datasets():
    try:
        import datasets as hf_datasets
        return hf_datasets
    except ImportError:
        print("[ERROR] `datasets` not installed. Run: pip install datasets")
        sys.exit(1)


def require_requests():
    try:
        import requests
        return requests
    except ImportError:
        print("[ERROR] `requests` not installed. Run: pip install requests")
        sys.exit(1)


def bytes_written(directory: Path) -> float:
    """Total bytes in all files under directory."""
    return sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())


def write_as_text(content: str, out_dir: Path, filename: str):
    """Write a single document to out_dir/filename.txt"""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Dataset downloaders
# ---------------------------------------------------------------------------

def download_codeparrot_small(out_dir: Path, max_files: int, max_bytes: float):
    """
    codeparrot/codeparrot-clean-train — cleaned Python from GitHub.
    Full dataset is ~50 GB; we stream it so you can stop at any size.
    """
    hf = require_datasets()
    print("Downloading codeparrot/codeparrot-clean-train (streaming)...")

    ds = hf.load_dataset(
        "codeparrot/codeparrot-clean-train",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    subdir = out_dir / "codeparrot"
    subdir.mkdir(parents=True, exist_ok=True)

    count   = 0
    skipped = 0

    for i, sample in enumerate(ds):
        content = sample.get("content", "").strip()
        if not content:
            skipped += 1
            continue

        write_as_text(content, subdir, f"{i:08d}.py")
        count += 1

        if count % 1000 == 0:
            gb = bytes_written(subdir) / 1e9
            print(f"  {count:,} files  {gb:.2f} GB", end="\r")

        if max_files and count >= max_files:
            break
        if max_bytes and bytes_written(subdir) >= max_bytes:
            break

    gb = bytes_written(subdir) / 1e9
    print(f"\ncodeparrot: {count:,} files  {gb:.2f} GB  skipped={skipped}")


def download_codeparrot_small_subset(out_dir: Path, max_files: int, max_bytes: float):
    """
    codeparrot/codeparrot-clean — small version, no gating, no login needed.
    ~400 MB of clean Python, good for a first training run.
    """
    hf = require_datasets()
    print("Downloading codeparrot/codeparrot-clean (small, no login required)...")

    # Try the small/valid split first — it's always available
    for split_name in ("train", "valid"):
        try:
            ds = hf.load_dataset(
                "codeparrot/codeparrot-clean",
                split=split_name,
                streaming=True,
            )
            break
        except Exception as e:
            print(f"  [{split_name}] failed: {e}")
            ds = None

    if ds is None:
        print("[ERROR] Could not load codeparrot/codeparrot-clean.")
        return

    subdir = out_dir / "codeparrot-small"
    subdir.mkdir(parents=True, exist_ok=True)
    count = 0

    for i, sample in enumerate(ds):
        content = sample.get("content", "").strip()
        if not content:
            continue
        write_as_text(content, subdir, f"{i:08d}.py")
        count += 1
        if count % 500 == 0:
            gb = bytes_written(subdir) / 1e9
            print(f"  {count:,} files  {gb:.3f} GB", end="\r")
        if max_files and count >= max_files:
            break
        if max_bytes and bytes_written(subdir) >= max_bytes:
            break

    gb = bytes_written(subdir) / 1e9
    print(f"\ncodeparrot-small: {count:,} files  {gb:.2f} GB")


def download_github_code(out_dir: Path, max_files: int, max_bytes: float):
    """
    codeparrot/github-code — multi-language GitHub code, no login, no gating.
    Streams Python subset. Adjust `language` list to add more languages.
    """
    hf = require_datasets()

    languages = [
        ("Python",      ".py"),
        ("JavaScript",  ".js"),
        ("TypeScript",  ".ts"),
        ("Java",        ".java"),
        ("Go",          ".go"),
        ("C++",         ".cpp"),
        ("Rust",        ".rs"),
    ]

    subdir = out_dir / "github-code"
    total_count = 0

    for lang, ext in languages:
        print(f"  Downloading github-code: {lang}")
        try:
            ds = hf.load_dataset(
                "codeparrot/github-code",
                streaming=True,
                split="train",
                filters=[["language", "==", lang]],
            )
        except Exception:
            # Older datasets version uses a different API
            try:
                ds = hf.load_dataset(
                    "codeparrot/github-code",
                    streaming=True,
                    split="train",
                )
                # Filter manually
                ds = ds.filter(lambda x: x.get("language") == lang)
            except Exception as e2:
                print(f"    [SKIP] {lang}: {e2}")
                continue

        lang_dir = subdir / lang.lower()
        lang_dir.mkdir(parents=True, exist_ok=True)
        count = 0

        for i, sample in enumerate(ds):
            content = sample.get("code", sample.get("content", "")).strip()
            if not content:
                continue
            write_as_text(content, lang_dir, f"{i:08d}{ext}")
            count += 1
            total_count += 1
            if count % 500 == 0:
                gb = bytes_written(subdir) / 1e9
                print(f"    {lang}: {count:,}  total: {total_count:,}  {gb:.2f} GB", end="\r")
            if max_files and total_count >= max_files:
                break
            if max_bytes and bytes_written(subdir) >= max_bytes:
                break

        print(f"    {lang}: {count:,} files")
        if (max_files and total_count >= max_files) or \
           (max_bytes and bytes_written(subdir) >= max_bytes):
            break

    gb = bytes_written(subdir) / 1e9
    print(f"\ngithub-code: {total_count:,} files  {gb:.2f} GB")


def download_stack_python(out_dir: Path, max_files: int, max_bytes: float):
    """
    bigcode/the-stack-dedup Python subset.
    Requires HuggingFace login: huggingface-cli login
    """
    hf = require_datasets()
    print("Downloading bigcode/the-stack-dedup (Python)...")
    print("NOTE: This requires a HuggingFace account and accepting the dataset license at")
    print("      https://huggingface.co/datasets/bigcode/the-stack-dedup")
    print("      Then run: huggingface-cli login")

    try:
        ds = hf.load_dataset(
            "bigcode/the-stack-dedup",
            data_dir="data/python",
            split="train",
            streaming=True,
        )
    except Exception as e:
        print(f"[ERROR] Could not load the-stack-dedup: {e}")
        print("Try --dataset stack-small instead (no login required).")
        return

    subdir = out_dir / "stack-python"
    subdir.mkdir(parents=True, exist_ok=True)

    count = 0
    for i, sample in enumerate(ds):
        content = sample.get("content", "").strip()
        if not content:
            continue

        write_as_text(content, subdir, f"{i:08d}.py")
        count += 1

        if count % 1000 == 0:
            gb = bytes_written(subdir) / 1e9
            print(f"  {count:,} files  {gb:.2f} GB", end="\r")

        if max_files and count >= max_files:
            break
        if max_bytes and bytes_written(subdir) >= max_bytes:
            break

    gb = bytes_written(subdir) / 1e9
    print(f"\nstack-python: {count:,} files  {gb:.2f} GB")


def download_stack_small(out_dir: Path, max_files: int, max_bytes: float):
    """
    bigcode/the-stack-smol — requires HuggingFace login + license acceptance.
    Kept here for users who have set that up.
    """
    hf = require_datasets()
    print("Downloading bigcode/the-stack-smol...")
    print("NOTE: This dataset is GATED. You must:")
    print("  1. Accept the license at https://huggingface.co/datasets/bigcode/the-stack-smol")
    print("  2. Run: huggingface-cli login")
    print("  If you haven't done that, use --dataset github-code instead (no login needed).")

    lang_ext = {
        "python":     ".py",
        "javascript": ".js",
        "typescript": ".ts",
        "java":       ".java",
        "cpp":        ".cpp",
        "go":         ".go",
        "rust":       ".rs",
    }

    subdir = out_dir / "stack-small"
    subdir.mkdir(parents=True, exist_ok=True)
    total_count = 0

    for lang, ext in lang_ext.items():
        print(f"  Language: {lang}")
        try:
            ds = hf.load_dataset(
                "bigcode/the-stack-smol",
                data_dir=f"data/{lang}",
                split="train",
                streaming=True,
            )
        except Exception as e:
            print(f"    [SKIP] {lang}: {e}")
            continue

        lang_dir = subdir / lang
        lang_dir.mkdir(exist_ok=True)
        count = 0

        for i, sample in enumerate(ds):
            content = sample.get("content", "").strip()
            if not content:
                continue
            write_as_text(content, lang_dir, f"{i:08d}{ext}")
            count += 1
            total_count += 1
            if total_count % 500 == 0:
                gb = bytes_written(subdir) / 1e9
                print(f"    {total_count:,} total  {gb:.3f} GB", end="\r")
            if max_files and total_count >= max_files:
                break
            if max_bytes and bytes_written(subdir) >= max_bytes:
                break

        print(f"    {lang}: {count:,} files")
        if (max_files and total_count >= max_files) or \
           (max_bytes and bytes_written(subdir) >= max_bytes):
            break

    gb = bytes_written(subdir) / 1e9
    print(f"\nstack-small: {total_count:,} files  {gb:.2f} GB")


def download_codessearchnet(out_dir: Path, max_files: int, max_bytes: float):
    """
    code_search_net — docstring + code pairs.
    Uses the 'code-search-net/code_search_net' path which works with
    datasets >= 2.x without trust_remote_code issues.
    """
    hf = require_datasets()
    print("Downloading code_search_net...")

    languages = ["python", "javascript", "java", "php", "ruby", "go"]
    subdir = out_dir / "codessearchnet"
    total_count = 0

    for lang in languages:
        print(f"  Language: {lang}")
        # Try multiple known dataset IDs — the name changed across versions
        loaded = None
        for dataset_id in [
            "code-search-net/code_search_net",
            "code_search_net",
        ]:
            try:
                loaded = hf.load_dataset(
                    dataset_id,
                    lang,
                    split="train",
                    trust_remote_code=True,
                )
                break
            except Exception as e:
                print(f"    [{dataset_id}] {e}")

        if loaded is None:
            print(f"    [SKIP] {lang}: all load attempts failed")
            continue

        lang_dir = subdir / lang
        lang_dir.mkdir(parents=True, exist_ok=True)

        for i, sample in enumerate(loaded):
            doc  = sample.get("func_documentation_string", "").strip()
            code = sample.get("whole_func_string", "").strip()
            if not code:
                continue
            content = f'"""\n{doc}\n"""\n{code}' if doc else code
            write_as_text(content, lang_dir, f"{i:08d}.py")
            total_count += 1
            if total_count % 500 == 0:
                gb = bytes_written(subdir) / 1e9
                print(f"    {total_count:,} total  {gb:.3f} GB", end="\r")
            if max_files and total_count >= max_files:
                break
            if max_bytes and bytes_written(subdir) >= max_bytes:
                break

        print(f"    {lang}: {i+1:,} samples")
        if (max_files and total_count >= max_files) or \
           (max_bytes and bytes_written(subdir) >= max_bytes):
            break

    gb = bytes_written(subdir) / 1e9
    print(f"\ncodessearchnet: {total_count:,} files  {gb:.2f} GB")


def download_humaneval(out_dir: Path):
    """
    OpenAI HumanEval — 164 Python programming problems with test cases.
    Used for evaluation (pass@k), not training.
    Saved as data/HumanEval.jsonl
    """
    requests = require_requests()
    url = (
        "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
    )
    out_path = Path("data/HumanEval.jsonl.gz")
    final_path = Path("data/HumanEval.jsonl")

    if final_path.exists():
        print(f"HumanEval already exists at {final_path}")
        return

    print(f"Downloading HumanEval from {url} ...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(out_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    import gzip
    with gzip.open(out_path, "rb") as gz:
        final_path.write_bytes(gz.read())
    out_path.unlink()

    # Count problems
    n = sum(1 for _ in open(final_path))
    print(f"HumanEval: {n} problems saved to {final_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out_dir   = Path(args.out_dir)
    max_bytes = args.max_gb * 1e9 if args.max_gb else None
    max_files = args.max_files

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir.resolve()}")

    dataset = args.dataset

    if dataset == "codeparrot-small":
        download_codeparrot_small_subset(out_dir, max_files, max_bytes)

    elif dataset == "codeparrot":
        download_codeparrot_small(out_dir, max_files, max_bytes)

    elif dataset == "stack-python":
        download_stack_python(out_dir, max_files, max_bytes)

    elif dataset == "stack-small":
        download_stack_small(out_dir, max_files, max_bytes)

    elif dataset == "codessearchnet":
        download_codessearchnet(out_dir, max_files, max_bytes)

    elif dataset == "github-code":
        download_github_code(out_dir, max_files, max_bytes)

    elif dataset == "humaneval":
        download_humaneval(out_dir)

    elif dataset == "all-small":
        print("=== Downloading all-small bundle (no login required) ===")
        print("  1/3  codeparrot-small  (~400 MB, Python)")
        download_codeparrot_small_subset(out_dir, max_files, max_bytes)
        print("  2/3  codessearchnet  (~600 MB, multi-language)")
        download_codessearchnet(out_dir, max_files, max_bytes)
        print("  3/3  humaneval  (eval benchmark, tiny)")
        download_humaneval(out_dir)

    total_gb = bytes_written(out_dir) / 1e9
    print(f"\nDone. Total in {out_dir}: {total_gb:.2f} GB")
    print("Next step: python scripts/train_tokenizer.py")


if __name__ == "__main__":
    main()
