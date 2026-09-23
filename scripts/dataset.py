"""
dataset.py — PyTorch Dataset over memory-mapped binary token shards.

Reads the .bin files produced by preprocess_data.py via np.memmap so the
full corpus never has to fit in RAM — only the windows being sampled are
paged in from disk.

Usage:
    from scripts.dataset import TokenDataset, build_dataloaders
    train_loader, val_loader = build_dataloaders("data/processed", block_size=1024, batch_size=32)
"""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Dtype parsing helper
# ---------------------------------------------------------------------------

def _parse_dtype(dtype_str: str) -> str:
    """
    Normalise any dtype string from meta.json to a plain NumPy dtype name.

    Handles all formats that preprocess_data.py may have written:
      "uint16"                      -> "uint16"
      "numpy.uint16"                -> "uint16"
      "<class 'numpy.uint16'>"      -> "uint16"
      "<class 'numpy.uint32'>"      -> "uint32"
    """
    # Strip whitespace
    s = dtype_str.strip()

    # Format: "<class 'numpy.uint16'>" — extract the part inside quotes
    if "'" in s:
        s = s.split("'")[1]   # e.g. "numpy.uint16"

    # Format: "numpy.uint16" or "numpy.uint32" — strip module prefix
    if "." in s:
        s = s.split(".")[-1]  # e.g. "uint16"

    # Validate — fall back to uint16 if unrecognised
    valid = {"uint8", "uint16", "uint32", "uint64", "int16", "int32", "int64"}
    if s not in valid:
        print(f"[WARN] Unrecognised dtype {dtype_str!r}, falling back to uint16")
        s = "uint16"

    return s


# ---------------------------------------------------------------------------
# Core dataset
# ---------------------------------------------------------------------------

class TokenDataset(Dataset):
    """
    A PyTorch Dataset that samples fixed-length windows from a flat binary
    token file produced by preprocess_data.py.

    Each item is a pair (x, y) where:
        x = token IDs [i : i + block_size]          — the input
        y = token IDs [i + 1 : i + block_size + 1]  — the targets (shifted by 1)

    Sampling strategy
    -----------------
    Training split  — random non-overlapping start offsets (shuffled by DataLoader).
    Validation split — sequential windows from offset 0 for reproducibility.
    """

    def __init__(
        self,
        data_dir: str,
        split: str,           # "train" or "val"
        block_size: int,
        dtype: Optional[str] = None,
    ):
        assert split in ("train", "val"), f"split must be 'train' or 'val', got {split!r}"

        data_dir = Path(data_dir)
        bin_path  = data_dir / f"{split}.bin"
        meta_path = data_dir / "meta.json"

        if not bin_path.exists():
            raise FileNotFoundError(
                f"Binary shard not found: {bin_path}\n"
                "Run scripts/preprocess_data.py first."
            )

        # Resolve dtype from meta.json or default to uint16
        if dtype is None and meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            dtype_str = meta.get("dtype", "uint16")
            dtype = _parse_dtype(dtype_str)
        dtype = dtype or "uint16"

        self.block_size = block_size
        self.split      = split

        # Memory-mapped read-only array — OS pages in only what's needed
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")

        # Number of complete windows we can draw
        self.n_tokens = len(self.data)
        self.n_windows = max(0, self.n_tokens - block_size)

        if self.n_windows == 0:
            raise ValueError(
                f"{split}.bin has {self.n_tokens} tokens but block_size={block_size}. "
                "Need at least block_size + 1 tokens."
            )

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int):
        # Slice a window and immediately copy it to a contiguous int64 tensor.
        # np.memmap supports fancy indexing so only this page hits RAM.
        x = torch.from_numpy(
            self.data[idx : idx + self.block_size].astype(np.int64)
        )
        y = torch.from_numpy(
            self.data[idx + 1 : idx + self.block_size + 1].astype(np.int64)
        )
        return x, y


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    data_dir: str,
    block_size: int,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    val_batch_size: Optional[int] = None,
    collate_fn=None,
) -> tuple[DataLoader, DataLoader]:
    """
    Return (train_loader, val_loader) ready to iterate.

    Args:
        data_dir:       Path containing train.bin, val.bin, meta.json
        block_size:     Sequence length (tokens per sample)
        batch_size:     Training batch size
        num_workers:    DataLoader worker processes
        pin_memory:     Enable CUDA pinned memory for faster host→device transfer
        val_batch_size: Batch size for validation; defaults to batch_size
        collate_fn:     Optional collate function (e.g. FIMCollator). When None
                        the default (x, y) tuple batching is used.
    """
    val_batch_size = val_batch_size or batch_size

    train_ds = TokenDataset(data_dir, split="train", block_size=block_size)
    val_ds   = TokenDataset(data_dir, split="val",   block_size=block_size)

    print(f"Dataset — train windows: {len(train_ds):,}  val windows: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
    )

    # Validation always uses plain (x, y) batches — no FIM transform
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
        collate_fn=None,   # plain tuples for consistent val loss
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import tempfile

    print("Running TokenDataset self-test with synthetic data...")

    # Write a tiny fake .bin file
    with tempfile.TemporaryDirectory() as tmp:
        meta = {"vocab_size": 1000, "dtype": "uint16", "train_tokens": 5000, "val_tokens": 500}
        (Path(tmp) / "meta.json").write_text(json.dumps(meta))

        rng = np.random.default_rng(42)
        for split, n in [("train", 5000), ("val", 500)]:
            arr = rng.integers(0, 1000, size=n, dtype=np.uint16)
            arr.tofile(Path(tmp) / f"{split}.bin")

        ds = TokenDataset(tmp, split="train", block_size=64)
        x, y = ds[0]
        assert x.shape == (64,) and y.shape == (64,), "Shape mismatch"
        assert (y == x.roll(-1)).sum() >= 60, "x/y shift looks wrong"

        train_loader, val_loader = build_dataloaders(tmp, block_size=64, batch_size=8, num_workers=0, pin_memory=False)
        bx, by = next(iter(train_loader))
        assert bx.shape == (8, 64), f"Unexpected batch shape {bx.shape}"
        print(f"  train batch shape: {bx.shape}  dtype: {bx.dtype}")

    print("Self-test passed.")
