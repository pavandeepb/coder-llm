"""
fim_collator.py — Fill-In-the-Middle (FIM) data collator.

FIM training teaches the model to predict a MIDDLE span given PREFIX + SUFFIX.
This is the key technique behind Copilot-style completions where the model
fills in code between what's already written above and below the cursor.

Two orderings are used (both in the literature):
  PSM:  <fim_prefix> PREFIX <fim_suffix> SUFFIX <fim_middle> MIDDLE <eos>
  SPM:  <fim_suffix> SUFFIX <fim_prefix> PREFIX <fim_middle> MIDDLE <eos>

50% of batches are standard causal LM (no transform).
Of the FIM half, 50% are PSM and 50% are SPM.

Reference: "Efficient Training of Language Models to Fill in the Middle"
           (Bavarian et al., 2022) — https://arxiv.org/abs/2207.14255

Usage:
    collator = FIMCollator(tokenizer, fim_rate=0.5, spm_rate=0.5)
    loader   = DataLoader(dataset, batch_size=8, collate_fn=collator)
"""

import random
from typing import Optional

import torch
from torch.nn.utils.rnn import pad_sequence


class FIMCollator:
    """
    Drop-in collate_fn for DataLoader.

    Takes a list of (x, y) tuples from TokenDataset — x is a 1-D token-ID
    tensor — and either passes it through unchanged (causal LM) or transforms
    it into FIM format.

    Args:
        tokenizer:    A HuggingFace PreTrainedTokenizerFast that has the FIM
                      special tokens already added.
        fim_rate:     Fraction of samples to apply FIM to (default 0.5).
        spm_rate:     Within FIM, fraction to use SPM order (default 0.5).
        pad_token_id: Override padding token ID (auto-detected from tokenizer).
    """

    # Special token strings (must match train_tokenizer.py + train_config.yaml)
    FIM_PREFIX = "<fim_prefix>"
    FIM_SUFFIX = "<fim_suffix>"
    FIM_MIDDLE = "<fim_middle>"
    EOT        = "<|endoftext|>"

    def __init__(
        self,
        tokenizer,
        fim_rate: float = 0.5,
        spm_rate: float = 0.5,
        pad_token_id: Optional[int] = None,
    ):
        self.fim_rate  = fim_rate
        self.spm_rate  = spm_rate

        # Resolve special token IDs
        self.pad_id = pad_token_id or tokenizer.pad_token_id or 0
        self.eos_id = tokenizer.eos_token_id or 0

        def tok_id(name: str) -> int:
            tid = tokenizer.convert_tokens_to_ids(name)
            if tid == tokenizer.unk_token_id:
                raise ValueError(
                    f"Special token {name!r} not in tokenizer vocabulary.\n"
                    "Re-run train_tokenizer.py with the extra_special_tokens "
                    "listed in train_config.yaml."
                )
            return tid

        self.prefix_id = tok_id(self.FIM_PREFIX)
        self.suffix_id = tok_id(self.FIM_SUFFIX)
        self.middle_id = tok_id(self.FIM_MIDDLE)

    # ------------------------------------------------------------------
    # Internal transform
    # ------------------------------------------------------------------

    def _fim_transform(self, ids: list[int]) -> list[int]:
        """
        Split `ids` into three parts at two random cut-points and
        re-arrange into PSM or SPM FIM format.

        Returns the new token ID list.  The sequence length may differ
        from the input by ±3 (3 special tokens added, no tokens removed).
        """
        n = len(ids)
        if n < 6:
            # Too short to split meaningfully — pass through as-is
            return ids

        # Pick two random split points (sorted)
        a = random.randint(1, n - 2)
        b = random.randint(a + 1, n - 1)

        prefix = ids[:a]
        middle = ids[a:b]
        suffix = ids[b:]

        if random.random() < self.spm_rate:
            # SPM: suffix-prefix-middle
            return (
                [self.suffix_id] + suffix +
                [self.prefix_id] + prefix +
                [self.middle_id] + middle +
                [self.eos_id]
            )
        else:
            # PSM: prefix-suffix-middle  (more common in literature)
            return (
                [self.prefix_id] + prefix +
                [self.suffix_id] + suffix +
                [self.middle_id] + middle +
                [self.eos_id]
            )

    # ------------------------------------------------------------------
    # Collate
    # ------------------------------------------------------------------

    def __call__(self, batch: list[tuple[torch.Tensor, torch.Tensor]]):
        """
        Args:
            batch: list of (x, y) from TokenDataset, each shape (T,)

        Returns:
            input_ids:      (B, T_max) — padded inputs
            labels:         (B, T_max) — padded labels (-100 on padding)
            attention_mask: (B, T_max) — 1 for real tokens, 0 for padding
        """
        input_seqs  = []
        label_seqs  = []

        for x, _ in batch:
            ids = x.tolist()

            if random.random() < self.fim_rate:
                # Apply FIM transform — new_ids is the full sequence to predict
                new_ids = self._fim_transform(ids)
                # For FIM samples:
                #   input  = new_ids[:-1]
                #   labels = new_ids[1:]
                inp = torch.tensor(new_ids[:-1], dtype=torch.long)
                lbl = torch.tensor(new_ids[1:],  dtype=torch.long)
            else:
                # Standard causal LM — predict every next token
                inp = x[:-1]
                lbl = x[1:]

            input_seqs.append(inp)
            label_seqs.append(lbl)

        # Pad to the longest sequence in the batch
        input_ids = pad_sequence(
            input_seqs, batch_first=True, padding_value=self.pad_id
        )
        labels = pad_sequence(
            label_seqs, batch_first=True, padding_value=-100  # CrossEntropy ignores -100
        )
        attention_mask = (input_ids != self.pad_id).long()

        return {
            "input_ids":      input_ids,
            "labels":         labels,
            "attention_mask": attention_mask,
        }


# ---------------------------------------------------------------------------
# Utility: add FIM special tokens to an existing tokenizer
# ---------------------------------------------------------------------------

def add_fim_tokens(tokenizer):
    """
    Add FIM and language-tag special tokens to a loaded tokenizer in-place.
    Call this once after loading the tokenizer, before training.

    Returns the number of tokens added (use this to resize the embedding table).
    """
    fim_tokens = [
        "<fim_prefix>",
        "<fim_suffix>",
        "<fim_middle>",
        "<|endoftext|>",
        "<|python|>",
        "<|javascript|>",
        "<|typescript|>",
        "<|cpp|>",
        "<|java|>",
        "<|go|>",
        "<|rust|>",
    ]
    # Only add tokens that aren't already present
    new_tokens = [t for t in fim_tokens if t not in tokenizer.get_vocab()]
    n_added = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    return n_added


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("FIMCollator self-test (no real tokenizer needed)...")

    # Minimal mock tokenizer
    class MockTok:
        pad_token_id = 0
        eos_token_id = 1
        unk_token_id = 3

        def convert_tokens_to_ids(self, tok):
            return {"<fim_prefix>": 10, "<fim_suffix>": 11, "<fim_middle>": 12}.get(tok, 3)

    collator = FIMCollator(MockTok(), fim_rate=1.0, spm_rate=0.5)

    # Fake batch of 4 sequences of length 32
    batch = [(torch.arange(32, dtype=torch.long), torch.arange(1, 33, dtype=torch.long))
             for _ in range(4)]

    out = collator(batch)
    print(f"  input_ids shape:      {out['input_ids'].shape}")
    print(f"  labels shape:         {out['labels'].shape}")
    print(f"  attention_mask shape: {out['attention_mask'].shape}")
    assert out["input_ids"].shape[0] == 4, "Batch size mismatch"
    assert (out["labels"] == -100).any(), "Padding labels should be -100"

    # Verify FIM tokens appear in output
    flat = out["input_ids"].flatten().tolist()
    assert 10 in flat or 11 in flat, "FIM prefix/suffix token not found in output"
    print("  FIM tokens present in output: OK")
    print("Self-test passed.")
