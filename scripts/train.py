"""
train.py — Full training loop for the Coder LLM, driven by train_config.yaml.

Features:
  - Hugging Face Accelerate for distributed / mixed-precision training
  - Gradient accumulation, gradient clipping
  - Cosine LR schedule with linear warm-up
  - Fill-In-the-Middle (FIM) training via FIMCollator
  - Periodic checkpoint saving (keeps last N)
  - Periodic validation loss evaluation
  - Optional Weights & Biases logging
  - Resume from a checkpoint

Usage:
    # Single GPU / CPU
    python scripts/train.py

    # Multi-GPU (DDP) — let accelerate handle the launch
    accelerate launch scripts/train.py

    # Override any config value
    python scripts/train.py --config configs/train_config.yaml --batch_size 8

    # Disable FIM for a plain causal LM run
    python scripts/train.py --no_fim
"""

import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import yaml

# Make project root importable from this script
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import GPT, GPTConfig
from scripts.dataset import build_dataloaders
from scripts.fim_collator import FIMCollator, add_fim_tokens


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",        type=str,   default="configs/train_config.yaml")
    p.add_argument("--resume",        type=str,   default=None,  help="Path to checkpoint dir to resume from")
    # Quick overrides
    p.add_argument("--batch_size",    type=int,   default=None)
    p.add_argument("--max_steps",     type=int,   default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--no_fim",        action="store_true",
                   help="Disable FIM training (plain causal LM)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, args) -> dict:
    """Overlay non-None CLI args onto the config dict."""
    t = cfg.setdefault("training", {})
    if args.batch_size is not None:
        t["batch_size"] = args.batch_size
    if args.max_steps is not None:
        t["max_steps"] = args.max_steps
    if args.learning_rate is not None:
        t["learning_rate"] = args.learning_rate
    return cfg


# ---------------------------------------------------------------------------
# LR schedule — cosine decay with warm-up
# ---------------------------------------------------------------------------

def get_lr(step: int, warmup_steps: int, max_steps: int, lr: float, min_lr_ratio: float = 0.1) -> float:
    min_lr = lr * min_lr_ratio
    if step < warmup_steps:
        return lr * step / max(warmup_steps, 1)
    if step >= max_steps:
        return min_lr
    # Cosine decay
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (lr - min_lr)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    accelerator,
    model,
    optimizer,
    step: int,
    val_loss: float,
    output_dir: str,
    keep_last_n: int,
):
    ckpt_dir = Path(output_dir) / f"step_{step:07d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights + config
    unwrapped = accelerator.unwrap_model(model)
    torch.save(unwrapped.state_dict(), ckpt_dir / "model.pt")
    with open(ckpt_dir / "config.json", "w") as f:
        import dataclasses
        json.dump(dataclasses.asdict(unwrapped.config), f, indent=2)

    # Save optimizer state
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")

    # Save training state (step, loss)
    with open(ckpt_dir / "trainer_state.json", "w") as f:
        json.dump({"step": step, "val_loss": val_loss}, f, indent=2)

    accelerator.print(f"Checkpoint saved: {ckpt_dir}")

    # Prune old checkpoints
    if keep_last_n > 0:
        all_ckpts = sorted(
            glob.glob(os.path.join(output_dir, "step_*")),
            key=lambda d: int(Path(d).name.split("_")[1]),
        )
        for old in all_ckpts[:-keep_last_n]:
            import shutil
            shutil.rmtree(old)
            accelerator.print(f"Removed old checkpoint: {old}")


def load_checkpoint(model, optimizer, ckpt_dir: str, device):
    ckpt_dir = Path(ckpt_dir)
    model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location=device))
    optimizer.load_state_dict(torch.load(ckpt_dir / "optimizer.pt", map_location=device))
    with open(ckpt_dir / "trainer_state.json") as f:
        state = json.load(f)
    return state["step"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val_loader, accelerator, max_batches: int = 20):
    model.eval()
    losses = []
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        # Val loader always returns plain (x, y) tuples
        if isinstance(batch, (list, tuple)):
            x, y = batch
        else:
            x, y = batch["input_ids"], batch["labels"]
        out = model(x, labels=y)
        losses.append(accelerator.gather(out["loss"]).mean().item())
    model.train()
    return sum(losses) / len(losses) if losses else float("nan")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg  = apply_overrides(load_config(args.config), args)

    # ------------------------------------------------------------------
    # Accelerate setup
    # ------------------------------------------------------------------
    try:
        from accelerate import Accelerator
        from accelerate.utils import set_seed
    except ImportError:
        print("[ERROR] accelerate not installed. Run: pip install accelerate")
        sys.exit(1)

    train_cfg = cfg["training"]
    log_cfg   = cfg.get("logging", {})
    ckpt_cfg  = cfg.get("checkpointing", {})
    data_cfg  = cfg.get("data", {})
    model_cfg_yaml = cfg.get("model", {})

    mixed_precision = train_cfg.get("mixed_precision", "bf16")
    use_wandb       = log_cfg.get("use_wandb", False)

    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        log_with="wandb" if use_wandb else None,
    )

    set_seed(train_cfg.get("seed", 42))

    accelerator.print("=" * 60)
    accelerator.print("GPT Training")
    accelerator.print(f"  Device        : {accelerator.device}")
    accelerator.print(f"  Mixed prec.   : {mixed_precision}")
    accelerator.print(f"  Num processes : {accelerator.num_processes}")
    accelerator.print("=" * 60)

    # ------------------------------------------------------------------
    # Weights & Biases
    # ------------------------------------------------------------------
    if use_wandb and accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="llm-training",
            config=cfg,
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_config = GPTConfig(
        vocab_size   = model_cfg_yaml.get("vocab_size",   32000),
        hidden_size  = model_cfg_yaml.get("hidden_size",  768),
        num_layers   = model_cfg_yaml.get("num_layers",   12),
        num_heads    = model_cfg_yaml.get("num_heads",    12),
        max_seq_len  = model_cfg_yaml.get("max_seq_len",  1024),
        dropout      = model_cfg_yaml.get("dropout",      0.1),
    )
    model = GPT(model_config)

    if train_cfg.get("gradient_checkpointing", False):
        # Manual gradient checkpointing — wrap each block's forward
        from torch.utils.checkpoint import checkpoint as torch_checkpoint

        def make_ckpt_forward(block):
            def forward_with_ckpt(x):
                # use_cache=False during training to allow checkpointing
                def _inner(x_):
                    out, _ = block(x_, use_cache=False)
                    return out
                return torch_checkpoint(_inner, x, use_reentrant=False)
            return forward_with_ckpt

        for i, block in enumerate(model.transformer.blocks):
            block._ckpt_forward = make_ckpt_forward(block)

        accelerator.print("Gradient checkpointing enabled.")

    n_params = model.num_parameters()
    accelerator.print(f"Model parameters: {n_params / 1e6:.1f}M")

    # ------------------------------------------------------------------
    # Tokenizer — needed for FIM collator special token IDs
    # ------------------------------------------------------------------
    tokenizer_path = data_cfg.get("tokenizer_path", "models/tokenizer")
    tokenizer = None
    fim_collator = None

    fim_cfg     = cfg.get("fim", {})
    use_fim     = fim_cfg.get("enabled", True) and not args.no_fim

    if use_fim:
        try:
            from transformers import PreTrainedTokenizerFast
            tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
            n_added = add_fim_tokens(tokenizer)
            if n_added > 0:
                accelerator.print(f"Added {n_added} FIM special tokens to tokenizer.")
                # Resize embedding table to cover new tokens
                model_unwrapped = accelerator.unwrap_model(model) if hasattr(model, "module") else model
                old_vocab = model_unwrapped.transformer.tok_emb.weight.shape[0]
                new_vocab = len(tokenizer)
                if new_vocab > old_vocab:
                    model_unwrapped.transformer.tok_emb = torch.nn.Embedding(
                        new_vocab, model_config.hidden_size
                    ).to(accelerator.device)
                    model_unwrapped.lm_head.weight = model_unwrapped.transformer.tok_emb.weight
                    accelerator.print(f"Resized embedding: {old_vocab} → {new_vocab}")

            fim_collator = FIMCollator(
                tokenizer,
                fim_rate=fim_cfg.get("fim_rate", 0.5),
                spm_rate=fim_cfg.get("spm_rate", 0.5),
            )
            accelerator.print(
                f"FIM training enabled  fim_rate={fim_cfg.get('fim_rate', 0.5)}  "
                f"spm_rate={fim_cfg.get('spm_rate', 0.5)}"
            )
        except Exception as e:
            accelerator.print(f"[WARN] Could not set up FIM collator: {e}")
            accelerator.print("       Falling back to plain causal LM training.")
            fim_collator = None
    else:
        accelerator.print("FIM training disabled (plain causal LM).")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    block_size  = model_config.max_seq_len
    batch_size  = train_cfg.get("batch_size", 8)
    num_workers = 0 if os.name == "nt" else 4  # Windows: use 0 workers

    train_loader, val_loader = build_dataloaders(
        data_dir       = data_cfg.get("processed_dir", "data/processed"),
        block_size     = block_size,
        batch_size     = batch_size,
        num_workers    = num_workers,
        pin_memory     = accelerator.device.type == "cuda",
        collate_fn     = fim_collator,   # None → default (x, y) collation
    )

    # ------------------------------------------------------------------
    # Optimiser — AdamW with parameter groups (no weight decay on biases/norms)
    # ------------------------------------------------------------------
    decay_params     = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay_params  = [p for n, p in model.named_parameters() if p.dim() < 2]

    optim_groups = [
        {"params": decay_params,    "weight_decay": train_cfg.get("weight_decay", 0.1)},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=train_cfg.get("learning_rate", 3e-4),
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=accelerator.device.type == "cuda",   # fused kernel when available
    )

    # ------------------------------------------------------------------
    # Wrap with Accelerate
    # ------------------------------------------------------------------
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(
            accelerator.unwrap_model(model), optimizer, args.resume, accelerator.device
        )
        accelerator.print(f"Resumed from step {start_step}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    max_steps        = train_cfg.get("max_steps", 100_000)
    warmup_steps     = train_cfg.get("warmup_steps", 2000)
    lr               = train_cfg.get("learning_rate", 3e-4)
    log_every        = log_cfg.get("log_every_steps", 50)
    save_every       = ckpt_cfg.get("save_every_steps", 1000)
    keep_last_n      = ckpt_cfg.get("keep_last_n", 3)
    output_dir       = ckpt_cfg.get("output_dir", "models/checkpoints")
    grad_clip        = 1.0
    grad_accum_steps = train_cfg.get("gradient_accumulation_steps", 1)

    model.train()
    step        = start_step
    raw_step    = 0          # counts every micro-batch
    tokens_seen = 0
    t0          = time.time()

    data_iter = iter(train_loader)
    accelerator.print(f"Starting training from step {step} / {max_steps}")

    while step < max_steps:
        # ------------------------------------------------------------------
        # Fetch next batch — cycle the dataloader if exhausted
        # ------------------------------------------------------------------
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        # ------------------------------------------------------------------
        # Unpack batch — FIM collator returns a dict; plain loader returns (x, y)
        # ------------------------------------------------------------------
        if isinstance(batch, dict):
            x              = batch["input_ids"]
            labels         = batch["labels"]
            attention_mask = batch.get("attention_mask")
        else:
            x, labels      = batch
            attention_mask = None

        # ------------------------------------------------------------------
        # Update learning rate
        # ------------------------------------------------------------------
        current_lr = get_lr(step, warmup_steps, max_steps, lr)
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        # ------------------------------------------------------------------
        # Forward + backward (gradient accumulation handled by Accelerate)
        # ------------------------------------------------------------------
        with accelerator.accumulate(model):
            out  = model(x, labels=labels)
            loss = out["loss"]
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        raw_step    += 1
        tokens_seen += x.numel() * accelerator.num_processes

        # Only advance the global step counter after a full accumulation cycle
        if raw_step % grad_accum_steps == 0:
            step += 1

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        if step % log_every == 0 and raw_step % grad_accum_steps == 0:
            dt = time.time() - t0
            tok_per_sec = tokens_seen / dt
            loss_val = accelerator.gather(loss).mean().item()

            accelerator.print(
                f"step {step:6d}/{max_steps}  "
                f"loss={loss_val:.4f}  "
                f"lr={current_lr:.2e}  "
                f"tok/s={tok_per_sec:,.0f}  "
                f"elapsed={dt/60:.1f}m"
            )

            if use_wandb:
                accelerator.log({
                    "train/loss": loss_val,
                    "train/lr":   current_lr,
                    "train/tokens_seen": tokens_seen,
                    "train/tok_per_sec": tok_per_sec,
                }, step=step)

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------
        if step % save_every == 0 and raw_step % grad_accum_steps == 0:
            val_loss = evaluate(model, val_loader, accelerator)
            accelerator.print(f"  val_loss={val_loss:.4f}")

            if use_wandb:
                accelerator.log({"val/loss": val_loss}, step=step)

            if accelerator.is_main_process:
                save_checkpoint(
                    accelerator, model, optimizer,
                    step, val_loss, output_dir, keep_last_n,
                )

    # ------------------------------------------------------------------
    # Final save
    # ------------------------------------------------------------------
    val_loss = evaluate(model, val_loader, accelerator)
    accelerator.print(f"\nTraining complete — final val_loss={val_loss:.4f}")
    if accelerator.is_main_process:
        save_checkpoint(
            accelerator, model, optimizer,
            step, val_loss, output_dir, keep_last_n=0  # keep final always
        )

    if use_wandb:
        accelerator.end_training()


if __name__ == "__main__":
    main()
