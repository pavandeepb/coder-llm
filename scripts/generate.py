"""
generate.py — Load a trained checkpoint and sample text from the model.

Usage:
    # Interactive prompt mode
    python scripts/generate.py --checkpoint models/checkpoints/step_0100000

    # One-shot generation
    python scripts/generate.py --checkpoint models/checkpoints/step_0100000 \\
        --prompt "Once upon a time" --max_new_tokens 300 --temperature 0.8

    # Greedy decode (deterministic)
    python scripts/generate.py --checkpoint models/checkpoints/step_0100000 \\
        --prompt "The theory of" --temperature 1.0 --top_k 1

    # Batch generation from a file of prompts (one per line)
    python scripts/generate.py --checkpoint models/checkpoints/step_0100000 \\
        --prompt_file prompts.txt --out_file generated.txt
"""

import argparse
import json
import sys
from pathlib import Path

import torch

# Make the project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import GPT, GPTConfig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate text from a trained GPT checkpoint.")

    p.add_argument("--checkpoint",      type=str,   required=True,
                   help="Path to a checkpoint directory (e.g. models/checkpoints/step_0100000)")
    p.add_argument("--tokenizer_path",  type=str,   default=None,
                   help="Tokenizer directory. Defaults to models/tokenizer next to checkpoint.")
    p.add_argument("--prompt",          type=str,   default="",
                   help="Text prompt to condition on (empty → unconditional <bos>).")
    p.add_argument("--prompt_file",     type=str,   default=None,
                   help="File with one prompt per line for batch generation.")
    p.add_argument("--out_file",        type=str,   default=None,
                   help="Write generated text here instead of stdout.")
    p.add_argument("--max_new_tokens",  type=int,   default=200)
    p.add_argument("--temperature",     type=float, default=0.8,
                   help="Sampling temperature (1.0 = neutral, <1.0 = sharper, >1.0 = more random).")
    p.add_argument("--top_k",           type=int,   default=50,
                   help="Keep only the top-k most likely tokens per step (0 = disabled).")
    p.add_argument("--top_p",           type=float, default=0.9,
                   help="Nucleus sampling probability threshold (1.0 = disabled).")
    p.add_argument("--repetition_penalty", type=float, default=1.1,
                   help="Penalise tokens that have already appeared (1.0 = no penalty).")
    p.add_argument("--num_samples",     type=int,   default=1,
                   help="Number of independent samples per prompt.")
    p.add_argument("--seed",            type=int,   default=None,
                   help="Random seed for reproducibility.")
    p.add_argument("--device",          type=str,   default="auto",
                   help="'auto', 'cpu', 'cuda', or 'cuda:N'.")
    p.add_argument("--compile",         action="store_true",
                   help="torch.compile the model for faster inference (PyTorch >= 2.0).")
    p.add_argument("--interactive",     action="store_true",
                   help="Run an interactive REPL: type prompts and get responses.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_model(ckpt_dir: str, device: torch.device) -> GPT:
    ckpt_dir = Path(ckpt_dir)
    config_path = ckpt_dir / "config.json"
    weights_path = ckpt_dir / "model.pt"

    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {ckpt_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(f"model.pt not found in {ckpt_dir}")

    with open(config_path) as f:
        config_dict = json.load(f)

    config = GPTConfig(**config_dict)
    model  = GPT(config)

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Loaded model from {ckpt_dir}  ({model.num_parameters() / 1e6:.1f}M params)")
    return model


def load_tokenizer(tokenizer_path: str):
    try:
        from transformers import PreTrainedTokenizerFast
    except ImportError:
        print("[ERROR] transformers not installed. Run: pip install transformers")
        sys.exit(1)

    if not Path(tokenizer_path).exists():
        print(f"[ERROR] Tokenizer not found at '{tokenizer_path}'.")
        print("        Run scripts/train_tokenizer.py to train one.")
        sys.exit(1)

    tok = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    print(f"Loaded tokenizer  vocab_size={tok.vocab_size}")
    return tok


# ---------------------------------------------------------------------------
# Generation helper
# ---------------------------------------------------------------------------

def generate_text(
    model: GPT,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    num_samples: int,
) -> list[str]:
    """
    Encode `prompt`, run model.generate() for `num_samples` independent
    continuations, decode and return them as strings.
    """
    eos_id = tokenizer.eos_token_id

    if prompt.strip():
        input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    else:
        # Unconditional: start with <bos>
        bos_id = tokenizer.bos_token_id or 0
        input_ids = torch.tensor([[bos_id]], dtype=torch.long)

    input_ids = input_ids.to(device)
    prompt_len = input_ids.shape[1]

    results = []
    for i in range(num_samples):
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=eos_id,
            )

        # Decode only the newly generated tokens
        new_ids   = output_ids[0, prompt_len:].tolist()
        generated = tokenizer.decode(new_ids, skip_special_tokens=True)
        results.append(generated)

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

SEPARATOR = "─" * 60

def print_result(prompt: str, samples: list[str], sample_idx_offset: int = 0):
    print(f"\n{SEPARATOR}")
    if prompt.strip():
        print(f"PROMPT: {prompt}")
    else:
        print("PROMPT: (unconditional)")
    for i, text in enumerate(samples):
        n = sample_idx_offset + i + 1
        label = f"[Sample {n}]" if len(samples) > 1 or sample_idx_offset > 0 else "[Generated]"
        print(f"\n{label}\n{text}")
    print(SEPARATOR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model = load_model(args.checkpoint, device)

    if args.compile:
        if hasattr(torch, "compile"):
            model = torch.compile(model)
            print("Model compiled with torch.compile.")
        else:
            print("torch.compile not available (need PyTorch >= 2.0). Skipping.")

    # ------------------------------------------------------------------
    # Load tokenizer
    # ------------------------------------------------------------------
    if args.tokenizer_path:
        tok_path = args.tokenizer_path
    else:
        # Convention: tokenizer lives next to the checkpoints folder
        tok_path = str(Path(args.checkpoint).parent.parent / "tokenizer")
        if not Path(tok_path).exists():
            tok_path = "models/tokenizer"

    tokenizer = load_tokenizer(tok_path)

    # ------------------------------------------------------------------
    # Collect prompts
    # ------------------------------------------------------------------
    prompts: list[str] = []

    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(prompts)} prompt(s) from {args.prompt_file}")
    elif not args.interactive:
        prompts = [args.prompt]

    # ------------------------------------------------------------------
    # Output file
    # ------------------------------------------------------------------
    out_fh = open(args.out_file, "w", encoding="utf-8") if args.out_file else None

    def emit(text: str):
        if out_fh:
            out_fh.write(text + "\n")
        else:
            print(text, end="")

    # ------------------------------------------------------------------
    # Batch mode
    # ------------------------------------------------------------------
    if prompts:
        for prompt in prompts:
            samples = generate_text(
                model, tokenizer, prompt, device,
                args.max_new_tokens, args.temperature,
                args.top_k, args.top_p, args.repetition_penalty,
                args.num_samples,
            )
            if out_fh:
                emit(f"=== PROMPT: {prompt} ===\n")
                for s in samples:
                    emit(s + "\n\n")
            else:
                print_result(prompt, samples)

    # ------------------------------------------------------------------
    # Interactive REPL
    # ------------------------------------------------------------------
    if args.interactive:
        print("\nInteractive generation mode. Type your prompt and press Enter.")
        print("Commands: :quit  :temp <float>  :tokens <int>  :samples <int>")
        print(SEPARATOR)

        temperature     = args.temperature
        max_new_tokens  = args.max_new_tokens
        num_samples     = args.num_samples
        sample_counter  = 0

        while True:
            try:
                prompt = input("\nPrompt> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not prompt:
                continue

            # Simple REPL commands
            if prompt == ":quit":
                break
            if prompt.startswith(":temp "):
                try:
                    temperature = float(prompt.split()[1])
                    print(f"  temperature → {temperature}")
                except ValueError:
                    print("  Usage: :temp <float>")
                continue
            if prompt.startswith(":tokens "):
                try:
                    max_new_tokens = int(prompt.split()[1])
                    print(f"  max_new_tokens → {max_new_tokens}")
                except ValueError:
                    print("  Usage: :tokens <int>")
                continue
            if prompt.startswith(":samples "):
                try:
                    num_samples = int(prompt.split()[1])
                    print(f"  num_samples → {num_samples}")
                except ValueError:
                    print("  Usage: :samples <int>")
                continue

            samples = generate_text(
                model, tokenizer, prompt, device,
                max_new_tokens, temperature,
                args.top_k, args.top_p, args.repetition_penalty,
                num_samples,
            )
            print_result(prompt, samples, sample_counter)
            sample_counter += num_samples

    if out_fh:
        out_fh.close()
        print(f"\nOutput written to {args.out_file}")


if __name__ == "__main__":
    main()
