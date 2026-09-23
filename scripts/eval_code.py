"""
eval_code.py — Evaluate a trained coder LLM on coding benchmarks.

Metrics implemented:
  1. Syntax check pass rate     — fraction of generated snippets that parse
  2. HumanEval pass@k           — fraction of HumanEval problems solved in k tries
  3. Perplexity on val set      — language model quality metric

HumanEval is the standard benchmark for code LLMs (164 Python problems).
Download it first:
    python scripts/download_data.py --dataset humaneval

Usage:
    # Full eval (syntax + HumanEval + perplexity)
    python scripts/eval_code.py --checkpoint models/checkpoints/step_0100000

    # Only syntax check on a prompt file
    python scripts/eval_code.py --checkpoint models/checkpoints/step_0100000 \\
        --mode syntax --prompt_file prompts.txt

    # HumanEval pass@1 and pass@10
    python scripts/eval_code.py --checkpoint models/checkpoints/step_0100000 \\
        --mode humaneval --n_samples 20 --k 1 5 10
"""

import argparse
import ast
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from model import GPT, GPTConfig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a coder LLM checkpoint.")
    p.add_argument("--checkpoint",      type=str,   required=True)
    p.add_argument("--tokenizer_path",  type=str,   default=None)
    p.add_argument("--config",          type=str,   default="configs/train_config.yaml")
    p.add_argument("--mode",            type=str,   default="all",
                   choices=["all", "syntax", "humaneval", "perplexity"])
    p.add_argument("--humaneval_path",  type=str,   default="data/HumanEval.jsonl")
    p.add_argument("--data_dir",        type=str,   default="data/processed",
                   help="Processed data dir for perplexity eval")
    p.add_argument("--n_samples",       type=int,   default=20,
                   help="Completions generated per HumanEval problem")
    p.add_argument("--k",               type=int,   nargs="+", default=[1, 5, 10],
                   help="k values for pass@k")
    p.add_argument("--temperature",     type=float, default=0.2)
    p.add_argument("--top_p",           type=float, default=0.95)
    p.add_argument("--max_new_tokens",  type=int,   default=512)
    p.add_argument("--timeout",         type=int,   default=10,
                   help="Seconds per test-case execution")
    p.add_argument("--prompt_file",     type=str,   default=None)
    p.add_argument("--device",          type=str,   default="auto")
    p.add_argument("--seed",            type=int,   default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model / tokenizer loading  (shared with generate.py)
# ---------------------------------------------------------------------------

def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_model(ckpt_dir: str, device: torch.device) -> GPT:
    ckpt_dir = Path(ckpt_dir)
    with open(ckpt_dir / "config.json") as f:
        config_dict = json.load(f)
    config = GPTConfig(**config_dict)
    model  = GPT(config)
    state  = torch.load(ckpt_dir / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"Loaded: {ckpt_dir}  ({model.num_parameters()/1e6:.1f}M params)")
    return model


def load_tokenizer(path: str):
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast.from_pretrained(path)
    print(f"Tokenizer: {path}  vocab={tok.vocab_size}")
    return tok


# ---------------------------------------------------------------------------
# 1. Syntax check
# ---------------------------------------------------------------------------

def check_syntax(code: str) -> tuple[bool, str]:
    """Return (is_valid, error_message). Checks Python syntax only."""
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, str(e)


def eval_syntax(model, tokenizer, device, prompts: list[str],
                max_new_tokens: int, temperature: float, top_p: float) -> dict:
    """
    Generate one completion per prompt, check if it's valid Python.
    Returns a results dict.
    """
    n_valid = 0
    results = []
    bos_id = tokenizer.bos_token_id or 0

    for prompt in prompts:
        if prompt.strip():
            ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt").to(device)
        else:
            ids = torch.tensor([[bos_id]], dtype=torch.long, device=device)

        with torch.no_grad():
            out_ids = model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=0,
            )

        generated = tokenizer.decode(out_ids[0, ids.shape[1]:], skip_special_tokens=True)
        is_valid, err = check_syntax(generated)
        n_valid += int(is_valid)
        results.append({"prompt": prompt, "generated": generated,
                         "syntax_ok": is_valid, "error": err})
        status = "OK" if is_valid else f"FAIL: {err}"
        print(f"  [{status}] {prompt[:60]!r}")

    pass_rate = n_valid / len(prompts) if prompts else 0.0
    print(f"\nSyntax pass rate: {n_valid}/{len(prompts)}  ({pass_rate*100:.1f}%)")
    return {"pass_rate": pass_rate, "n_valid": n_valid, "n_total": len(prompts), "details": results}


# ---------------------------------------------------------------------------
# 2. HumanEval  pass@k
# ---------------------------------------------------------------------------

def load_humaneval(path: str) -> list[dict]:
    problems = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


def _run_code_with_tests(code: str, test: str, timeout: int) -> bool:
    """
    Execute `code` + `test` in a fresh subprocess.
    Returns True if the code exits 0 within `timeout` seconds.
    """
    full_code = code + "\n\n" + test + "\n\ncheck(candidate)\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                     delete=False, encoding="utf-8") as f:
        f.write(full_code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased estimator of pass@k from n total samples with c correct ones.
    Formula from Chen et al. 2021 (HumanEval paper).
    """
    if n - c < k:
        return 1.0
    # Use log space to avoid overflow for large n
    # pass@k = 1 - C(n-c, k) / C(n, k)
    import math
    log_numerator   = sum(math.log(n - c - i) for i in range(k))
    log_denominator = sum(math.log(n - i) for i in range(k))
    return 1.0 - math.exp(log_numerator - log_denominator)


def eval_humaneval(
    model, tokenizer, device,
    problems: list[dict],
    n_samples: int,
    k_values: list[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    timeout: int,
) -> dict:
    """
    For each HumanEval problem generate `n_samples` completions,
    run the test suite, and compute pass@k.
    """
    all_results = []
    total_problems = len(problems)

    print(f"\nEvaluating {total_problems} HumanEval problems  "
          f"({n_samples} samples each, k={k_values})...")
    print("NOTE: This executes generated code in subprocesses. "
          "Only run on trusted checkpoints.")

    for pi, problem in enumerate(problems):
        task_id  = problem["task_id"]
        prompt   = problem["prompt"]          # function signature + docstring
        test     = problem["test"]            # assert-based test suite
        entry    = problem["entry_point"]     # function name

        # Encode prompt
        ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)

        n_correct = 0
        completions = []

        for s in range(n_samples):
            with torch.no_grad():
                out_ids = model.generate(
                    ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=0,
                    eos_token_id=tokenizer.eos_token_id,
                )
            completion = tokenizer.decode(out_ids[0, ids.shape[1]:], skip_special_tokens=True)

            # Stop at first natural break (double newline after function body)
            # so we don't pass extra code to the test harness
            if "\n\n" in completion:
                completion = completion.split("\n\n")[0]

            full_solution = prompt + completion
            passed = _run_code_with_tests(full_solution, test, timeout)
            n_correct += int(passed)
            completions.append({"completion": completion, "passed": passed})

        problem_results = {
            "task_id":    task_id,
            "n_samples":  n_samples,
            "n_correct":  n_correct,
            "completions": completions,
        }
        all_results.append(problem_results)

        # Running pass@1 estimate
        p1 = _estimate_pass_at_k(n_samples, n_correct, 1)
        print(f"  [{pi+1:3d}/{total_problems}] {task_id}  "
              f"correct={n_correct}/{n_samples}  pass@1={p1:.3f}")

    # Aggregate pass@k across all problems
    pass_at_k = {}
    for k in k_values:
        if k > n_samples:
            print(f"  [SKIP] pass@{k} requires n_samples >= {k}")
            continue
        estimates = [
            _estimate_pass_at_k(r["n_samples"], r["n_correct"], k)
            for r in all_results
        ]
        pass_at_k[f"pass@{k}"] = sum(estimates) / len(estimates)

    print("\n=== HumanEval Results ===")
    for metric, value in pass_at_k.items():
        print(f"  {metric}: {value:.4f}  ({value*100:.1f}%)")

    return {"pass_at_k": pass_at_k, "problem_results": all_results}


# ---------------------------------------------------------------------------
# 3. Perplexity on validation set
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_perplexity(model, data_dir: str, block_size: int,
                    device: torch.device, max_batches: int = 50) -> dict:
    """
    Compute perplexity on val.bin.
    perplexity = exp(mean cross-entropy loss)
    """
    from scripts.dataset import TokenDataset
    from torch.utils.data import DataLoader

    try:
        ds = TokenDataset(data_dir, split="val", block_size=block_size)
    except FileNotFoundError as e:
        print(f"[SKIP] Perplexity eval: {e}")
        return {}

    loader = DataLoader(ds, batch_size=4, shuffle=False,
                        num_workers=0, pin_memory=False)

    losses = []
    model.eval()
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        out = model(x, labels=y)
        losses.append(out["loss"].item())

    avg_loss = sum(losses) / len(losses) if losses else float("nan")
    ppl      = math.exp(avg_loss) if avg_loss < 300 else float("inf")
    print(f"\nPerplexity (val): {ppl:.2f}  (avg loss={avg_loss:.4f})")
    return {"val_loss": avg_loss, "perplexity": ppl}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device    = resolve_device(args.device)
    model     = load_model(args.checkpoint, device)
    block_size = model.config.max_seq_len

    # Tokenizer path resolution
    if args.tokenizer_path:
        tok_path = args.tokenizer_path
    else:
        tok_path = str(Path(args.checkpoint).parent.parent / "tokenizer")
        if not Path(tok_path).exists():
            tok_path = "models/tokenizer"
    tokenizer = load_tokenizer(tok_path)

    results = {}

    # ------------------------------------------------------------------
    # Syntax eval
    # ------------------------------------------------------------------
    if args.mode in ("all", "syntax"):
        print("\n=== Syntax Check ===")
        if args.prompt_file and Path(args.prompt_file).exists():
            prompts = Path(args.prompt_file).read_text().splitlines()
            prompts = [p.strip() for p in prompts if p.strip()]
        else:
            # Default sanity prompts
            prompts = [
                "def fibonacci(n):\n    ",
                "def binary_search(arr, target):\n    ",
                "class Stack:\n    def __init__(self):\n        ",
                "import os\ndef list_files(path):\n    ",
                "def merge_sort(arr):\n    ",
            ]
        results["syntax"] = eval_syntax(
            model, tokenizer, device, prompts,
            args.max_new_tokens, args.temperature, args.top_p,
        )

    # ------------------------------------------------------------------
    # HumanEval
    # ------------------------------------------------------------------
    if args.mode in ("all", "humaneval"):
        print("\n=== HumanEval ===")
        he_path = args.humaneval_path
        if not Path(he_path).exists():
            print(f"[SKIP] HumanEval file not found: {he_path}")
            print("       Run: python scripts/download_data.py --dataset humaneval")
        else:
            problems = load_humaneval(he_path)
            results["humaneval"] = eval_humaneval(
                model, tokenizer, device, problems,
                n_samples    = args.n_samples,
                k_values     = args.k,
                max_new_tokens = args.max_new_tokens,
                temperature  = args.temperature,
                top_p        = args.top_p,
                timeout      = args.timeout,
            )

    # ------------------------------------------------------------------
    # Perplexity
    # ------------------------------------------------------------------
    if args.mode in ("all", "perplexity"):
        print("\n=== Perplexity ===")
        results["perplexity"] = eval_perplexity(
            model, args.data_dir, block_size, device
        )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_path = Path(args.checkpoint) / "eval_results.json"
    with open(out_path, "w") as f:
        # Convert any non-serialisable values
        def _default(o):
            if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
                return str(o)
            raise TypeError(f"Not serialisable: {type(o)}")
        json.dump(results, f, indent=2, default=_default)

    print(f"\nResults saved to {out_path}")

    # Print summary
    print("\n=== Summary ===")
    if "syntax" in results:
        r = results["syntax"]
        print(f"  Syntax pass rate : {r['pass_rate']*100:.1f}%")
    if "humaneval" in results:
        for metric, val in results["humaneval"]["pass_at_k"].items():
            print(f"  HumanEval {metric:8s}: {val*100:.1f}%")
    if "perplexity" in results:
        print(f"  Perplexity (val) : {results['perplexity'].get('perplexity', 'N/A'):.2f}")


if __name__ == "__main__":
    main()
