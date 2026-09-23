"""
verify_setup.py
Run this after setup_env.sh to confirm PyTorch sees your GPU and the rest
of the training stack is importable.

    python scripts/verify_setup.py
"""

import sys


def check(label, fn):
    try:
        result = fn()
        print(f"[OK]   {label}: {result}")
        return True
    except Exception as e:
        print(f"[FAIL] {label}: {e}")
        return False


def main():
    print("=" * 60)
    print("Environment verification")
    print("=" * 60)

    ok = True

    ok &= check("Python version", lambda: sys.version.split()[0])

    import torch
    ok &= check("PyTorch version", lambda: torch.__version__)
    ok &= check("CUDA available", lambda: torch.cuda.is_available())

    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        ok &= check("GPU count", lambda: n)
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            print(f"[OK]   GPU {i}: {props.name} "
                  f"({props.total_memory / 1024**3:.1f} GB, "
                  f"compute capability {props.major}.{props.minor})")
        ok &= check("CUDA version (torch build)", lambda: torch.version.cuda)
        ok &= check("cuDNN version", lambda: torch.backends.cudnn.version())

        # Quick real op on the GPU, not just a flag check
        def matmul_test():
            a = torch.randn(2048, 2048, device="cuda")
            b = torch.randn(2048, 2048, device="cuda")
            c = a @ b
            torch.cuda.synchronize()
            return f"{tuple(c.shape)} on {c.device}"
        ok &= check("GPU matmul smoke test", matmul_test)
    else:
        print("[WARN] No CUDA device detected — training will run on CPU "
              "(fine for testing, far too slow for real training).")

    for pkg in ["transformers", "datasets", "tokenizers", "accelerate",
                "numpy", "matplotlib", "tqdm"]:
        ok &= check(f"{pkg} import", lambda p=pkg: __import__(p).__version__)

    print("=" * 60)
    print("All checks passed." if ok else "Some checks failed — see [FAIL] lines above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
