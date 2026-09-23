# Coder LLM — Build Your Own Code Generation AI From Scratch

A complete, end-to-end pipeline to train a **GPT-style language model that generates code**, built entirely from scratch using PyTorch. No pre-trained weights, no fine-tuning — raw code in, a working AI code model out.

Inspired by how GitHub Copilot works, this project teaches the model to complete code at the cursor position using a technique called **Fill-In-the-Middle (FIM)** training.

---

## What This Project Does

You feed it code files. It trains a neural network to understand and generate code. After training, you can give it a function signature and it will write the body. Give it a partial file and it will fill in the gap.

```
Input prompt:
  def binary_search(arr, target):

Model output:
      left, right = 0, len(arr) - 1
      while left <= right:
          mid = (left + right) // 2
          if arr[mid] == target:
              return mid
          elif arr[mid] < target:
              left = mid + 1
          else:
              right = mid - 1
      return -1
```

---

## Hardware This Was Built For

| Component | Spec |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 6 GB Laptop |
| RAM | 16 GB |
| OS | Windows 11 |
| Python | 3.11+ |

Every setting in this project — batch size, sequence length, model size, mixed precision — is tuned specifically for a **6 GB VRAM** GPU. It will not OOM. If you have a bigger GPU, see the scaling section at the bottom.

---

## How It Works — The Big Picture

```
Your code files (.py, .js, .ts, .java, .cpp ...)
        │
        ▼
  [1] Download Data          scripts/download_data.py
        │  Pull code from GitHub (CodeParrot, CodeSearchNet)
        │
        ▼
  [2] Train Tokenizer        scripts/train_tokenizer.py
        │  Learn a vocabulary of 32,000 subword tokens from your code
        │  Add special tokens: <fim_prefix> <fim_suffix> <fim_middle>
        │
        ▼
  [3] Preprocess Data        scripts/preprocess_data.py
        │  Convert all code files → flat sequence of token IDs
        │  Save as memory-mapped binary (train.bin, val.bin)
        │  Never loads full dataset into RAM
        │
        ▼
  [4] Train                  scripts/train.py
        │  42M parameter GPT transformer learns to predict next token
        │  50% of batches use FIM transform (fill-in-the-middle)
        │  Checkpoints saved every 2000 steps
        │
        ▼
  [5] Generate               scripts/generate.py
        │  Load a checkpoint, give it a prompt, get code back
        │
        ▼
  [6] Evaluate               scripts/eval_code.py
        │  HumanEval pass@k benchmark
        │  Syntax check pass rate
        │  Perplexity on validation set
```

---

## Project Structure

```
llm-training-project/
│
├── model.py                    The GPT model — every layer built from scratch
├── start.bat                   Double-click to run the full pipeline
├── requirements.txt            Python dependencies
├── setup_env.sh                Linux/Mac environment setup script
│
├── configs/
│   └── train_config.yaml       All settings in one place — edit this to change anything
│
├── scripts/
│   ├── download_data.py        Fetch code datasets from HuggingFace
│   ├── train_tokenizer.py      Train a BPE tokenizer on your code
│   ├── preprocess_data.py      Tokenise code files → binary shards
│   ├── dataset.py              PyTorch Dataset — reads .bin files via memmap
│   ├── fim_collator.py         Fill-In-the-Middle data collator
│   ├── train.py                Full training loop
│   ├── generate.py             Code generation / interactive REPL
│   ├── eval_code.py            Benchmark evaluation
│   └── verify_setup.py         Check GPU + libraries are working
│
├── data/
│   ├── raw/                    Downloaded code files go here
│   │   └── codeparrot-small/   Python files from GitHub (~80k files, ~1 GB)
│   ├── processed/              Tokenised binary shards
│   │   ├── train.bin           Training token IDs (uint16)
│   │   ├── val.bin             Validation token IDs (uint16)
│   │   └── meta.json           Vocab size, token counts, dtype
│   └── HumanEval.jsonl         164 Python eval problems
│
├── models/
│   ├── tokenizer/              Trained BPE tokenizer (HuggingFace format)
│   └── checkpoints/            Saved model checkpoints
│       └── step_NNNNNNN/
│           ├── model.pt
│           ├── config.json
│           ├── optimizer.pt
│           └── trainer_state.json
│
└── logs/                       Training logs
```

---

## The Model — What's Inside

The model (`model.py`) is a **decoder-only GPT transformer** — the same fundamental architecture as GPT-2, GPT-3, and CodeLlama, built from scratch.

### Architecture Choices

| Component | What We Use | Why |
|---|---|---|
| Attention | Multi-head causal self-attention | Standard for autoregressive generation |
| QKV | Fused single projection | Faster, less memory |
| Flash Attention | `scaled_dot_product_attention` | PyTorch 2.0+ uses Flash Attention automatically |
| Activation | SwiGLU | Outperforms GELU at same parameter count |
| Normalisation | RMSNorm (pre-norm) | More stable training than post-norm LayerNorm |
| Positional encoding | Learned absolute embeddings | Simple, works well at 1024 context |
| Weight tying | Embedding ↔ LM head | Saves ~60M params, improves generalisation |
| Init | GPT-2 scheme with residual scaling | Stable training from random init |

### Model Size (tuned for 6 GB GPU)

```
hidden_size  : 512
num_layers   : 8
num_heads    : 8
head_dim     : 64   (512 / 8)
max_seq_len  : 1024 tokens
vocab_size   : 32,000
dropout      : 0.0  (no dropout — regularise via data diversity instead)

Total params : ~42 Million
VRAM usage   : ~1.5–2.5 GB  (leaves headroom for activations + optimizer)
```

### How a Forward Pass Works

```
Input token IDs  [523, 41, 892, 7, ...]   shape: (batch, seq_len)
        │
        ▼
Token Embedding   +   Positional Embedding     → (batch, seq_len, 512)
        │
        ▼
  ┌─── Transformer Block × 8 ──────────────────────────────┐
  │                                                         │
  │   RMSNorm → Causal Self-Attention → residual add        │
  │   RMSNorm → SwiGLU MLP           → residual add        │
  │                                                         │
  └─────────────────────────────────────────────────────────┘
        │
        ▼
  Final RMSNorm
        │
        ▼
  LM Head (Linear, weight-tied to embedding)
        │
        ▼
  Logits   (batch, seq_len, 32000)   — one score per token in vocab
        │
        ▼
  Cross-Entropy Loss  (comparing predicted next token vs actual next token)
```

---

## Fill-In-the-Middle (FIM) Training

This is the key feature that makes the model useful for code completion — not just end-of-file generation.

### What FIM Does

Normal LM training only teaches the model to predict what comes **after** the cursor.
FIM training also teaches it to predict what goes **between** two pieces of code.

```
Normal training:
  Input:   def add(a, b):
  Predict:     return a + b          ← only completion at end

FIM training:
  Input:   def add(a, b):
           ___FILL_HERE___
           result = add(1, 2)
  Predict: return a + b              ← completion IN THE MIDDLE
```

### How FIM Works in the Data

During training, 50% of code samples are randomly transformed:

```
Original:  [PREFIX tokens] [MIDDLE tokens] [SUFFIX tokens]

PSM format:
  <fim_prefix> PREFIX <fim_suffix> SUFFIX <fim_middle> MIDDLE <eos>

SPM format (50% of FIM samples):
  <fim_suffix> SUFFIX <fim_prefix> PREFIX <fim_middle> MIDDLE <eos>
```

The model learns that when it sees `<fim_middle>`, it must predict the missing code given both what came before and what comes after. This is implemented in `scripts/fim_collator.py`.

### Using FIM at Generation Time

```python
prefix = "def factorial(n):\n    "
suffix = "\n\nprint(factorial(5))"
prompt = f"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>"
# Feed this to generate.py → model fills in the function body
```

---

## The Tokenizer

Built with HuggingFace `tokenizers`, using **Byte-Pair Encoding (BPE)** — the same algorithm used by GPT-2, GPT-4, and CodeLlama.

### What BPE Does

Starts with individual bytes, then merges the most frequent pairs repeatedly until it reaches the target vocabulary size (32,000 tokens).

```
"return"     → single token  [8901]         (common keyword)
"def"        → single token  [523]
"fibonacci"  → two tokens    [1284, 8823]   (less common)
"__init__"   → two tokens    [201, 7341]
" "×4        → single token  [198]          (4-space indent is one token)
```

### Special Tokens Added

```
<unk>           Unknown token (fallback)
<pad>           Padding
<bos>           Begin of sequence
<eos>           End of sequence
<fim_prefix>    FIM: start of prefix section
<fim_suffix>    FIM: start of suffix section
<fim_middle>    FIM: model generates from here
<|endoftext|>   Document separator (between files)
<|python|>      Language tag — prepended to Python files
<|javascript|>  Language tag — prepended to JS files
<|typescript|>  Language tag — prepended to TS files
<|cpp|>         Language tag — prepended to C/C++ files
<|java|>        Language tag — prepended to Java files
<|go|>          Language tag — prepended to Go files
<|rust|>        Language tag — prepended to Rust files
```

Language tags help the model understand which language it's working in, allowing it to switch styles appropriately.

---

## The Training Data

### What Was Downloaded

| Dataset | Source | Content | Size |
|---|---|---|---|
| codeparrot-clean | `codeparrot/codeparrot-clean` | Cleaned Python from GitHub | ~400 MB |
| CodeSearchNet | `code-search-net/code_search_net` | Function + docstring pairs | ~600 MB |
| HumanEval | OpenAI | 164 Python eval problems | tiny |

**Total: ~1 GB of code** — appropriate for a 42M model. Larger models need more data.

### How Data Is Preprocessed

Each code file gets:
1. A language tag prepended (`<|python|>`)
2. Tokenised (converted to integer IDs)
3. An `<|endoftext|>` token appended (marks end of document)
4. Appended to a flat sequence

The result is one giant sequence of token IDs, split 99.5% / 0.5% into `train.bin` and `val.bin`. These are memory-mapped — the OS pages in only the windows being trained on, so 1 GB of data never loads fully into RAM.

---

## Training

### The Training Loop (`scripts/train.py`)

```
For each step (up to 50,000):

  1. Sample a random 1024-token window from train.bin
  2. Apply FIM transform (50% chance)
  3. Forward pass → compute loss
  4. Backward pass → compute gradients
  5. Accumulate gradients for 16 steps
  6. Clip gradients to max norm 1.0
  7. AdamW optimizer step → update 42M weights
  8. Adjust learning rate (cosine schedule)

Every 50 steps:   print loss, lr, tokens/sec
Every 2000 steps: evaluate val loss → save checkpoint
```

### Learning Rate Schedule

```
Steps 0 → 500:    Linear warmup    0 → 3e-4
Steps 500 → 50k:  Cosine decay     3e-4 → 3e-5
```

Warmup prevents the model from making huge destructive updates at the start when gradients are noisy.

### Why Gradient Accumulation

The GPU can only fit `batch_size=2` sequences at once (limited VRAM). But training is more stable with larger batches. So we run 16 micro-batches and accumulate their gradients before updating, giving an effective batch of 32 sequences = ~32,000 tokens per step.

### Mixed Precision (fp16)

Weights and activations stored as 16-bit floats during the forward/backward pass. This halves VRAM usage compared to fp32. The master weights and optimizer states stay in fp32 for numerical stability.

### What the Loss Means

Loss is cross-entropy: how surprised the model is by the actual next token.

```
Loss ~9-10    Random guessing (where training starts)
Loss ~5-6     Learned basic token frequencies
Loss ~3-4     Learned Python keywords, indentation, common patterns
Loss ~2-2.5   Writing plausible function bodies
Loss ~1.8     Good code completions — this is the target
```

### Training Time on RTX 3050 6GB

```
~1,500–2,000 tokens/second

1,000 steps  ≈  30 minutes   (verify it works)
5,000 steps  ≈  2.5 hours    (basic Python structure)
10,000 steps ≈  5 hours      (decent completions, worth testing)
50,000 steps ≈  24 hours     (full training run)
```

---

## Generating Code

### One-shot completion

```powershell
python scripts/generate.py \
  --checkpoint models/checkpoints/step_0050000 \
  --prompt "def quicksort(arr):" \
  --temperature 0.2
```

### Interactive mode (like a REPL)

```powershell
python scripts/generate.py \
  --checkpoint models/checkpoints/step_0050000 \
  --interactive
```

Type a prompt, press Enter, get a completion. While in the REPL:

```
:temp 0.1       more deterministic output
:temp 0.8       more creative/varied output
:tokens 512     change max completion length
:samples 3      generate 3 variations at once
:quit           exit
```

### Temperature Guide for Code

```
0.1 - 0.3    Best for code  — deterministic, picks most likely tokens
0.5 - 0.7    Some variation — good for exploring alternatives
0.8 - 1.0    Creative but risky — may hallucinate APIs
```

### Resume Training

```powershell
python scripts/train.py --resume models/checkpoints/step_0010000
```

The optimizer state is saved with each checkpoint, so training resumes exactly where it left off.

---

## Evaluating the Model

```powershell
python scripts/eval_code.py --checkpoint models/checkpoints/step_0050000
```

Three metrics are computed:

### 1. Syntax Pass Rate
Generates completions for common Python prompts, checks if they parse without `SyntaxError`. A basic sanity check.

### 2. HumanEval pass@k
The standard benchmark for code LLMs (164 Python problems from OpenAI).
- Generates `n` completions per problem
- Runs each through the actual test suite in a subprocess
- Reports the fraction of problems solved in `k` tries

```
pass@1    Did it solve it first try?
pass@5    Did any of 5 attempts solve it?
pass@10   Did any of 10 attempts solve it?
```

### 3. Perplexity
`exp(val_loss)` — how "surprised" the model is by unseen code. Lower is better.

### Realistic Targets for 42M Model

```
HumanEval pass@1:    3–8%     (GPT-2 level, mostly syntax-correct)
Syntax pass rate:    60–80%
Perplexity:          ~8–15
```

Not GPT-4, but a real, working code model you trained yourself from nothing.

---

## Configuration Reference

Everything lives in `configs/train_config.yaml`. Key settings:

```yaml
model:
  vocab_size: 32000       # must match tokenizer
  hidden_size: 512        # model width — main VRAM knob
  num_layers: 8           # model depth
  num_heads: 8            # attention heads
  max_seq_len: 1024       # context window

training:
  batch_size: 2           # per-GPU — lower if OOM
  gradient_accumulation_steps: 16   # raise if you lower batch_size
  learning_rate: 3.0e-4
  max_steps: 50000
  mixed_precision: fp16   # fp16 for RTX 3050; bf16 for RTX 30xx desktop+
  gradient_checkpointing: true    # saves ~40% VRAM, keep this on

fim:
  enabled: true
  fim_rate: 0.5           # fraction of batches that use FIM
```

---

## Quickstart — Complete Commands

```powershell
cd f:\llm-training-project\llm-training-project

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# Verify GPU
python scripts/verify_setup.py

# Download data (~1 GB, no login needed)
python scripts/download_data.py --dataset all-small

# Train tokenizer
python scripts/train_tokenizer.py

# Preprocess  (--num_workers 0 required on Windows)
python scripts/preprocess_data.py --num_workers 0

# Quick test run — 30 min, just verify it trains
python scripts/train.py --max_steps 1000

# Full training — leave overnight
python scripts/train.py

# OR — just double-click:
start.bat
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `CUDA out of memory` | Set `batch_size: 1` and `gradient_accumulation_steps: 32` in config |
| `fp16 loss is NaN` | Lower `learning_rate` to `1.0e-4` |
| `UnicodeDecodeError` on Windows | Already fixed — all file opens use `encoding="utf-8"` |
| `numpy.uint16` dtype error | Already fixed — `_parse_dtype()` handles all formats |
| Multiprocessing hang on Windows | Always use `--num_workers 0` |
| `the-stack-smol` gated error | Use `--dataset codeparrot-small` instead (no login) |
| Loss not decreasing after 1000 steps | Check data — run `python scripts/preprocess_data.py` and verify token count > 1M |
| Very slow training | Close Chrome GPU acceleration, Discord, other GPU apps |

---

## Scaling Up (Future)

If you get a bigger GPU later, just edit `train_config.yaml`:

```yaml
# 85M model — needs ~10 GB VRAM
hidden_size: 768
num_layers: 12
num_heads: 12
max_seq_len: 2048
batch_size: 4

# 350M model — needs ~24 GB VRAM
hidden_size: 1024
num_layers: 24
num_heads: 16
max_seq_len: 4096
batch_size: 8
```

More data helps too — `--dataset codeparrot --max_gb 50` streams up to 50 GB of Python.

---

## Key Files Explained

| File | What it does |
|---|---|
| `model.py` | The entire GPT architecture: `GPTConfig`, `RMSNorm`, `CausalSelfAttention`, `MLP`, `TransformerBlock`, `GPT`. The `generate()` method handles sampling with KV-cache. |
| `scripts/train_tokenizer.py` | Trains BPE on code files. Adds FIM + language tag tokens. Saves HuggingFace-compatible tokenizer. |
| `scripts/preprocess_data.py` | Walks `data/raw/`, filters by extension, prepends language tags, inserts `<\|endoftext\|>` between files, tokenises, writes flat binary. |
| `scripts/dataset.py` | `TokenDataset` reads `train.bin`/`val.bin` via `np.memmap`. Each sample is a sliding window of 1024 tokens. Never loads full data into RAM. |
| `scripts/fim_collator.py` | `FIMCollator` — PyTorch `collate_fn` that randomly applies PSM/SPM FIM transforms to batches during training. |
| `scripts/train.py` | Full training loop: Accelerate, cosine LR, AdamW with param groups, grad clipping, checkpoint save/resume/prune, val evaluation, optional W&B. |
| `scripts/generate.py` | Loads a checkpoint, encodes a prompt, runs `model.generate()` with top-k/nucleus sampling and KV-cache. Supports batch + interactive modes. |
| `scripts/eval_code.py` | Syntax check, HumanEval pass@k (executes generated code in subprocess), perplexity. |
| `start.bat` | Checks each step, skips completed ones, launches training. Double-click to run everything. |

---

## Dependencies

```
torch >= 2.0          Deep learning framework + Flash Attention
transformers >= 4.44  Tokenizer loading/saving
tokenizers >= 0.19    Fast BPE tokenizer training
datasets >= 2.20      HuggingFace dataset streaming
accelerate >= 0.33    Multi-GPU + mixed precision training
numpy >= 1.26         Memory-mapped binary arrays
tqdm >= 4.66          Progress bars
pyyaml                Config file parsing
wandb                 Optional — experiment tracking
```

Install with:
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

---

*Built from scratch — no pre-trained weights, no black boxes. Every weight in the model was initialised randomly and learned entirely from your training data.*
