"""
model.py — GPT-style decoder-only transformer, built from scratch.

Architecture matches configs/train_config.yaml:
  hidden_size, num_layers, num_heads, vocab_size, max_seq_len, dropout

Usage:
    from model import GPT, GPTConfig
    cfg = GPTConfig.from_yaml("configs/train_config.yaml")
    model = GPT(cfg)
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    vocab_size: int = 32000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 8
    max_seq_len: int = 1024
    dropout: float = 0.1
    # MLP inner dimension — 4× hidden is standard for GPT
    mlp_ratio: float = 4.0
    # Whether to use bias in attention projections and MLP
    bias: bool = False

    @classmethod
    def from_yaml(cls, path: str) -> "GPTConfig":
        with open(path) as f:
            cfg = yaml.safe_load(f)
        m = cfg.get("model", {})
        return cls(
            vocab_size=m.get("vocab_size", cls.vocab_size),
            hidden_size=m.get("hidden_size", cls.hidden_size),
            num_layers=m.get("num_layers", cls.num_layers),
            num_heads=m.get("num_heads", cls.num_heads),
            max_seq_len=m.get("max_seq_len", cls.max_seq_len),
            dropout=m.get("dropout", cls.dropout),
        )

    @property
    def head_dim(self) -> int:
        assert self.hidden_size % self.num_heads == 0, (
            f"hidden_size {self.hidden_size} must be divisible by num_heads {self.num_heads}"
        )
        return self.hidden_size // self.num_heads

    @property
    def mlp_hidden(self) -> int:
        return int(self.hidden_size * self.mlp_ratio)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation (no learned bias, more efficient)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with:
      - fused QKV projection for efficiency
      - scaled dot-product attention (uses Flash Attention when available)
      - causal mask applied via is_causal=True in sdp attention
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.dropout = config.dropout

        # Fused Q, K, V in one linear layer — 3× hidden
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.bias)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,                   # (B, T, C)
        past_kv: Optional[tuple] = None,   # cached (k, v) for inference
        use_cache: bool = False,
    ):
        B, T, C = x.shape

        # Project to Q, K, V
        qkv = self.qkv(x)                                    # (B, T, 3C)
        q, k, v = qkv.split(self.hidden_size, dim=2)         # each (B, T, C)

        # Reshape for multi-head attention: (B, heads, T, head_dim)
        def reshape(t):
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        # Append to KV cache if provided (for auto-regressive generation)
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v) if use_cache else None

        # Scaled dot-product attention — uses Flash Attention 2 under the
        # hood when running on CUDA with PyTorch >= 2.0
        attn_dropout = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=attn_dropout,
            is_causal=(past_kv is None),   # causal only for fresh sequences
        )

        # Merge heads back: (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.out_proj(y))
        return y, present_kv


class MLP(nn.Module):
    """
    Position-wise feed-forward with SwiGLU activation (GPT-NeoX / LLaMA style).
    SwiGLU uses two parallel projections and a gating mechanism, which tends
    to outperform the original GELU two-layer MLP at the same parameter count.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        # For SwiGLU we project to 2× inner dim and split in the activation
        inner = config.mlp_hidden
        self.gate_proj = nn.Linear(config.hidden_size, inner, bias=config.bias)
        self.up_proj   = nn.Linear(config.hidden_size, inner, bias=config.bias)
        self.down_proj = nn.Linear(inner, config.hidden_size, bias=config.bias)
        self.dropout   = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: silu(gate) * up
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: Norm → Attention → Norm → MLP."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size)
        self.attn  = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.hidden_size)
        self.mlp   = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[tuple] = None,
        use_cache: bool = False,
    ):
        attn_out, present_kv = self.attn(self.norm1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, present_kv


# ---------------------------------------------------------------------------
# Full GPT model
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    """
    Decoder-only GPT-style transformer.

    Parameters are initialised using the GPT-2 scheme:
      - Normal(0, 0.02) for embeddings and linear weights
      - Scaled down by 1/sqrt(2 * n_layers) for residual projections
        (out_proj and down_proj) to keep activations stable at init.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            tok_emb  = nn.Embedding(config.vocab_size, config.hidden_size),
            pos_emb  = nn.Embedding(config.max_seq_len, config.hidden_size),
            drop     = nn.Dropout(config.dropout),
            blocks   = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)]),
            norm     = RMSNorm(config.hidden_size),
        ))
        # Language model head — shares weights with the token embedding (weight tying)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.tok_emb.weight

        self._init_weights()

        n_params = self.num_parameters()
        print(f"GPT initialised: {n_params / 1e6:.1f}M parameters")

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self):
        std = 0.02
        residual_scale = std / math.sqrt(2 * self.config.num_layers)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Residual projections get scaled down init
                if name.endswith(("out_proj", "down_proj")):
                    nn.init.normal_(module.weight, mean=0.0, std=residual_scale)
                else:
                    nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,           # (B, T)
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None,  # (B, T) for training
    ):
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, (
            f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"
        )

        # Positional offset when using KV cache
        past_len = past_key_values[0][0].shape[2] if past_key_values else 0
        pos = torch.arange(past_len, past_len + T, dtype=torch.long, device=input_ids.device)

        x = self.transformer.drop(
            self.transformer.tok_emb(input_ids) + self.transformer.pos_emb(pos)
        )

        present_key_values = []
        for i, block in enumerate(self.transformer.blocks):
            past_kv = past_key_values[i] if past_key_values else None
            x, present_kv = block(x, past_kv=past_kv, use_cache=use_cache)
            present_key_values.append(present_kv)

        x = self.transformer.norm(x)
        logits = self.lm_head(x)                    # (B, T, vocab_size)

        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": logits, "past_key_values": present_key_values}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.pos_emb.weight.numel()
        return n

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,   # (1, T) prompt
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Auto-regressive generation with top-k / nucleus (top-p) sampling.
        Returns the full sequence (prompt + generated tokens).
        """
        self.eval()
        past_key_values = None

        for _ in range(max_new_tokens):
            # On the first step pass the full prompt; after that just the new token
            idx_cond = input_ids if past_key_values is None else input_ids[:, -1:]
            out = self(idx_cond, past_key_values=past_key_values, use_cache=True)
            past_key_values = out["past_key_values"]

            logits = out["logits"][:, -1, :]     # (1, vocab_size)

            # Repetition penalty — down-weight tokens that already appear
            if repetition_penalty != 1.0:
                for token_id in set(input_ids[0].tolist()):
                    logits[0, token_id] /= repetition_penalty

            logits = logits / max(temperature, 1e-8)

            # Top-k filtering
            if top_k > 0:
                top_k = min(top_k, logits.size(-1))
                kth_val = torch.topk(logits, top_k).values[:, -1, None]
                logits = logits.masked_fill(logits < kth_val, float("-inf"))

            # Nucleus (top-p) filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)   # (1, 1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        return input_ids


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = GPTConfig(vocab_size=1000, hidden_size=128, num_layers=2, num_heads=4, max_seq_len=64)
    model = GPT(cfg)
    ids = torch.randint(0, 1000, (2, 32))
    out = model(ids, labels=ids)
    print(f"loss={out['loss'].item():.4f}  logits={out['logits'].shape}")

    # Generation test
    prompt = torch.randint(0, 1000, (1, 5))
    generated = model.generate(prompt, max_new_tokens=10, temperature=0.8)
    print(f"generated shape: {generated.shape}")
