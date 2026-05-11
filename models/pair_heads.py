# models/pair_heads.py — Lightweight pair-head architecture variants for Route P.
"""Allowlisted pair-head variants. All variants default to identity (no-op) and
must be explicitly enabled via config `pair_arch.enabled` and `pair_arch.type`."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Base / identity (no change from mainline) ──────────────────────────
class BasePairArch(nn.Module):
    """Identity — no pair-head architecture change."""
    def forward(self, pair_logits: torch.Tensor, hidden: torch.Tensor,
                seq_positions: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return pair_logits


# ── Distance bucket conditioned head ──────────────────────────────────
class DistanceBucketHead(nn.Module):
    """Injects distance-bucket embedding into pair features before logit computation.
    Uses learnable embeddings for each distance bucket, added as bias to logits."""
    def __init__(self, dist_buckets: int = 32, dist_max: int = 512):
        super().__init__()
        self.dist_buckets = dist_buckets
        self.dist_max = dist_max
        self.bucket_bias = nn.Embedding(dist_buckets, 1)
        nn.init.zeros_(self.bucket_bias.weight)

    def forward(self, pair_logits: torch.Tensor, hidden: torch.Tensor,
                seq_positions: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        device = pair_logits.device
        B, L = pair_logits.shape[0], pair_logits.shape[1]
        idx = torch.arange(L, device=device)
        dist = (idx[:, None] - idx[None, :]).abs().clamp_max(self.dist_max)
        buckets = torch.div(dist * self.dist_buckets, self.dist_max + 1,
                           rounding_mode="floor").clamp_max(self.dist_buckets - 1)
        bias = self.bucket_bias(buckets).squeeze(-1)  # [L, L]
        return pair_logits + bias.unsqueeze(0)


# ── Long-range gate ───────────────────────────────────────────────────
class LongRangeGate(nn.Module):
    """Learnable scalar gate applied to long-range pair logits (> threshold)."""
    def __init__(self, distance_threshold: int = 48):
        super().__init__()
        self.threshold = distance_threshold
        self.gate = nn.Parameter(torch.tensor(1.0))

    def forward(self, pair_logits: torch.Tensor, hidden: torch.Tensor,
                seq_positions: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        device = pair_logits.device
        L = pair_logits.shape[1]
        idx = torch.arange(L, device=device)
        dist = (idx[:, None] - idx[None, :]).abs()
        long_mask = (dist >= self.threshold).unsqueeze(0).to(pair_logits.dtype)
        short_mask = 1.0 - long_mask
        return pair_logits * short_mask + pair_logits * long_mask * self.gate


# ── Stem continuity refine (anti-diagonal conv) ────────────────────────
class StemContinuityRefine(nn.Module):
    """Light 2D conv along anti-diagonal to strengthen continuous stem patterns."""
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.drop = nn.Dropout2d(dropout)
        # Init near identity (dirac delta)
        nn.init.zeros_(self.conv.weight)
        self.conv.weight.data[0, 0, 1, 1] = 0.5
        nn.init.zeros_(self.conv.bias)

    def forward(self, pair_logits: torch.Tensor, hidden: torch.Tensor,
                seq_positions: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = pair_logits.unsqueeze(1)  # [B, 1, L, L]
        residual = self.conv(self.drop(x))
        return pair_logits + residual.squeeze(1) * 0.1  # weak residual


# ── Random conv control (same params, no anti-diagonal bias) ─────────
class RandomConvControl(nn.Module):
    """Same-parameter 2D conv residual, randomly initialized — capacity control."""
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.drop = nn.Dropout2d(dropout)
        # Default random init — no structural bias

    def forward(self, pair_logits: torch.Tensor, hidden: torch.Tensor,
                seq_positions: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = pair_logits.unsqueeze(1)
        residual = self.conv(self.drop(x))
        return pair_logits + residual.squeeze(1) * 0.1


# ── Registry ──────────────────────────────────────────────────────────
ARCH_REGISTRY = {
    "base": BasePairArch,
    "distance_bucket_conditioned_head": DistanceBucketHead,
    "distance_bucket": DistanceBucketHead,
    "long_range_gate": LongRangeGate,
    "stem_continuity_refine": StemContinuityRefine,
    "stem_refine": StemContinuityRefine,
    "random_conv": RandomConvControl,
}


def build_pair_arch(config: dict) -> nn.Module:
    """Build pair-architecture module from config."""
    cfg = config.get("pair_arch", {}) or {}
    if not cfg.get("enabled", False):
        return BasePairArch()
    arch_type = cfg.get("type", "base")
    arch_class = ARCH_REGISTRY.get(arch_type)
    if arch_class is None:
        raise ValueError(f"Unknown pair_arch type: {arch_type}. "
                         f"Available: {list(ARCH_REGISTRY)}")
    kwargs = {}
    if arch_type == "long_range_gate":
        kwargs["distance_threshold"] = int(cfg.get("distance_threshold", 48))
    elif "distance_bucket" in arch_type:
        kwargs["dist_buckets"] = int(cfg.get("dist_buckets", 32))
        kwargs["dist_max"] = int(cfg.get("distmax", 512))
    elif "stem" in arch_type:
        kwargs["dropout"] = float(cfg.get("dropout", 0.0))
    return arch_class(**kwargs)
