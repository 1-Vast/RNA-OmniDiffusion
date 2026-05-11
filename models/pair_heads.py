# models/pair_heads.py — Lightweight pair-head architecture variants for Route P.
"""Allowlisted pair-head variants. All variants default to identity (no-op) and
must be explicitly enabled via config.

Config keys:
  pair_residual:
    enabled: false        # default disabled
    type: "conv2d"        # conv2d = generic 2D conv residual on pair logits
    kernel_size: 3
    residual_scale: 0.1

Historical note: The former "stem_continuity_refine" module was a weak 2D conv
residual with anti-diagonal initialization that showed val F1 improvement
(+0.0128), but random-conv control matched it. The gain is from generic
residual capacity, not stem-specific structural bias. See docs/mainline.md.
"""

from __future__ import annotations

import warnings

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


# ── Pair-logit residual conv (generic, replaces "stem continuity refine") ──
class PairResidualConv(nn.Module):
    """Small 2D conv residual on pair logits — generic capacity module.

    Formerly called "stem_continuity_refine" or "stem_refine". Ablation showed
    random-conv init matches anti-diagonal init (both +0.0128 F1 over mainline).
    The gain is from residual capacity, not stem-specific bias.
    """
    def __init__(self, kernel_size: int = 3, residual_scale: float = 0.1,
                 init: str = "zero", dropout: float = 0.0):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=padding)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.scale = residual_scale
        if init == "zero":
            nn.init.zeros_(self.conv.weight)
            nn.init.zeros_(self.conv.bias)
        elif init == "stem":
            nn.init.zeros_(self.conv.weight)
            self.conv.weight.data[0, 0, 1, 1] = 0.5
            nn.init.zeros_(self.conv.bias)

    def forward(self, pair_logits: torch.Tensor, hidden: torch.Tensor,
                seq_positions: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = pair_logits.unsqueeze(1)  # [B, 1, L, L]
        residual = self.conv(self.drop(x))
        return pair_logits + residual.squeeze(1) * self.scale


# ── Registry ──────────────────────────────────────────────────────────
ARCH_REGISTRY = {
    "base": BasePairArch,
    "conv2d": PairResidualConv,
    "pair_residual_conv": PairResidualConv,
    "distance_bucket_conditioned_head": DistanceBucketHead,
    "distance_bucket": DistanceBucketHead,
    "long_range_gate": LongRangeGate,
    "random_conv": PairResidualConv,  # random init is default
    # Legacy names — print deprecation warning
    "stem_continuity_refine": PairResidualConv,
    "stem_refine": PairResidualConv,
}

_LEGACY_NAMES = {"stem_continuity_refine", "stem_refine"}


def build_pair_arch(config: dict) -> nn.Module:
    """Build pair-architecture module from config.

    Supports both legacy `pair_arch` and new `pair_residual` config sections.
    """
    # Try new config first, then legacy
    cfg = config.get("pair_residual", None)
    if cfg is None:
        cfg = config.get("pair_arch", {}) or {}
    if not cfg.get("enabled", False):
        return BasePairArch()
    arch_type = cfg.get("type", "base")
    if arch_type in _LEGACY_NAMES:
        warnings.warn(
            f"pair_residual type='{arch_type}' is deprecated. Use type='conv2d' instead. "
            "Previous validation showed the gain comes from generic residual capacity, "
            "not stem-specific initialization.",
            DeprecationWarning, stacklevel=2,
        )
    arch_class = ARCH_REGISTRY.get(arch_type)
    if arch_class is None:
        raise ValueError(f"Unknown pair_residual type: {arch_type}. "
                         f"Available: {list(ARCH_REGISTRY)}")
    kwargs = {}
    if arch_type == "long_range_gate":
        kwargs["distance_threshold"] = int(cfg.get("distance_threshold", 48))
    elif "distance_bucket" in arch_type:
        kwargs["dist_buckets"] = int(cfg.get("dist_buckets", 32))
        kwargs["dist_max"] = int(cfg.get("distmax", 512))
    else:
        kwargs["kernel_size"] = int(cfg.get("kernel_size", 3))
        kwargs["residual_scale"] = float(cfg.get("residual_scale", 0.1))
        kwargs["init"] = str(cfg.get("init", "zero"))
    return arch_class(**kwargs)
