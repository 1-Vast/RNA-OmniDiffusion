# Architecture

RNA-OmniPrefold validated model architecture after systematic cleanup.

## Core Pipeline

```
RNA sequence
  -> Transformer encoder (MS-MPRM backbone)
    -> Token heads (sequence / structure / general)
    -> Pair head (MLP bilinear projection)
      -> PairRefine (2D conv blocks) [optional]
      -> Pair logit residual conv [optional, default disabled]
      -> Distance bias
    -> Strict Nussinov decode (valid non-crossing structures)
```

## Validated Mainline Components

| Component | Status | Description |
|---|---|---|
| Transformer encoder | Enabled | 8-layer, 512-dim, 8-head transformer with task/time/segment/position embeddings |
| MS-MPRM masking | Enabled | Multi-scale masked pair-relation modeling with pair-aware masking |
| Pair head (MLP) | Enabled | Bilinear projection from sequence hidden states |
| PairRefine | Enabled | 1 block, 16-channel 2D conv refinement on pair logits |
| Distance bias | Enabled | Learnable distance-bucket embedding added to pair logits |
| Nussinov decode | Enabled | Strict non-crossing dynamic programming decode |
| Corrected lr schedule | Enabled | lr=0.001, warmup=50, 300 steps |

## Optional Experimental Modules

### Pair-logit residual conv (`pair_residual`)

Small 2D conv residual applied to pair logits. Formerly called "stem_continuity_refine" or "stem_refine".

| Setting | Default | Notes |
|---|---|---|
| `pair_residual.enabled` | `false` | Must be explicitly enabled |
| `pair_residual.type` | `"conv2d"` | `conv2d` = generic 2D conv; legacy `stem_refine` accepted |
| `kernel_size` | `3` | Conv kernel size |
| `residual_scale` | `0.1` | Weak residual multiplier |
| `init` | `"zero"` | `"zero"`, `"random"`, or `"stem"` (legacy anti-diagonal) |

Validation results (300-step, val split): the module improved F1 from 0.3223 to 0.3312 (+0.0128), but a random-conv control with identical parameter count achieved the same F1. The gain is interpreted as generic residual capacity, not stem-specific structural bias.

## Removed/Disabled Modules

| Module | Reason |
|---|---|
| StructureQueryAdapter (query adapter) | No improvement over mainline |
| Struct_aux heads (tag auxiliary) | Degrades or neutral F1 |
| Preference loss / buffer | No benefit over no-pref baseline |
| Coach policy | No demonstrated benefit |
| Typed data training | No demonstrated benefit |
| LLM semantic tokens | Dropped F1 from 0.572 → 0.385 |
| LLM reranker | F1=0.1025 < Rule=0.1223 |
| LLM importance | Reduces F1 by ~0.005 |
| PairLossPolicy | Weight changes do not affect F1 |
| External encoder distillation | No contribution beyond D-only |

## Training Hyperparameters

| Parameter | Value |
|---|---|
| LR | 0.001 |
| Warmup steps | 50 |
| Batch size | 24 |
| λ_pair | 5.0 |
| λ_struct | 1.0 |
| λ_seq | 1.0 |
| Pair negative ratio | 5 |
| Pair positive weight | 10.0 |
| Dropout | 0.1 |
| Weight decay | 0.01 |
| Gradient clip | 1.0 |
| Optimizer | AdamW |
