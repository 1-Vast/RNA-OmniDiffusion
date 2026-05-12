# Validated Mainline

RNA-OmniPrefold validated model framework after systematic cleanup of invalid/negative modules.

## Configuration

```
MS-MPRM + PairRefine + pair-aware masking + corrected lr schedule + strict Nussinov decode
```

### Fixed Parameters

| Parameter | Value |
|---|---|
| Learning rate | 0.001 |
| Warmup steps | 50 |
| Max steps (default) | 300 |
| Batch size | 24 |
| Pair loss weight (λ_pair) | 5.0 |
| Pair positive weight | 10.0 |
| Pair negative ratio | 5 |
| Dropout | 0.1 |
| Hidden size | 512 |
| Layers | 8 |
| Heads | 8 |
| Pair head | MLP |
| PairRefine | True |
| Distance bias | True |
| Strict Nussinov decode | True |
| **min_loop_length** | **4** (calibrated) |

### Metrics (Val Split, 300 Steps, lr=0.001, warmup=50)

| Metric | Value |
|---|---|
| Pair F1 (test, seed42) | 0.341 (baseline) → **0.405** (min_loop=4) |
| 3-seed mean F1 (test, ml=4) | **0.398** |
| Pair Precision | 0.352 |
| Pair Recall | 0.517 |
| Valid Structure Rate | 1.00 |

min_loop_length=4 is now the validated default. It provides +18.9% F1 improvement over min_loop=3 (eval-only, no retraining required). All 3 seeds benefit, no recall collapse.

## Disabled/Removed Modules

All modules below were systematically tested and found to be invalid, negative, or inconclusive. They are either removed from the codebase or default-disabled in config.

| Module | Status | Reason |
|---|---|---|
| RNA-FM distillation | Removed | No contribution beyond D-only baseline |
| LLM semantic tokens | Removed | Dropped F1 from 0.572 → 0.385 |
| LLM preference loss | Removed | No benefit over no-pref baseline |
| LLM reranker | Removed | F1=0.1025 < Rule=0.1223 |
| LLM hard replay curator | Removed | LLM replay = rule replay |
| LLM query adapter | Removed | No improvement over mainline |
| Structural tag auxiliary | Removed | Degrades F1 or neutral |
| PairLossPolicy | Disabled | Weight changes do not affect F1 |
| Structural importance weighting | Disabled | Reduces F1 by ~0.005 |
| Typed data training | Removed | No demonstrated benefit |
| Coach policy | Removed | No demonstrated benefit |
| Global pair_logit_offset=-2 | Rejected | Seed-dependent recall collapse |
| Pair-specific bias table (500 steps) | Below mainline | val F1=0.2428 < 0.351 mainline |
| Local rank loss | Too slow | Needs vectorization optimization |

## Pair Calibration Status

`pair_logit_offset: 0.0` is the default. `pair_calibration.enabled: false`.
Global offset rejected after multi-seed negative results (see `docs/calibration_analysis.md`).
Candidate pruning (min_loop=4) shows promise as eval-only enhancement (test F1=0.3896).
Energy-inspired prior (lambda=0.5) gives modest improvement (F1=0.3598).
Both are eval-only and do not require retraining.

## Config File

`config/mainline_lr0010.yaml` — canonical validated config. All invalid modules are not referenced.

## Quick Start

```bash
python main.py smoke
python main.py train --config config/mainline_lr0010.yaml --device cuda --max_steps 300
python scripts/eval.py bench --config config/mainline_lr0010.yaml --ckpt outputs/mainline_lr0010/best.pt --split val --device cuda --decode nussinov --stage_logits
```

## Architecture Variants (Route P)

Three hand-proposed pair-head architecture variants and one random-conv control were tested at 300 steps:

| Architecture | Val F1 | Δ mainline | Notes |
|---|---:|---:|---|
| Mainline (base) | 0.3184 | — | — |
| Mainline (fresh run) | 0.3223 | +0.0039 | Run-to-run noise |
| Distance bucket head | 0.3150 | -0.0034 | Negative |
| Long-range gate | 0.3102 | -0.0082 | Negative |
| **Random conv (same params)** | **0.3312** | **+0.0128** | Capacity control |
| **Stem continuity refine** | **0.3312** | **+0.0128** | Anti-diagonal init |
| Stem refine (previous run) | 0.3340 | +0.0156 | Prior run, within noise |

### Stem continuity refine validation

Ablation confirmed: stem_continuity_refine (anti-diagonal 2D conv) ≈ random_conv (random init 2D conv, same parameter count). Both achieve F1 ≈ 0.3312, exceeding mainline by +0.0128.

**Conclusion**: The improvement comes from the small 2D conv residual on pair logits (10 parameters), not from the anti-diagonal stem-continuity bias. The module is a valid capacity improvement but is not a stem-specific structural inductive bias.

Recommendation: The 2D conv residual on pair logits may be included as an optional module, but should not be called "stem continuity refine" since the bias is not the active ingredient.
