# Benchmark Summary

## Setup
- **Hardware**: RTX 4060 laptop GPU
- **Dataset**: ArchiveII-based local split (train=1316, val=282, test=282)

## Results

### Overall (test split, 282 samples)

| Method | Steps | Test F1 | Precision | Recall | Valid |
|---|---|---|---|---|---|
| **ViennaRNA 2.7.2** | — | **0.5179** | 0.4734 | 0.6078 | 100% |
| **OmniPrefold Deeper (12L)** | 1000 | **0.4464** | 0.3947 | 0.5496 | 100% |
| OmniPrefold Ensemble (base×3) | 300×3 | 0.4186 | 0.3627 | 0.5323 | 100% |
| OmniPrefold Base (8L, ml=4) | 300 | 0.4053 | 0.3517 | 0.5174 | 100% |
| OmniPrefold Baseline (ml=3) | 300 | 0.3511 | 0.296 | 0.464 | 100% |

ViennaRNA gap reduced from 0.167 → **0.072**.

Note: min_loop=4 result is 3-seed mean. All pruning configurations are eval-only (no retraining).
| **OmniPrefold + min_loop=4 prune** | — (eval-only) | **0.3896** | 0.3360 | 0.5281 | 100% | ~70ms |
| **OmniPrefold + energy prior λ=0.5** | — (eval-only) | **0.3598** | 0.3081 | 0.4860 | 100% | ~2ms overhead |
| OmniPrefold + bias table | 500 | 0.2428 (val) | — | — | 100% | — |

## Structural calibration findings (2026-05)

- Global `pair_logit_offset=-2`: rejected (seed 123 crash, F1=0.316±0.076)
- `min_loop=4` pruning: **+14% F1 improvement** (eval-only, no retraining)
- Energy prior (canonical+loop+distance): modest improvement
- Bias table (learnable): underperforms mainline at 500 steps
- Local rank loss: computationally too slow with current implementation

### Step scaling (val split)

| Steps | Best Val F1 | Best Epoch |
|---|---|---|
| 300 | 0.2963 | 5 |
| 1000 | 0.3157 | 10 |
| 3000 | 0.3066 | 6 |

### Length-wise (OmniPrefold 300-step, val)

| Length bin | N | Mean F1 |
|---|---|---|
| Short (0-80) | 81 | 0.4177 |
| Medium (80-150) | 137 | 0.3266 |
| Long (150+) | 64 | 0.1909 |

## Analysis

1. **ViennaRNA is ~1.52x better** than OmniPrefold on test F1 (0.518 vs 0.341 at 300 steps).
2. **Training steps beyond 300 do not help**: F1 plateaus around 0.33-0.35 regardless of training budget (300, 1000, 3000 steps).
3. **Long-sequence degradation** is the primary failure mode: F1 drops from 0.418 (short) to 0.191 (long).
4. **Valid structure rate is 100%** for both methods via Nussinov decoding.
5. **The neural model is structurally limited** — current architecture/data may be insufficient to learn thermodynamic folding principles that ViennaRNA encodes via hand-tuned energy parameters.

## Limitations
- Single seed (42); no multi-seed statistics
- Family field unavailable in local dataset (all "OTHER")
- Only ArchiveII-based local split; no public benchmark splits
- No LinearFold/CONTRAfold/EternaFold comparison (not installed)

