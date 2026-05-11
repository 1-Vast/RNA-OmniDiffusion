# Model Bottleneck Diagnosis

## Root Cause: Poor Negative Pair Calibration

The model plateaus at F1≈0.33-0.35 because **56% of non-pair positions get probability > 0.25** (the Nussinov threshold), causing massive over-pairing (93% of test samples over-pair).

## Key Numbers

| Metric | Value | Implication |
|---|---|---|
| **Neg prob > 0.25** | **56.2%** | Half of non-pairs pass the Nussinov threshold — candidate pool polluted |
| Neg prob > 0.50 | 35.7% | One-third of non-pairs the model is confident ARE pairs — severe miscalibration |
| Neg prob > 0.75 | 14.3% | Model has very high false confidence |
| Pos prob > 0.25 | 96.1% | Most true pairs detected — recall is good |
| **Over-pairing rate** | **93%** | Almost all samples predict too many pairs |
| Under-pairing rate | 0% | Never predicts too few — bias is strongly toward over-prediction |
| Pos logit mean | +2.36 | True pairs get strong signal |
| Neg logit mean | -1.14 | Non-pairs get weak negative signal (should be much lower) |
| Rank accuracy | 87% | Ranking quality is decent — pair logits DO correlate with true pairs |
| **Pos logit P10** | **-0.04** | 10% of true pairs get NEGATIVE logits (missed entirely) |

## Training Loss Dynamics

| Metric | Trend |
|---|---|
| Train pair_loss | 0.67 → 0.45 (decreasing) — still learning |
| Val pair_loss | 0.64 → 0.45 (decreasing) — still improving |
| Val F1 | 0.17 → 0.30 (increasing) — still climbing |
| Margin | 0.06 → 3.28 (increasing) — separation improving |
| Neg logit mean | **always reported as 0.00 in trainlog (bug)** but actually -1.28 in reality |

## Why ViennaRNA is Better

ViennaRNA uses decades of hand-tuned thermodynamic parameters that give near-zero probability to non-canonical or energetically unfavorable pairs. Its false positive rate is much lower. The learned model cannot match this precision with only 1316 training samples.

## Length Degradation

| Length | Omni F1 | RNAfold F1 | Δ |
|---|---:|---:|---:|
| 0-60 | 0.35 | 0.57 | 0.22 |
| 60-80 | 0.44 | 0.54 | 0.10 |
| 80-120 | 0.37 | 0.57 | 0.19 |
| 120-200 | 0.27 | 0.50 | 0.23 |
| 200-350 | 0.21 | 0.41 | 0.20 |
| 350+ | 0.20 | 0.36 | 0.16 |

Both methods degrade with length, but ViennaRNA degrades less steeply.

## Verified Non-Issues

| Check | Result |
|---|---|
| Nussinov threshold tuning | Minimal effect (±0.005 F1 over 0.10-0.35 range) |
| Training convergence | Pair loss still decreasing — model not stuck |
| Valid structure rate | 100% — Nussinov decode works correctly |
| Positive pair detection | 96% positives above threshold — recall is good |

## Probable Limiting Factors

1. **Training data scale**: 1316 samples with ~8,700 positive pairs is insufficient to learn precise pair discrimination across the L² candidate space
2. **Negative saturation**: 3.5M negative pairs per batch at 5:1 ratio may overwhelm the BCE loss, causing the model to saturate at moderate negative logits (-1.14)
3. **Architecture capacity for pair task**: An 8-layer transformer may need more data to learn the complex RNA folding grammar

## Practical Path Forward

| Priority | Action | Expected Impact | Tested |
|---|---|---|---|
| 1 | Train on larger dataset (bpRNA 10k+) | Higher — more data = better generalization | No |
| 2 | **Train-time logit bias (bias=-2.0)** | **+0.017 F1 (0.341→0.358)** | **Yes ✓** |
| 3 | Longer training (10k+ steps on larger data) | Medium | Partial (1000/3000 tested, no gain) |
| 4 | Add thermodynamic features as input | High — ViennaRNA knowledge as priors | No |
| 5 | Use position-specific features (GC, conservation) | Medium | No |

## Multi-Seed Validation (bias=-2.0, 300 steps)

| Seed | offset=0 F1 | offset=-2 F1 | Δ |
|---|---|---:|---:|
| 42 | 0.341 | 0.358 | **+0.017** |
| 123 | 0.329 | 0.228 | -0.101 |
| 2024 | 0.383 | 0.363 | -0.020 |

| Offset | Mean F1 | Std |
|---|---|---|
| 0 | 0.351 | ±0.028 |
| -2 | 0.316 | ±0.076 |

**Conclusion: bias=-2.0 is NOT robust across seeds.** Seed 123 crashes (F1 drops from 0.329 to 0.228) because the aggressive negative bias prevents the model from learning at certain initializations. The fix helps seed 42 but hurts seeds 123 and 2024.

**Decision: Do NOT enter mainline as default.** Keep `pair_logit_offset: 0.0` as default. The offset parameter can be offered as an optional calibration tuning knob for users who want to experiment, but the validated mainline should use offset=0.

### Fine-Grid Offset Sweep (seed=42, 300 steps)

| Offset | Test F1 |
|---|---|
| 0 (default) | 0.341 |
| -1.0 | 0.312 |
| -1.5 | 0.341 |
| -2.0 | **0.358** |
| -2.5 | 0.342 |
| -3.0 | 0.339 |

Peak at -2.0 (single seed 42), but this benefit is seed-dependent.
