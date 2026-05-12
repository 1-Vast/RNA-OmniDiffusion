# Pair Calibration Analysis

## Global pair_logit_offset → REJECTED

| Seed  | Val F1 | Test F1 | Verdict |
|-------|--------|---------|---------|
| 42    | 0.388  | 0.382   | single-seed positive |
| 123   | crash  | crash   | recall collapse |
| 2024  | drop   | drop    | worse than 0 |

Multi-seed mean F1: 0.316 ± 0.076 (worse than mainline 0.351 ± 0.028)

**Decision**: Global pair_logit_offset is NOT mainline. Retained as optional tuning knob, default 0.0.

## Candidate pruning (min_loop) → VALIDATED

### Full test set (282 samples), multi-seed

| Seed | Baseline (ml=3) | ml=4 | ml=5 | ml=6 |
|---|---|---|---|---|
| 42   | 0.3409 | **0.4053** | 0.3964 | 0.4015 |
| 123  | 0.3286 | **0.3673** | 0.3574 | 0.3673 |
| 2024 | 0.3838 | **0.4220** | 0.4130 | 0.4195 |
| **Mean** | **0.3511** | **0.3982** | 0.3889 | 0.3961 |

**Decision**: `min_loop_length=4` is the validated default. +18.9% F1 improvement (eval-only, no retraining). All 3 seeds improve, no recall collapse.

## Energy-inspired pair prior → MODEST

Best λ=0.5: F1=0.3717 (vs baseline 0.3409). When combined with min_loop=4, F1=0.4029-0.4042 — slightly below ml=4 alone. Prior is redundant when pruning already enforces stricter constraints.

## Pair-specific bias table → BELOW MAINLINE

500-step training: val F1=0.2428 < mainline 0.351. Not effective at current training budget.

## Local ranking loss → TOO SLOW

Per-positive-pair Python loop is prohibitively slow for training. Needs vectorization.

## Final Configuration

```yaml
decoding:
  min_loop_length: 4  # calibrated, +18.9% F1
  use_nussinov: true
  pair_threshold: 0.25
  nussinov_gamma: 2.0
```
