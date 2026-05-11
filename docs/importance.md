# Route O: Structural Importance Masking

**Status**: NEGATIVE — pair-level structural importance weighting does not improve performance.

## Hypothesis

Per-position structural importance scores, when converted to pair-level loss weights, would guide the model to focus training on structurally meaningful base-pair positions, improving F1.

## Mechanism

1. `scripts/importance.py` generates per-position `_imp` values for each training sample
2. `models/collator.py` converts `_imp[L]` to pair-level weights `W[i,j] = imp[i] * imp[j]` (outer product)
3. `models/omni.py` `compute_omni_loss` applies these weights element-wise to the pair BCE loss, normalized to mean 1.0

## Prior Claim (DISPROVEN)

Previous summary claimed rule importance achieved F1=0.3331 vs baseline 0.3235 on test split. This was a **confound**: the "importance" training used a 500-sample subset while the baseline used 1316 samples. The `_imp` field was generated but **never consumed** by the training pipeline until this fix.

## Clean Comparison (Val Split, 300 Steps, lr=0.001, warmup=50)

| Method | Train Samples | Val F1 | Precision | Recall | Δ baseline-500 |
|---|---:|---:|---:|---:|---:|
| Baseline (1316 full) | 1316 | 0.3201 | 0.2718 | 0.4166 | — |
| Baseline (500 subset) | 500 | **0.3406** | 0.2876 | 0.4498 | — |
| Rule importance | 500 | 0.3349 | 0.2841 | 0.4340 | -0.0057 |
| Random importance | 500 | 0.3354 | 0.2837 | 0.4371 | -0.0052 |
| LLM importance | 41 | — | — | — | inconclusive |

## Key Findings

1. **Training on 500-sample subset outperforms full 1316 samples** (0.3406 vs 0.3201, +0.0205). Suggests the full dataset contains harder/noisier samples that dilute learning signal at 300 steps.

2. **Pair-level importance weighting REDUCES F1** compared to same 500-sample baseline without weighting:
   - Rule: -0.0057
   - Random: -0.0052

3. **Rule ≈ Random**: The type of importance (structural rule vs random) makes no difference. Both show the same small negative effect.

4. **Root cause of null effect**: The rule-based `_imp` distribution has very low variance:
   - Min: 0.0, P10: 0.70, Median: 0.70, Mean: 0.82, P90: 1.0
   - Outer product of near-uniform values → near-uniform pair weights after mean normalization
   - Effective weight range after normalization: ~0.85 to ~1.22 (narrow)

5. **LLM importance inconclusive**: Only 41 samples available. Cannot provide fair comparison.

## Implementation Changes

- `models/collator.py`: Added `_imp` extraction and `pair_importance` [B,L,L] outer product construction
- `models/omni.py`: Added importance weighting layer to `compute_omni_loss` (composite weight = pos_weight * plp_weight * imp_normalized)
- `models/training.py`: Added `structural_importance` config section with `enabled` and `weight_strength` fields

## Conclusion

**Route O is a confirmed negative.** Pair-level structural importance weighting does not improve pair prediction F1. The mechanism is either neutral (within noise) or slightly harmful. The sparsity of the importance signal (most positions scored near 1.0, low variance) is the likely cause.

**Recommendation**: Close Route O. Do not pursue LLM-based importance further (41 samples insufficient, and even if scaled, rule importance already showed no benefit).
