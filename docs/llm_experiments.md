# LLM Experiments

Final status: **LLM has not demonstrated independent value as a model-performance module in this repository.**

After systematic evaluation of 16+ LLM integration routes, **no route exceeded deterministic rule-based baselines or hand-designed controls.**

## Route Summary

| Route | F1 Impact | Conclusion |
|---|---|---|
| LLM semantic tokens | -0.187 | Input distribution pollution / coarse signal |
| LLM preference loss | +0.0007 | Negligible, dominated by Pair BCE |
| RAG preference | â‰? | Same as no-pref |
| LLM reranker | -0.020 | Worse than rule reranker |
| Constrained LLM reranker | 0.000 | Equals rule, no added value |
| LLM hard replay curator | 0.000 | LLM replay = rule replay |
| LLM decode policy proposer | N/A | Not independently tested |
| LLM query adapter | 0.000 | No improvement over mainline |
| LLM structural importance (Route O) | -0.005 | Reduces F1; importance scores lack variance |
| LLM macro config planner (Route I) | +0.122 | Exposed lr direction, but grid search also finds it; useful as experiment planner, not model module |
| LLM architecture hypothesis (Route P) | N/A | LLM not tested; hand-proposed architecture worked |
| Pair-logit residual conv (hand) | +0.013 | Hand design; gain confirmed but from conv capacity, not specific bias |

## Key Conclusions

1. **LLM has no role inside the model forward or training loop.** All attempts to inject LLM signals (tokens, embeddings, weights, reranking, preferences) either degraded performance or added no benefit.

2. **LLM as experiment planner has limited utility.** Route I identified a learning rate search direction, but grid search also finds it. This is an automation convenience, not a scientific contribution.

3. **Hand-designed architectures outperform LLM proposals.** Route P's stem continuity refine was hand-proposed based on error-slice analysis. LLM architecture proposals were not tested because hand proposals already worked.

4. **LLM may remain useful as an offline experiment planner or hypothesis generator**, but no LLM pathway is currently included in the validated model pipeline.

## Route O (Structural Importance Masking) â€?Detail

Original claim: rule importance F1=0.3331 > baseline 0.3235. **Disproven.** The `_imp` field was never wired into the training pipeline. The F1 difference was a training-subset confound (500 vs 1316 samples). When properly implemented, importance weighting reduces F1 by ~0.005.

## Route P (Stem Continuity Refine) â€?Detail

Hand-proposed 2D conv residual improved val F1 from 0.3184 to 0.3312 (+0.0128). However, random-conv control achieved identical F1. The gain is generic residual capacity, not stem-specific structural bias. Module renamed to `pair_residual_conv` and kept optional/disabled by default. LLM was not involved in this result.

## Route Q (LLM-Guided Experiment Planner) â€?Detail

Route Q tested whether LLM can serve as an offline experiment planner, proposing config changes from an allowlisted search space based on aggregated error taxonomy (pair-distance, structure-pattern, sequence-property breakdowns). LLM never enters the model forward or training loop.

**Results (300-step, val split)**:
- Mainline: F1 = 0.3184
- Best hand proposal (residual_conv): F1 = 0.3293 (+0.011)
- Hand lr0012: F1 = 0.2181 (-0.100) â€?training instability at higher lr
- LLM proposal (pairrefine 32ch Ã— 2blk, warmup=0): F1 = 0.2431 (-0.075) â€?larger model fails to converge at 300 steps
- Other proposals: not fully tested

**Route Q Level: 1** â€?LLM proposal did not outperform hand proposal or mainline. The winning hand proposal (residual conv) was already known from Route P. LLM's architecture suggestion (wider PairRefine) was counterproductive at 300 steps.

**Conclusion**: LLM remains useful only as an offline assistant/planner, not a validated performance contributor. All routes tested (O, P, Q) confirm that hand-designed approaches match or exceed LLM proposals.

**Route Q updated results (v2, with fixed LLM planner)**:
- Mainline: 0.3184
- Hand_residual_conv: 0.3293 (+0.011)
- LLM_Tune_lr_lambda (lr=0.0008, lambda_pair=7.0): 0.3235 (+0.0051)
- LLM_Enable_pair_residual (kernel_size=5): 0.3160 (-0.0024)

Best LLM proposal (+0.0051) exceeded mainline but did not exceed best hand proposal. Route Q remains **Level 1**: LLM planner has no independent advantage over hand planner.

**Route Q Combo Test (fresh mainline comparison)**:
Fresh mainline 300-step training achieved F1=0.3321 â€?exceeding all modification runs:
- Mainline: 0.3321
- Combo (LLM tune + residual): 0.3314 (-0.0007)
- Hand residual: 0.3293 (-0.0028)
- LLM tune only: 0.3235 (-0.0086)

No modification clearly exceeds mainline. All differences within run-to-run training noise (~0.014 F1). Previously observed "gains" were likely within noise. Route Q Level 0/1.



## Route R (LLM Implementation Audit) ¡ª Detail

Route R tested whether LLM can serve as an implementation audit assistant, proposing testable bug/mismatch hypotheses from codebase structure, training logs, and error taxonomy. LLM never enters the model forward or training loop.

**Results**:
- Hand audit: 8 hypotheses, 2 confirmed (min_loop_consistency: medium; warmup lr: info)
- Random audit: 8 hypotheses, 0 confirmed (all unknown ¡ª no matching programmatic checks)
- LLM audit: 6 hypotheses, 0 confirmed (all unknown ¡ª no matching programmatic checks)

**Route R Level: 1** ¡ª LLM did not identify bugs that hand audit missed. Hand audit found a real (minor) min_loop alignment issue where 28 val-set pairs have |i-j| < decode min_loop=3, meaning the model trains on pairs it cannot decode. LLM hypotheses could not be programmatically verified.
