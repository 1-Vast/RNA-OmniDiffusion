# Negative / Inconclusive Results

This document records all experimental routes tested but not adopted as mainline contributions in RNA-OmniPrefold.

| Route | Result | Evidence | Decision |
|---|---|---|---|
| RNA-FM distillation | Weak | Isolated contribution small; D-only matched teacher on toy data | Deleted |
| LLM semantic tokens | Negative | Pair F1 dropped from 0.5723 to 0.3851 | Not recovered |
| LLM preference (full-budget) | Inconclusive | 500-step: no-pref 0.2440, oracle pref 0.2447, RAG pref 0.2442 | Not mainline |
| Low-label | Confounded | same-step advantage was epoch exposure artifact | Not pursued |
| Fine-grained relation_mask | Insufficient | Only affects token denoising, not Pair BCE | Not mainline |
| PairLossPolicy / weighted BCE | Negative | Vectorized but hard_negative_weight=2.0 does not change F1 | Not continued |
| LLM architecture proposer | Suspended | Non-LLM search space too small | Not continued |
| LLM reranker (free) | Negative | F1=0.1025 vs Rule=0.1223 | Not valid module |
| LLM reranker (constrained) | Inconclusive | F1=0.1223 equals Rule, no added value | Not valid module |
| No-LLM structural tagaux | Negative | λ=0.03 unchanged (0.2009), λ=0.20 degrades (0.1899) | Not recommended |
| LLM structured tags / CLIP | Not entered | No-LLM tagaux showed no positive signal | Deferred |
| Structural Importance Masking | **Negative** | Rule imp 0.3349 < baseline 0.3406 (-0.0057). Random imp = rule imp. Mechanism confirmed non-beneficial | Closed |
| Stem continuity refine (as structural bias) | **Not validated** | Gain +0.0128 F1, but random-conv control matches. Capacity effect, not stem-specific bias. | Renamed to pair_residual_conv |

## Details

### RNA-FM Distillation
Sequence-level teacher distillation tested as optional pretraining signal. Toy comparison: D-only and teacher both reached Pair F1 0.8333. Fine-tune val loss slightly worse with teacher (1.0245 vs 0.9643). Removed from mainline.

### LLM Semantic Tokens
Sample-level biological hints injected as condition tokens. Coverage low (most SEM_UNKNOWN). F1 dropped from 0.5723 to 0.3851. Conclusion: semantic tokens are not part of mainline.

### LLM Preference / RAG Preference
Low-beta (0.005) + warmup shows early-stage signal (+0.0121 at 300 steps). Full-budget (500 steps) shows negligible gain (+0.0007). Preference is not a cumulative performance booster.

### Low-label Confound
Low10 appeared to outperform full at same-step (0.1954 vs 0.1852). Same-epoch comparison corrected: low10 45 steps (0.1414) vs full 300 steps (0.1852). Artifact of imbalanced epoch exposure.

### PairLossPolicy
Implemented 100% vectorized weighted BCE with canonical/distance masks. hard_negative_weight=2.0 does not change Pair F1 from MS-MPRM baseline at 300 steps.

### LLM Reranker
Free LLM reranker (F1=0.1025) underperforms Rule Reranker (F1=0.1223). Constrained LLM equals Rule. Oracle Top-K upper bound exists (0.1490) but LLM cannot approach it.

### Structural Tag Auxiliary
Struct_aux infrastructure built. No-LLM tagaux at λ=0.03 unchanged from MS-MPRM. λ=0.20 degrades F1 to 0.1899. Auxiliary task interferes with Pair BCE.

### Structural Importance Masking (Route O)
Per-position structural importance scores converted to pair-level BCE loss weights. Original claim of rule importance F1=0.3331 exceeding baseline was a confound: the importance data (500 samples) was a subset of the full training set (1316 samples), and subset training alone explains the gain (baseline-500 = 0.3406).

Clean comparison at lr=0.001, warmup=50, 300 steps on val split:
- Baseline-500 (no importance): 0.3406
- Rule importance: 0.3349 (-0.0057)
- Random importance: 0.3354 (-0.0052)

Both rule and random importance weighting REDUCE F1 compared to no-weighting baseline. Rule ≈ Random confirms mechanism is not benefiting from structural signal. Root cause: importance scores lack variance (median 0.70, mean 0.82), producing near-uniform pair weights after normalization. Route O closed.

### Stem Continuity Refine (as structural inductive bias)
Stem continuity refine (2D conv residual on pair logits with anti-diagonal initialization) improved val F1 from 0.3184/0.3223 to 0.3312 (+0.0128) at 300 steps. However, a random-conv control (same parameter count, random initialization) achieved the identical F1=0.3312. Ablation confirmed the gain comes from generic 2D conv residual capacity (+10 params), not from anti-diagonal stem-continuity initialization. The module has been renamed to `pair_residual_conv` and documented as an optional capacity module, not a verified structural inductive bias.
