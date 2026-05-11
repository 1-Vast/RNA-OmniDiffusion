# LLM-guided Preference Optimization

## Current Status

**2026-05-09**: Preference optimization evaluated across multiple budgets and configurations.

### Key Results

| Budget | Method | Pair F1 | vs baseline |
|---|---|---|---|
| 300-step | baseline | 0.1852 | — |
| 300-step | oracle β=0.005 warmup | 0.1973 | +0.0121 |
| 500-step | baseline | 0.2440 | — |
| 500-step | oracle β=0.005 warmup | 0.2447 | +0.0007 |
| 500-step | RAG β=0.005 warmup | 0.2442 | +0.0002 |

### Critical Findings

1. **Preference helps at early stages (300 steps, +0.0121)** but provides diminishing returns at later stages (500 steps, +0.0007).
2. **Low-beta (0.005) + warmup (start_epoch_ratio=0.3) is essential.** High beta (2.0) without warmup destroys training (oracle F1=0.1727 vs baseline 0.1852).
3. **Low-label same-step comparisons are confounded by epoch exposure.** Low10 300-step (36 epochs) outperforms full 300-step (5.5 epochs), but same-epoch low10 45-step (0.1414) underperforms full (0.1852).
4. **Preference is more of an early-stage regularizer** than a full-budget performance booster.
5. **LLM/RAG does not clearly outperform rule-based preference** at current judge quality.

### Valid Configuration

```yaml
preference:
  enabled: true
  beta: 0.005
  start_epoch_ratio: 0.3
  min_confidence: 0.6
  loss_type: ranking
  skip_empty_pairdiff: true
  confidence_weighted: true
```

### Failed Configurations

- beta=2.0 + no warmup: Destroys training (gradient overwhelms BCE).
- Weighted BCE: Significantly worse than baseline.
- Full-matrix pair BCE: Too strong, degrades performance.

## Motivation

Supervised pair-relation BCE treats every annotated pair equally. In practice, some pairs are structurally more salient than others. Preference optimization provides a complementary signal: the model learns to rank candidate structures via pairwise comparison, without requiring finer-grained per-pair supervision.

## Candidate Generation

Candidate structures are generated from a trained checkpoint by sweeping Nussinov decode hyperparameters:

```bash
conda run -n DL python scripts/cand.py \
  --config config/candidate.yaml \
  --ckpt outputs/candidate/best.pt \
  --input dataset/archive/val.jsonl \
  --out outputs/pref/cand.jsonl \
  --limit 16 --device cuda
```

Each sample receives multiple candidates (different gamma × threshold combinations), deduplicated and subsampled.

## Rule Preference Baseline

A heuristic judge ranks candidates by weighted structural features (pair count, canonical ratio, stem continuity, etc). This provides a zero-cost baseline:

```bash
conda run -n DL python scripts/judge.py \
  --input outputs/pref/cand.jsonl \
  --out outputs/pref/rulebuf.jsonl \
  --mode rule
```

## LLM Preference Critic

A general-purpose LLM (accessed via .env credentials) compares candidate structures without seeing ground-truth labels. The LLM outputs preferred/rejected candidate IDs and a confidence score.

Mock (random control):
```bash
conda run -n DL python scripts/judge.py \
  --input outputs/pref/cand.jsonl \
  --out outputs/pref/llmbuf.mock.jsonl \
  --mode llm --provider mock
```

Real LLM call:
```bash
conda run -n DL python scripts/judge.py \
  --input outputs/pref/cand.jsonl \
  --out outputs/pref/llmbuf.jsonl \
  --mode llm --provider env --limit 16
```

The `.env` file must contain:
```
LLM_BASE_URL=...
LLM_TOKEN=...
LLM_MODEL=...
```

## Pair-level Ranking Loss

Preference is converted into a differentiable pair-level ranking objective:

```
score_good = mean(pair_logits on preferred-only pairs)
score_bad  = mean(pair_logits on rejected-only pairs)
loss       = -log sigmoid(score_good - score_bad)
```

Only pairs that differ between preferred and rejected candidates contribute. The loss is skipped when the difference set is empty.

The total training loss becomes:

```
L = L_supervised + beta * L_ranking
```

## Controls

To verify that preference signals are meaningful, the pipeline supports three control judges:

- **Rule preference**: heuristic ranking baseline.
- **Random preference**: random preferred/rejected assignment, establishing a noise floor.
- **Shuffled preference**: shuffle buffer pairings across samples, breaking sample-level correspondence.

## Safety Boundaries

- The LLM does not see ground-truth structures.
- The LLM does not generate pseudo labels or pair annotations.
- The LLM does not enter model forward.
- The LLM is not required for inference.
- Preference is applied only during training, gated behind `preference.enabled=true`.
- API keys are read from `.env` only; never logged, saved, or committed.

## Difference from Failed Semantic Tokens

| Aspect | Semantic Tokens (failed) | Preference (current) |
|---|---|---|
| LLM role | Generate semantic condition tokens | Compare candidate structures |
| Signal type | Sample-level conditioning | Pair-level ranking |
| Integration | Model forward (token injection) | Loss-only (gradient signal) |
| Coverage | Low (most SEM_UNKNOWN) | Full (all samples get candidates) |

## Experiments

### Quick smoke (5-step training)

```bash
# Rule preference
conda run -n DL python main.py train --config config/rulepref.yaml --device cuda --max_steps 5

# LLM preference
conda run -n DL python main.py train --config config/pref.yaml --device cuda --max_steps 5
```

### Full run requirements

A full preference run requires:
1. A trained checkpoint (Stage 2).
2. Candidate JSONL generated.
3. Preference buffer (rule or LLM).
4. Training with preference enabled.

Preference is not claimed as a main improvement until validated against the baseline on the same dataset with the same seed and training budget.
