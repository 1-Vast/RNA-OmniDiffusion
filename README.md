# RNA-OmniPrefold

Relation-aware Masked Denoising for Constraint-Guided RNA Folding.

## Validated Configuration

```
MS-MPRM + PairRefine + pair-aware masking + corrected lr schedule + strict Nussinov decode
```

## Key Results

| Config | lr | Steps | F1 |
|---|---|---|---|
| **Corrected baseline** | **0.0010** | **300** | **0.3235** |
| MS-MPRM + hard replay | 0.0001 | 300 | 0.2076 |
| MS-MPRM | 0.0001 | 500 | 0.2527 |
| MS-MPRM | 0.0001 | 300 | 0.2009 |

## Quick Start

```bash
python main.py overview
python main.py smoke
python main.py train --config config/candidate.yaml --device cuda
python scripts/eval.py bench --config config/candidate.yaml --ckpt outputs/candidate/best.pt --split test --device cuda --decode nussinov --stage_logits
```

## LLM Status

After systematic evaluation of 15+ LLM integration routes (semantic tokens, preference optimization, reranker, curriculum, hard replay curator, decode policy, query adapter, PairLossPolicy, structural auxiliary, macro configuration planner), **no LLM route demonstrated independent contribution beyond deterministic rule-based baselines**.

The corrected learning-rate schedule (lr=0.0010) was identified through macro configuration search and verified by controlled grid search, improving F1 from 0.2009 to 0.3235.

See [docs/negative.md](docs/negative.md) for complete experimental record.

## Repository

```
main.py
models/  omni.py training.py dataset.py collator.py decode.py mask.py
scripts/ eval.py data.py run.py searchplan.py audiflow.py rerank.py
config/  candidate.yaml
docs/    negative.md architecture.md
utils/   metric.py struct.py
```
