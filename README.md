# RNA-OmniPrefold

Relation-aware Masked Denoising for Constraint-Guided RNA Folding.

## Validated Mainline

```
MS-MPRM + PairRefine + pair-aware masking + corrected lr schedule + strict Nussinov decode
```

| Config | lr | Steps | Val F1 | Split |
|---|---|---|---|---|
| **mainline_lr0010** | **0.001** | **300** | **0.3184** | val |
| mainline + pair conv residual | 0.001 | 300 | 0.3312 | val |

## Quick Start

```bash
python main.py overview
python main.py smoke
python main.py train --config config/mainline_lr0010.yaml --device cuda --max_steps 300
python scripts/eval.py bench --config config/mainline_lr0010.yaml --ckpt outputs/mainline_lr0010/best.pt --split val --device cuda --decode nussinov --stage_logits
```

## LLM Status

After systematic evaluation of 16+ LLM integration routes (semantic tokens, preference optimization, reranker, curriculum, hard replay curator, decode policy, query adapter, PairLossPolicy, structural auxiliary, macro configuration planner, structural importance masking, **architecture hypothesis generator**), **no LLM route demonstrated independent contribution beyond deterministic rule-based baselines or hand-designed architectures**.

The corrected learning-rate schedule (lr=0.0010, warmup=50) was the most impactful single change. A small 2D conv residual on pair logits (+10 params) improves val F1 by +0.0128, but this is a capacity effect (random conv matches anti-diagonal conv), not a structural inductive bias.

See [docs/negative.md](docs/negative.md) and [docs/mainline.md](docs/mainline.md) for complete details.

## Repository

```
main.py
models/  omni.py training.py dataset.py collator.py decode.py mask.py pair_heads.py
scripts/ eval.py sliceeval.py searchplan.py
config/  mainline_lr0010.yaml candidate.yaml
docs/    mainline.md negative.md importance.md
utils/   metric.py struct.py slices.py
```
