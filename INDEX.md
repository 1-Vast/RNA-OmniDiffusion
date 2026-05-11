# RNA-OmniPrefold Index

## Core

- `main.py`: CLI entry point.
- `models/omni.py`: RNA encoder, pair head, PairRefine blocks.
- `models/training.py`: Config loading, training loop, loss computation.
- `models/dataset.py`: JSONL dataset loader.
- `models/collator.py`: Batch construction, pair-aware masking.
- `models/decode.py`: Strict Nussinov decoder.
- `models/mask.py`: Masking utilities.
- `models/pair_heads.py`: Optional pair-logit residual conv module.
- `utils/metric.py`: Pair F1, structure evaluation.
- `utils/struct.py`: Dot-bracket parsing.
- `utils/slices.py`: Error taxonomy slicers.

## Configs

- `config/mainline_lr0010.yaml`: **Validated mainline** (MS-MPRM + PairRefine + pair-aware masking + lr=0.001 + Nussinov).
- `config/candidate.yaml`: Canonical config (do not edit).
- `config/archP_hand_*.yaml`: Route P architecture variant configs (distance bucket, long-range gate, stem refine).
- `config/stem_refine_*.yaml`: Stem refine ablation configs (off, random_conv, verify_300, diag).

## Scripts

- `scripts/eval.py`: Benchmark / evaluation.
- `scripts/sliceeval.py`: Error taxonomy slice evaluation.
- `scripts/audit.py`: Repository audit and cleanup.
- `scripts/searchplan.py`: LLM macro config proposer (experimental, not model module).

## Docs

- `docs/mainline.md`: Validated mainline specification.
- `docs/architecture.md`: Architecture and validated/non-validated components.
- `docs/negative.md`: All negative / inconclusive / not-independent routes.
- `docs/llm_experiments.md`: LLM experiment summary (final conclusion: no model contribution).
- `docs/importance.md`: Route O structural importance report (confirmed negative).

## Mainline Commands

```bash
python main.py smoke
python main.py train --config config/mainline_lr0010.yaml --device cuda --max_steps 300
python scripts/eval.py bench --config config/mainline_lr0010.yaml --ckpt outputs/mainline_lr0010/best.pt --split val --device cuda --decode nussinov --stage_logits
```
