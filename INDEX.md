# RNA-OmniPrefold Index

## Core

- `main.py`: CLI entry point (train, eval, infer, smoke, overview, params).
- `models/omni.py`: RNA encoder, pair head, PairRefine blocks.
- `models/training.py`: Config loading, training loop, loss computation.
- `models/dataset.py`: JSONL dataset loader.
- `models/collator.py`: Batch construction, pair-aware masking.
- `models/decode.py`: Strict Nussinov decoder.
- `models/pair_heads.py`: Optional pair-logit residual conv module.

## Configs

- `config/mainline_strongest.yaml`: **Validated mainline** (MS-MPRM + PairRefine + lr=0.001 + Nussinov).
- `config/candidate.yaml`: Canonical config (do not edit).

## Scripts

- `scripts/eval.py`: Benchmark evaluation.
- `scripts/audit.py`: Repository audit and cleanup.

## Utils

- `utils/metric.py`: Pair F1, structure evaluation.
- `utils/struct.py`: Dot-bracket parsing.

## Docs

- `docs/mainline.md`: Validated mainline specification.
- `docs/architecture.md`: Architecture and components.
- `docs/negative.md`: All negative / inconclusive routes.
- `docs/experiments_summary.md`: Experiment conclusions.

## Mainline Commands

```bash
python main.py smoke
python main.py train --config config/mainline_strongest.yaml --device cuda --max_steps 300
python scripts/eval.py bench --config config/mainline_strongest.yaml --ckpt outputs/<dir>/best.pt --split val --device cuda --decode nussinov --stage_logits
```
