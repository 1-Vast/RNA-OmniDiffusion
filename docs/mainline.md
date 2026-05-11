# Validated Mainline

RNA-OmniPrefold validated model framework after systematic cleanup of invalid/negative modules.

## Configuration

```
MS-MPRM + PairRefine + pair-aware masking + corrected lr schedule + strict Nussinov decode
```

### Fixed Parameters

| Parameter | Value |
|---|---|
| Learning rate | 0.001 |
| Warmup steps | 50 |
| Max steps (default) | 300 |
| Batch size | 24 |
| Pair loss weight (λ_pair) | 5.0 |
| Pair positive weight | 10.0 |
| Pair negative ratio | 5 |
| Dropout | 0.1 |
| Hidden size | 512 |
| Layers | 8 |
| Heads | 8 |
| Pair head | MLP |
| PairRefine | True |
| Distance bias | True |
| Strict Nussinov decode | True |

### Metrics (Val Split, 300 Steps, lr=0.001, warmup=50)

| Metric | Value |
|---|---|
| Pair F1 | 0.3184 |
| Pair Precision | 0.2696 |
| Pair Recall | 0.4168 |
| Valid Structure Rate | 1.00 |

## Disabled/Removed Modules

All modules below were systematically tested and found to be invalid, negative, or inconclusive. They are either removed from the codebase or default-disabled in config.

| Module | Status | Reason |
|---|---|---|
| RNA-FM distillation | Removed | No contribution beyond D-only baseline |
| LLM semantic tokens | Removed | Dropped F1 from 0.572 → 0.385 |
| LLM preference loss | Removed | No benefit over no-pref baseline |
| LLM reranker | Removed | F1=0.1025 < Rule=0.1223 |
| LLM hard replay curator | Removed | LLM replay = rule replay |
| LLM query adapter | Removed | No improvement over mainline |
| Structural tag auxiliary | Removed | Degrades F1 or neutral |
| PairLossPolicy | Disabled | Weight changes do not affect F1 |
| Structural importance weighting | Disabled | Reduces F1 by ~0.005 |
| Typed data training | Removed | No demonstrated benefit |
| Coach policy | Removed | No demonstrated benefit |

## Config File

`config/mainline_lr0010.yaml` — canonical validated config. All invalid modules are not referenced.

## Quick Start

```bash
python main.py smoke
python main.py train --config config/mainline_lr0010.yaml --device cuda --max_steps 300
python scripts/eval.py bench --config config/mainline_lr0010.yaml --ckpt outputs/mainline_lr0010/best.pt --split val --device cuda --decode nussinov --stage_logits
```

## Architecture Variants (Route P)

Three hand-proposed pair-head architecture variants were tested at 300 steps:

| Architecture | Val F1 | Δ mainline |
|---|---|---|
| Mainline (base) | 0.3184 | — |
| Distance bucket head | 0.3150 | -0.0034 |
| Long-range gate | 0.3102 | -0.0082 |
| **Stem continuity refine** | **0.3340** | **+0.0156** |

Stem continuity refine improves F1 by +0.0156, with the largest gains in fragmented_stem_cases (+0.037) and low_gc_sequences (+0.025).
