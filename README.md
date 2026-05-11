# RNA-OmniPrefold

A lightweight RNA secondary-structure prediction framework built around MS-MPRM (Multi-Scale Masked Pair-Relation Modeling), PairRefine, pair-aware masking, and strict Nussinov decoding.

## Validated Mainline

```
RNA sequence → Transformer encoder (MS-MPRM) → Pair head → PairRefine →
pair-aware masking → pair probability matrix → strict Nussinov decode →
valid dot-bracket structure
```

| Component | Status |
|---|---|
| MS-MPRM backbone | Enabled |
| PairRefine (2D conv refinement) | Enabled |
| Pair-aware masking | Enabled |
| Distance bias | Enabled |
| Strict Nussinov decode | Enabled |
| Corrected lr=0.001, warmup=50 | Enabled |
| LLM / semantic / preference / reranker | **Disabled** |

## Quick Start

```bash
python main.py smoke
python main.py train --config config/mainline_strongest.yaml --device cuda --max_steps 300
python scripts/eval.py bench --config config/mainline_strongest.yaml --ckpt outputs/<dir>/best.pt --split val --device cuda --decode nussinov --stage_logits
```

## Key Results

| Config | lr | Steps | Val F1 |
|---|---|---|---|
| mainline_strongest | 0.001 | 300 | ~0.332 |

## LLM Status

16+ LLM integration routes were systematically evaluated (semantic tokens, preference optimization, reranker, curriculum, hard replay curator, decode policy, query adapter, PairLossPolicy, structural auxiliary, macro configuration planner, structural importance masking, architecture hypothesis generator, config search planner, implementation audit, causal reasoning planner). **No route demonstrated independent performance gains beyond deterministic rule-based baselines or hand-designed controls under controlled comparison.**

LLM is not part of the validated training or inference pipeline. It may remain useful only as an offline experiment assistant.

See `docs/negative.md` for the complete experimental record.

## Repository

```
main.py
config/
  mainline_strongest.yaml        # Validated mainline config
  candidate.yaml                  # Canonical config (do not edit)
models/
  omni.py                         # RNA encoder, pair head, PairRefine
  pair_heads.py                   # Optional pair-head variants (default disabled)
  decode.py                       # Strict Nussinov decoder
  training.py                     # Config loading, training loop, loss
  dataset.py                      # JSONL dataset loader
  collator.py                     # Batch construction, pair-aware masking
utils/
  metric.py                       # Pair F1, structure evaluation
  struct.py                       # Dot-bracket parsing
scripts/
  eval.py                         # Benchmark / evaluation
  audit.py                        # Repository audit and cleanup
docs/
  mainline.md                     # Validated mainline specification
  architecture.md                 # Architecture and components
  negative.md                     # All negative/inconclusive routes
  experiments_summary.md          # Experiment conclusions
```
