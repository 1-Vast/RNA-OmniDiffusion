# Experiment Summary

Final conclusions from systematic evaluation of model components and LLM integration routes.

## Validated Components

| Component | Impact | Decision |
|---|---|---|
| Corrected lr (0.0001→0.001) | +0.13 F1 | **Keep** — largest single improvement |
| PairRefine (1 block, 16ch) | +0.01-0.02 F1 | **Keep** |
| Pair-aware masking | Small but real | **Keep** |
| Strict Nussinov decode | Ensures 100% valid structures | **Keep** |
| Distance bias | Marginal but harmless | **Keep** (default enabled) |

## Optional / Non-Default Components

| Component | Impact | Decision |
|---|---|---|
| Pair residual conv (pair_residual) | +0.01 F1 in some runs, not stable | **Optional, default disabled** |
| PairRefine depth/channel variants | No clear gain at 300 steps | Not recommended |

## Negative / Removed Components

| Route | Impact | Decision |
|---|---|---|
| RNA-FM distillation | No contribution beyond D-only | Removed |
| LLM semantic tokens | F1 dropped 0.57→0.39 | Removed |
| LLM preference / RAG preference | No benefit | Removed |
| LLM reranker | Worse than rule reranker | Removed |
| LLM hard replay curator | LLM = rule | Removed |
| LLM query adapter | No improvement | Removed |
| Structural tag auxiliary | Degrades F1 | Removed |
| PairLossPolicy | No effect | Disabled |
| Structural importance (Route O) | F1 reduced by 0.005 | Negative |
| LLM config planner (Route Q) | Within training noise | Not independent |
| LLM implementation audit (Route R) | Found 0 issues; hand found 2 | Not independent |
| LLM causal reasoning (Route S) | LLM ≈ hand | Not independent |
| Stem continuity refine claim | Gain = random conv (capacity, not bias) | Renamed |

## LLM Final Conclusion

Across 5 routes (O, P, Q, R, S) tested under controlled comparison, LLM has **not demonstrated independent value** as a model-performance module, configuration planner, implementation auditor, or causal reasoning planner. The strongest and simplest configuration is a clean non-LLM pipeline.

## Current Best Configuration

```
MS-MPRM + PairRefine + pair-aware masking + lr=0.001 + strict Nussinov
```
- Val F1: ~0.332
- Config: `config/mainline_strongest.yaml`
