# Benchmark Summary

## Setup

- **Model**: RNA-OmniPrefold (MS-MPRM + PairRefine + pair-aware masking + lr=0.001 + Nussinov)
- **Baseline**: ViennaRNA 2.7.2 (minimum free energy, default parameters)
- **Dataset**: ArchiveII-based local split (282 test samples)
- **Training**: 300 steps, batch=24, single seed=42
- **Hardware**: RTX 4060 laptop GPU

## Results

### Overall (test split, 282 samples)

| Method | F1 | Precision | Recall | Valid | Time/seq (ms) |
|---|---|---|---|---|---|
| **ViennaRNA** | **0.5179** | 0.4734 | 0.6078 | 100% | 11.5 |
| OmniPrefold | 0.3409 | 0.2872 | 0.4532 | 100% | 60.8 |

### Val split (282 samples)

| Method | F1 | Precision | Recall | Valid | Time/seq (ms) |
|---|---|---|---|---|---|
| **ViennaRNA** | **0.5011** | 0.4590 | 0.5841 | 100% | 9.8 |
| OmniPrefold | 0.3220 | 0.2732 | 0.4172 | 100% | 54.4 |

### Length-wise (OmniPrefold, val)

| Length bin | N | Mean F1 |
|---|---|---|
| Short (0-80) | 81 | 0.4177 |
| Medium (80-150) | 137 | 0.3266 |
| Long (150+) | 64 | 0.1909 |

## Analysis

1. **ViennaRNA outperforms RNA-OmniPrefold** by ~1.5x on F1 (0.518 vs 0.341 on test). This is expected: ViennaRNA uses decades of thermodynamic parameter optimization, while OmniPrefold trains at 300 steps from scratch.

2. **Length degradation**: OmniPrefold's F1 drops sharply with sequence length (0.418 → 0.327 → 0.191), suggesting insufficient long-range interaction modeling at 300 training steps.

3. **Both methods achieve 100% valid structures** via Nussinov decoding.

4. **Runtime**: ViennaRNA is ~5x faster (11.5 ms vs 60.8 ms per sequence on GPU).

## Limitations

- Single seed (42), no multi-seed statistics
- 300 training steps only (insufficient for full convergence)
- ArchiveII-based local split, not public benchmark splits
- No multi-seed training, no family-wise breakdown (family field empty)

## Next Steps (if pursued)

1. Train to convergence (1000-5000 steps) for fair comparison with thermodynamics
2. Multi-seed (42, 123, 2024) for mean ± std
3. Public benchmark splits (bpRNA, RNAStrAlign)
4. Additional baselines (LinearFold, CONTRAfold)
5. Family-wise and pseudoknot analysis
