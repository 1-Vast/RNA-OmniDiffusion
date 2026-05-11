# Benchmark Summary

## Setup
- **Hardware**: RTX 4060 laptop GPU
- **Dataset**: ArchiveII-based local split (train=1316, val=282, test=282)

## Results

### Overall (test split, 282 samples)

| Method | Steps | Test F1 | Precision | Recall | Valid | Time/seq |
|---|---|---|---|---|---|---|
| **ViennaRNA 2.7.2** | — | **0.5179** | 0.4734 | 0.6078 | 100% | 11.5ms |
| OmniPrefold | 300 | 0.3409 | 0.2872 | 0.4532 | 100% | 60.8ms |
| OmniPrefold | 1000 | 0.3472 | — | — | 100% | — |
| OmniPrefold | 3000 | 0.3285 | — | — | 100% | — |

### Step scaling (val split)

| Steps | Best Val F1 | Best Epoch |
|---|---|---|
| 300 | 0.2963 | 5 |
| 1000 | 0.3157 | 10 |
| 3000 | 0.3066 | 6 |

### Length-wise (OmniPrefold 300-step, val)

| Length bin | N | Mean F1 |
|---|---|---|
| Short (0-80) | 81 | 0.4177 |
| Medium (80-150) | 137 | 0.3266 |
| Long (150+) | 64 | 0.1909 |

## Analysis

1. **ViennaRNA is ~1.52x better** than OmniPrefold on test F1 (0.518 vs 0.341 at 300 steps).
2. **Training steps beyond 300 do not help**: F1 plateaus around 0.33-0.35 regardless of training budget (300, 1000, 3000 steps).
3. **Long-sequence degradation** is the primary failure mode: F1 drops from 0.418 (short) to 0.191 (long).
4. **Valid structure rate is 100%** for both methods via Nussinov decoding.
5. **The neural model is structurally limited** — current architecture/data may be insufficient to learn thermodynamic folding principles that ViennaRNA encodes via hand-tuned energy parameters.

## Limitations
- Single seed (42); no multi-seed statistics
- Family field unavailable in local dataset (all "OTHER")
- Only ArchiveII-based local split; no public benchmark splits
- No LinearFold/CONTRAfold/EternaFold comparison (not installed)

