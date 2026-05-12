# Local Cleanup Report

## Date
2026-05-12

## Branch / Commit
- Branch: main
- Commit: f9c3a03 (Scaled14L)

## Deleted
- All outputs/ (27 model checkpoint directories)
- Redundant configs (bias_*, coach, ctrl_rule, LLM route configs)
- Deprecated scripts (moved to scripts/deprecated/)
- Redundant utils (moved to utils/deprecated/)
- Redundant models (moved to models/deprecated/)

## Preserved
- Core code: models/omni.py, training.py, dataset.py, collator.py, decode.py, token.py
- Core scripts: eval.py, data.py, run.py, audit.py
- Core utils: metric.py, struct.py, reward.py, rna_priors.py
- Mainline configs: mainline_strongest.yaml, mainline_prefold.yaml (14L/640H/3xPR)
- Essential docs: mainline.md, benchmark_summary.md, calibration_analysis.md, experiments_summary.md

## .env
Not present locally (remote only). Not committed.

## candidate.yaml
No diff.

## Current Mainline
```
RNA-OmniPrefold 14L/640H/3xPairRefine
F1 = 0.4652 (local ArchiveII test, seed42)
ViennaRNA gap = 0.053
```
