# Final Cleanup Report

## Phase 1: Name Unification

No action needed — codebase already uses "RNA-OmniPrefold" throughout. No "RNA-OmniDistill" references found.

## Phase 2: Outputs & Artifacts Deletion

| Path | Action | Reason |
|---|---|---|
| `outputs/` (46 GB, 101 dirs) | **Deleted** | Historical experiment outputs |
| `dataset/derived/` (13 files) | **Deleted** | Generated importance/replay/synth data |
| `scripts/_llm_probe.py` | **Deleted** | Temporary LLM test script |
| `outputs/scr_*.txt` | **Deleted** | Scratch training logs |
| `outputs/llm_*.json` | **Deleted** | LLM cache/proposal files |
| `dataset/archive/` | **Kept** | Real training/val/test data |
| `.env` | **Kept** | Required for LLM credential storage |

## Phase 3: Configs

| Config | Status |
|---|---|
| `config/mainline_strongest.yaml` | **Kept** — main validated config |
| `config/mainline_prefold.yaml` | **Created** — alias for strongest |
| `config/candidate.yaml` | **Untouched** |
| `config/archP_*.yaml`, `config/routeQ_*.yaml`, etc. | **Kept** — archived experiment configs |
| `config/stem_refine_*.yaml` | **Kept** — ablation configs |

## Phase 4: Docs

| Doc | Status |
|---|---|
| `README.md` | **Rewritten** — clean non-LLM mainline |
| `INDEX.md` | **Updated** — current structure |
| `docs/mainline.md` | **Kept** — validated mainline |
| `docs/architecture.md` | **Kept** — architecture |
| `docs/negative.md` | **Kept** — all failed routes |
| `docs/experiments_summary.md` | **Created** — conclusions |
| `docs/llm_experiments.md` | **Kept** — archived LLM results |
| `docs/importance.md` | **Kept** — Route O detail |
| `docs/final_cleanup_report.md` | **This file** |

## Phase 5: Verification

- `compileall`: Pass
- `smoke`: Pass (smoke_ok)
- `candidate.yaml`: No diff
- `.env`: Exists, not staged
- `outputs/`: Deleted
- No large files staged
