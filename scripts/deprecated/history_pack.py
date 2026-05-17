# scripts/history_pack.py — Route S: structured experiment history for causal reasoning.
"""Compiles all known experimental results into a structured JSON pack
for LLM causal reasoning. Quasi-static — updated when new results appear."""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build_history_pack() -> dict:
    return {
        "current_strongest_mainline": {
            "model": "MS-MPRM + PairRefine + pair-aware masking + lr=0.001 + strict Nussinov",
            "val_f1": 0.3321,
            "val_split": 282,
            "train_split": 1316,
            "lr": 0.001,
            "warmup_steps": 50,
            "effective_lr_at_300": 0.001,
            "batch_size": 24,
            "lambda_pair": 5.0,
            "lambda_struct": 1.0,
            "pair_positive_weight": 10.0,
            "pair_negative_ratio": 5,
            "dropout": 0.1,
            "decode": "strict Nussinov, min_loop=3, gamma=2.0, allow_wobble=true",
            "status": "strongest clean baseline, no modifications exceed it",
        },

        "positive_findings": [
            {
                "id": "corrected_lr",
                "description": "Old lr=0.0001 with warmup=500 caused severe under-training (effective lr ~0.00006 at step 300). Correcting to lr=0.001 improved F1 from ~0.20 to ~0.33.",
                "impact": "largest single change (+0.13 F1)",
                "mechanism": "optimization dynamics — higher lr reaches better loss basin at 300 steps",
            },
            {
                "id": "pairrefine",
                "description": "PairRefine (1 block, 16 channels) provides small 2D conv refinement on pair logits. Ablation confirms it helps vs no-refine.",
                "impact": "moderate (~0.01-0.02 F1)",
                "mechanism": "2D spatial smoothing on pair logit field",
            },
            {
                "id": "pair_aware_masking",
                "description": "Masking strategy that considers pair structures during token masking. MS-MPRM multi-scale masking.",
                "impact": "small but real",
                "mechanism": "Better masking distribution for pair structure learning",
            },
            {
                "id": "nussinov_decode",
                "description": "Strict Nussinov DP decode ensures valid non-crossing structures. 100% valid rate.",
                "impact": "ensures validity",
                "mechanism": "Dynamic programming projection onto valid structure space",
            },
            {
                "id": "min_loop_consistency_issue",
                "description": "Route R hand audit found 28 val-set pairs with |i-j| < decode min_loop=3. Model trains on pairs it cannot decode.",
                "impact": "minor/medium (small count, but potential F1 cap)",
                "mechanism": "Train/eval constraint mismatch in pair distance filtering",
                "status": "not yet exploited — no fix applied",
            },
        ],

        "negative_findings": [
            {"route": "semantic_tokens", "result": "F1 dropped 0.572->0.385", "mechanism": "coarse input pollution"},
            {"route": "preference_loss", "result": "no benefit over no-pref baseline", "mechanism": "dwarfed by BCE pair loss (lambda=5.0)"},
            {"route": "reranker", "result": "LLM=0.1025 < Rule=0.1223", "mechanism": "LLM cannot rank candidate structures"},
            {"route": "query_adapter", "result": "no improvement", "mechanism": "cross-attention to encoder adds noise"},
            {"route": "tag_aux", "result": "degrades F1", "mechanism": "auxiliary loss interferes with pair BCE"},
            {"route": "importance_weighting", "result": "F1 reduced by 0.005", "mechanism": "importance scores lack variance, near-uniform weights"},
            {"route": "PairLossPolicy", "result": "weight changes don't affect F1", "mechanism": "vectorized but BCE dominates"},
            {"route": "stem_continuity_refine", "result": "gain = random conv (capacity, not bias)", "mechanism": "10-param 2D conv helps regardless of init"},
            {"route": "LLM_config_planner (Q)", "result": "best LLM proposal +0.005 then -0.008 (noise)", "mechanism": "LLM config suggestions within run-to-run noise"},
            {"route": "LLM_combo_test (Q)", "result": "LLM tune + residual < fresh mainline", "mechanism": "noise-level; fresh mainline retrain = highest"},
            {"route": "LLM_audit (R)", "result": "LLM found 0 verifiable issues; hand found 2", "mechanism": "LLM audit hypotheses too generic"},
        ],

        "confound_corrections": [
            {"id": "low_label_confound", "description": "Low10 appearing better was epoch-exposure artifact"},
            {"id": "importance_subset_confound", "description": "Route O rule importance 0.3331 was training-subset confound; _imp not consumed"},
            {"id": "stem_bias_confound", "description": "Stem refine gain = random conv gain; not stem-specific"},
            {"id": "noise_level_gains", "description": "LLM tune and residual gains fell within run-to-run noise when mainline was freshly retrained (0.3321)"},
        ],

        "error_taxonomy_highlights": {
            "short_range": {"precision": 0.19, "recall": 0.53, "f1": 0.28},
            "long_range": {"precision": 0.26, "recall": 0.30, "f1": 0.28},
            "over_pairing": {"samples": 132, "f1": 0.235},
            "fragmented_stem": {"samples": 16, "f1": 0.09},
            "sparse_pairs": {"samples": 38, "f1": 0.14},
            "pair_count_bias": "+18 predicted vs true",
        },

        "allowed_intervention_classes": [
            "constraint_consistency_fix",
            "decode_calibration",
            "schedule_refinement",
            "pairrefine_capacity_sweep",
            "pair_loss_balance_sweep",
            "data_filtering_or_constraint_alignment",
            "checkpoint_selection_or_early_stopping",
            "evaluation_slice_targeted_fix",
        ],

        "forbidden_interventions": [
            "LLM module", "semantic token", "preference", "reranker",
            "sample weighting", "synthetic data", "new backbone", "external encoder",
            "importance scoring", "query adapter", "tag auxiliary",
        ],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mainline_config", default="config/mainline_strongest.yaml")
    p.add_argument("--mainline_ckpt", default="outputs/mainline_strongest/best.pt")
    p.add_argument("--split", default="val")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_json", default="outputs/reports/routeS_history_pack.json")
    p.add_argument("--out_md", default="outputs/reports/routeS_history_pack.md")
    args = p.parse_args()

    pack = build_history_pack()

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(pack, indent=2))

    # MD
    lines = ["# Route S: Experiment History Pack", ""]
    lines.append("## Current Strongest Mainline")
    for k, v in pack["current_strongest_mainline"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Positive Findings")
    for pf in pack["positive_findings"]:
        lines.append(f"- **{pf['id']}**: {pf['description']} (impact: {pf.get('impact','?')})")
    lines.append("")
    lines.append("## Negative Findings")
    for nf in pack["negative_findings"]:
        lines.append(f"- **{nf['route']}**: {nf['result']}")
    lines.append("")
    lines.append("## Confound Corrections")
    for cc in pack["confound_corrections"]:
        lines.append(f"- {cc['id']}: {cc['description']}")
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(lines))
    print(f"History pack: {args.out_json}")


if __name__ == "__main__":
    main()
