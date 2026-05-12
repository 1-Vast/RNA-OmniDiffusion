# scripts/routeS_materialize.py — Route S: convert causal hypotheses to experiments.
"""Reads hypotheses JSON, outputs YAML configs and eval-only tests."""

from __future__ import annotations

import argparse, json, sys, yaml
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def materialize_experiments(hyps: list[dict], base_config: dict, group: str,
                             out_root: Path, report_lines: list):
    """Convert hypotheses to concrete experiments."""
    experiments = []
    seen = set()

    for h in hyps:
        exp = h.get("minimal_experiment", {})
        inter_class = exp.get("intervention_class", "unknown")
        name = h.get("name", "unknown").replace(" ", "_")

        # Skip duplicates
        if name in seen: continue
        seen.add(name)

        cfg = deepcopy(base_config)
        cfg["training"]["output_dir"] = f"outputs/routeS_{group}_{name}"

        # Apply config changes based on intervention class
        if inter_class == "constraint_consistency_fix":
            # No config change — this is a code change (loss mask)
            cfg["_routeS_note"] = "exclude |i-j|<min_loop from pair loss mask"
            experiments.append({
                "name": name, "group": group, "hypothesis": h["causal_claim"],
                "intervention_class": inter_class,
                "config_path": str(out_root / group / f"{name}.yaml"),
                "needs_training": True,
                "needs_code_change": True,
            })
            continue

        elif inter_class == "decode_calibration":
            # Eval-only: change gamma
            gamma_val = exp.get("config_or_code_change", "")
            if "gamma" in str(gamma_val).lower():
                import re
                nums = re.findall(r'[\d.]+', str(gamma_val))
                if nums:
                    cfg.setdefault("decoding", {})["nussinov_gamma"] = float(nums[0])
            cfg["training"]["output_dir"] = f"outputs/routeS_{group}_{name}"
            experiments.append({
                "name": name, "group": group, "hypothesis": h["causal_claim"],
                "intervention_class": inter_class,
                "config_path": str(out_root / group / f"{name}.yaml"),
                "needs_training": False, "needs_code_change": False,
            })

        elif inter_class == "schedule_refinement":
            if "warmup" in exp.get("config_or_code_change", "").lower():
                cfg["training"]["warmup_steps"] = 0
            if "lr" in exp.get("config_or_code_change", "").lower():
                import re
                nums = re.findall(r'[\d.]+', str(exp.get("config_or_code_change", "")))
                for n in nums:
                    v = float(n)
                    if 0.0005 <= v <= 0.002:
                        cfg["training"]["lr"] = v; break
            experiments.append({
                "name": name, "group": group, "hypothesis": h["causal_claim"],
                "intervention_class": inter_class,
                "config_path": str(out_root / group / f"{name}.yaml"),
                "needs_training": True, "needs_code_change": False,
            })

        elif inter_class == "pairrefine_capacity_sweep":
            cfg["model"]["pairrefine_blocks"] = 2
            cfg["model"]["pairrefine_channels"] = 16
            experiments.append({
                "name": name, "group": group, "hypothesis": h["causal_claim"],
                "intervention_class": inter_class,
                "config_path": str(out_root / group / f"{name}.yaml"),
                "needs_training": True, "needs_code_change": False,
            })

        elif inter_class == "pair_loss_balance_sweep":
            cfg["training"]["lambda_pair"] = 7.0
            experiments.append({
                "name": name, "group": group, "hypothesis": h["causal_claim"],
                "intervention_class": inter_class,
                "config_path": str(out_root / group / f"{name}.yaml"),
                "needs_training": True, "needs_code_change": False,
            })

        elif inter_class == "checkpoint_selection_or_early_stopping":
            cfg["training"]["save_best_by"] = "val_pair_f1"
            experiments.append({
                "name": name, "group": group, "hypothesis": h["causal_claim"],
                "intervention_class": inter_class,
                "config_path": str(out_root / group / f"{name}.yaml"),
                "needs_training": True, "needs_code_change": False,
            })

        else:
            experiments.append({
                "name": name, "group": group, "hypothesis": h["causal_claim"],
                "intervention_class": inter_class,
                "config_path": str(out_root / group / f"{name}.yaml"),
                "needs_training": True, "needs_code_change": False,
            })

        # Write config
        path = Path(cfg["training"]["output_dir"])
        cfg_path = out_root / group / f"{name}.yaml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    # Report
    report_lines.append(f"## {group}")
    report_lines.append(f"| Experiment | Needs Train | Code Change | Intervention |")
    report_lines.append("|---|---|---|---|")
    for e in experiments:
        report_lines.append(f"| {e['name']} | {e['needs_training']} | {e['needs_code_change']} | {e['intervention_class']} |")
    report_lines.append("")

    return experiments


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hand", default="outputs/reports/routeS_hand_hypotheses.json")
    p.add_argument("--random", default="outputs/reports/routeS_random_hypotheses.json")
    p.add_argument("--llm", default="outputs/reports/routeS_llm_hypotheses.json")
    p.add_argument("--base_config", default="config/mainline_strongest.yaml")
    p.add_argument("--out_root", default="config/routeS")
    p.add_argument("--report", default="outputs/reports/routeS_materialized.md")
    args = p.parse_args()

    base_config = load_yaml(Path(args.base_config).resolve())
    out_root = Path(args.out_root)
    report_lines = ["# Route S: Materialized Experiments", ""]

    all_exps = []
    for source, group in [(args.hand, "hand"), (args.random, "random"), (args.llm, "llm")]:
        path = Path(source)
        if not path.exists():
            report_lines.append(f"## {group} — missing ({source})")
            continue
        hyps = json.loads(path.read_text())
        exps = materialize_experiments(hyps, base_config, group, out_root, report_lines)
        all_exps.extend(exps)

    # Summary
    train_count = sum(1 for e in all_exps if e["needs_training"])
    eval_count = sum(1 for e in all_exps if not e["needs_training"])
    report_lines.append("## Summary")
    report_lines.append(f"- Total experiments: {len(all_exps)}")
    report_lines.append(f"- Needs training: {train_count}")
    report_lines.append(f"- Eval-only: {eval_count}")

    out_dir = Path(args.report).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    report = "\n".join(report_lines)
    Path(args.report).write_text(report)
    print(report)


if __name__ == "__main__":
    main()
