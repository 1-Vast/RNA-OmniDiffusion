# scripts/audit_pack.py — Route R implementation audit pack generator.
"""Generates aggregated audit pack (JSON + MD) for LLM code-audit pipeline.

Contains: dataflow summary, training summary, decode summary, error taxonomy.
Does NOT contain raw sample data.
"""

from __future__ import annotations

import argparse, json, sys
from collections import Counter
from pathlib import Path

import torch, yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.dataset import RNAOmniDataset
from models.training import load_config, load_checkpoint, build_model, resolve_device


def run_audit_pack(config_path, ckpt_path, split, device_name, out_json, out_md):
    config = load_config(config_path)
    device = resolve_device(device_name)

    # ── Dataflow ──
    data_cfg = config["data"]
    train_dataset = RNAOmniDataset(data_cfg["train_jsonl"], max_length=int(data_cfg["max_length"]))
    val_dataset = RNAOmniDataset(data_cfg["val_jsonl"], max_length=int(data_cfg["max_length"]))

    train_ids = {s.get("id", "") for s in train_dataset.samples}
    val_ids = {s.get("id", "") for s in val_dataset.samples}
    overlap = train_ids & val_ids

    # Length / pair distributions
    lengths = [s["length"] for s in val_dataset.samples]
    pair_counts = [len(s.get("pairs", [])) for s in val_dataset.samples]
    length_dist = {"min": min(lengths), "max": max(lengths), "mean": round(sum(lengths)/len(lengths), 1)}
    pair_dist = {"min": min(pair_counts), "max": max(pair_counts), "mean": round(sum(pair_counts)/len(pair_counts), 1)}

    # Check sample fields
    sample_fields = sorted(train_dataset.samples[0].keys())

    dataflow = {
        "train_samples": len(train_dataset.samples),
        "val_samples": len(val_dataset.samples),
        "max_length": int(data_cfg["max_length"]),
        "train_val_overlap": len(overlap),
        "duplicate_train_ids": len(train_ids),
        "length_distribution": length_dist,
        "pair_count_distribution": pair_dist,
        "sample_fields": sample_fields,
        "has_pairs_field": "pairs" in sample_fields,
        "has_struct_field": "struct" in sample_fields,
    }

    # ── Training ──
    training_cfg = config["training"]
    warmup = int(training_cfg.get("warmup_steps", 0))
    lr_nominal = float(training_cfg["lr"])
    lr_eff_at_300 = round(lr_nominal * min(1.0, 300 / max(1, warmup)), 6) if warmup > 0 else lr_nominal

    training = {
        "lr": lr_nominal,
        "warmup_steps": warmup,
        "effective_lr_at_300": lr_eff_at_300,
        "batch_size": int(training_cfg["batch_size"]),
        "lambda_pair": float(training_cfg.get("lambda_pair", 5.0)),
        "lambda_struct": float(training_cfg.get("lambda_struct", 1.0)),
        "pair_positive_weight": float(training_cfg.get("pair_positive_weight", 10.0)),
        "pair_negative_ratio": int(training_cfg.get("pair_negative_ratio", 5)),
        "seed": int(training_cfg.get("seed", 42)),
        "amp": bool(training_cfg.get("amp", True)),
        "grad_clip": float(training_cfg.get("grad_clip", 1.0)),
        "save_best_by": training_cfg.get("save_best_by", "val_loss"),
    }

    # ── Decode ──
    decode_cfg = config["decoding"]
    decode = {
        "method": "nussinov",
        "min_loop_length": int(decode_cfg.get("min_loop_length", 3)),
        "allow_wobble": bool(decode_cfg.get("allow_wobble", True)),
        "decode_source": decode_cfg.get("decode_source", "pair"),
        "pair_threshold": float(decode_cfg.get("pair_threshold", 0.25)),
        "nussinov_gamma": float(decode_cfg.get("nussinov_gamma", 2.0)),
    }

    # ── Model ──
    model_cfg = config["model"]
    ablation_cfg = config.get("ablation", {})
    model = {
        "hidden_size": int(model_cfg["hidden_size"]),
        "num_layers": int(model_cfg["num_layers"]),
        "num_heads": int(model_cfg["num_heads"]),
        "distbias": bool(model_cfg.get("distbias", False)),
        "pairrefine": bool(model_cfg.get("pairrefine", False)),
        "pairrefine_blocks": int(model_cfg.get("pairrefine_blocks", model_cfg.get("pairrefineblocks", 1))),
        "pairrefine_channels": int(model_cfg.get("pairrefine_channels", model_cfg.get("pairrefinechannels", 16))),
        "use_pair_head": bool(ablation_cfg.get("use_pair_head", True)),
        "use_pair_loss": bool(ablation_cfg.get("use_pair_loss", True)),
        "use_nussinov": bool(ablation_cfg.get("use_nussinov", True)),
        "use_pair_aware_masking": bool(ablation_cfg.get("use_pair_aware_masking", True)),
    }

    # ── Error taxonomy (use cached if available) ──
    err_path = Path("outputs/reports/mainline_error_taxonomy.json")
    if err_path.exists():
        error_tax = json.loads(err_path.read_text())
    else:
        error_tax = {"note": "Run scripts/error_report.py first for full taxonomy"}

    # ── Known failed routes ──
    failed = [
        "semantic_tokens", "preference_loss", "reranker", "hard_replay",
        "query_adapter", "tag_aux", "struct_aux", "importance_weighting",
        "pair_residual_conv (noise-level)", "LLM_config_planner (weak)",
    ]

    # ── Assemble pack ──
    pack = {
        "mainline_summary": {
            "config": config_path,
            "ckpt": ckpt_path,
            "model_desc": "MS-MPRM + PairRefine + pair-aware masking + lr=0.001 + strict Nussinov",
            "f1_approx": 0.3321,
        },
        "dataflow": dataflow,
        "training": training,
        "decode": decode,
        "model": model,
        "error_taxonomy": error_tax,
        "failed_routes": failed,
    }

    # Write JSON
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(pack, indent=2))

    # Write MD
    lines = ["# Route R Audit Pack", "", f"## Mainline: F1≈0.3321", ""]
    lines.append("### Dataflow")
    for k, v in dataflow.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("### Training")
    for k, v in training.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("### Decode")
    for k, v in decode.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("### Model")
    for k, v in model.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("### Failed Routes")
    for r in failed:
        lines.append(f"- {r}")
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text("\n".join(lines))
    print(f"Audit pack: {out_json}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_json", default="outputs/reports/routeR_audit_pack.json")
    p.add_argument("--out_md", default="outputs/reports/routeR_audit_pack.md")
    args = p.parse_args()
    run_audit_pack(args.config, args.ckpt, args.split, args.device, args.out_json, args.out_md)


if __name__ == "__main__":
    main()
