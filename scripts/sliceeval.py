# scripts/sliceeval.py — Error taxonomy slicer for RNA structure predictions.
"""Evaluate model predictions across structural error slices."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.dataset import RNAOmniDataset
from models.decode import generate_structure_seq2struct, nussinov_decode
from models.training import load_config, load_checkpoint, build_model, resolve_device
from utils.metric import base_pair_f1
from utils.slices import SLICE_REGISTRY, dominant_error


def run_slice_eval(config_path: str, ckpt_path: str, split: str, device_name: str,
                   decode: str, out: str):
    config = load_config(config_path)
    device = resolve_device(device_name)

    ckpt_config, tokenizer, checkpoint = load_checkpoint(ckpt_path, device)
    model = build_model(ckpt_config, tokenizer, device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    jsonl_key = {"train": "train_jsonl", "val": "val_jsonl", "test": "test_jsonl"}[split]
    dataset = RNAOmniDataset(config["data"][jsonl_key],
                             max_length=int(config["data"]["max_length"]))

    rows = []
    for sample in dataset.samples:
        seq = sample["seq"]
        true_struct = sample.get("struct", "")
        pred_struct = generate_structure_seq2struct(
            model, tokenizer, seq, config["decoding"], device)
        f1 = base_pair_f1(pred_struct, true_struct)
        from utils.metric import base_pair_precision, base_pair_recall
        prec = base_pair_precision(pred_struct, true_struct)
        rec = base_pair_recall(pred_struct, true_struct)
        rows.append({
            "id": sample.get("id", ""), "seq": seq,
            "true_struct": true_struct, "pred_struct": pred_struct,
            "pair_f1": f1, "pair_precision": prec, "pair_recall": rec,
            "length": sample["length"], "pair_count": len(sample.get("pairs", [])),
            "sample": sample,
        })

    # Build slice tables
    report_lines = ["# Slice Evaluation", f"split={split} samples={len(rows)}", ""]
    report_lines.append("| Slice | N | F1 | Precision | Recall | Pair Ratio | Dominant Error |")
    report_lines.append("|---|---:|---:|---:|---:|---:|---|")

    slice_results = {}
    # Pair-based slices need pairs, others need sample only
    _pair_slices = {"short_range_pairs", "medium_range_pairs", "long_range_pairs", "canonical_pairs"}
    _pred_slices = {"under_pairing_cases", "over_pairing_cases", "boundary_shift_cases"}

    for name, (desc, check_fn) in SLICE_REGISTRY.items():
        matching = []
        for r in rows:
            try:
                if name in _pair_slices:
                    if name == "canonical_pairs":
                        ok = check_fn(r["sample"]["seq"], r["sample"]["pairs"])
                    else:
                        ok = check_fn(r["sample"]["pairs"])
                    if ok: matching.append(r)
                elif name in _pred_slices:
                    if check_fn(r["sample"], r["pred_struct"]):
                        matching.append(r)
                else:
                    if check_fn(r["sample"]):
                        matching.append(r)
            except Exception:
                continue

        if not matching:
            report_lines.append(f"| {name} | 0 | — | — | — | — | — |")
            slice_results[name] = {"N": 0, "f1": None}
            continue

        avg_f1 = sum(m["pair_f1"] for m in matching) / len(matching)
        avg_prec = sum(m["pair_precision"] for m in matching) / len(matching)
        avg_recall = sum(m["pair_recall"] for m in matching) / len(matching)
        avg_pair_ratio = sum(m["pair_count"] / max(1, m["length"]) for m in matching) / len(matching)
        dom = dominant_error({"pair_f1": avg_f1, "pair_precision": avg_prec, "pair_recall": avg_recall})

        report_lines.append(
            f"| {name} | {len(matching)} | {avg_f1:.4f} | {avg_prec:.4f} | {avg_recall:.4f} | {avg_pair_ratio:.3f} | {dom} |")
        slice_results[name] = {
            "N": len(matching), "f1": avg_f1, "precision": avg_prec,
            "recall": avg_recall, "pair_ratio": avg_pair_ratio, "dominant_error": dom,
        }

    report_lines.append("")
    report_lines.append("## Summary")
    report_lines.append("")
    # Overall metrics
    overall_f1 = sum(r["pair_f1"] for r in rows) / len(rows)
    report_lines.append(f"Overall val F1: {overall_f1:.4f}")
    report_lines.append(f"Total samples: {len(rows)}")

    report = "\n".join(report_lines)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(report)

    # Also write JSON
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps({
        "overall_f1": overall_f1, "split": split, "samples": len(rows),
        "slices": slice_results,
    }, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--device", default="cuda")
    p.add_argument("--decode", default="nussinov")
    p.add_argument("--out", default="outputs/reports/slice_eval.md")
    args = p.parse_args()
    run_slice_eval(args.config, args.ckpt, args.split, args.device, args.decode, args.out)


if __name__ == "__main__":
    main()
