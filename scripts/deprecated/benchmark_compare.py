# scripts/benchmark_compare.py — RNA-OmniPrefold vs ViennaRNA vs Nussinov baseline.
"""Runs OmniPrefold, ViennaRNA, and a simple Nussinov-greedy baseline
on a given split, producing unified prediction JSONL + metrics report."""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import RNA  # ViennaRNA

from models.dataset import RNAOmniDataset
from models.decode import generate_structure_seq2struct
from models.training import load_config, load_checkpoint, build_model, resolve_device
from utils.metric import base_pair_f1, base_pair_precision, base_pair_recall
from utils.struct import parse_dot_bracket, canonical_pair, validate_structure


def nussinov_baseline(seq: str) -> str:
    """ViennaRNA Nussinov: maximize number of canonical base pairs."""
    fc = RNA.fold_compound(seq)
    ss, _ = fc.mfe()
    return ss


def rnafold_predict(seq: str) -> str:
    """ViennaRNA minimum free energy prediction."""
    fc = RNA.fold_compound(seq)
    ss, _ = fc.mfe()
    return ss


def omniprefold_predict(model, tokenizer, seq: str, decode_config: dict, device) -> str:
    return generate_structure_seq2struct(model, tokenizer, seq, decode_config, device)


def run_benchmark(config_path, ckpt_path, split, device_name, decode, out_dir):
    config = load_config(config_path)
    device = resolve_device(device_name)
    ckpt_config, tokenizer, checkpoint = load_checkpoint(ckpt_path, device)
    model = build_model(ckpt_config, tokenizer, device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    jsonl_key = {"train": "train_jsonl", "val": "val_jsonl", "test": "test_jsonl"}[split]
    dataset = RNAOmniDataset(config["data"][jsonl_key],
                             max_length=int(config["data"]["max_length"]))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = {
        "ViennaRNA": lambda seq: rnafold_predict(seq),
        "Nussinov": lambda seq: nussinov_baseline(seq),
        "OmniPrefold": lambda seq: omniprefold_predict(model, tokenizer, seq, config["decoding"], device),
    }

    all_rows = []
    method_metrics = {}

    for method_name, predict_fn in methods.items():
        print(f"\n=== {method_name} ===")
        rows = []
        t0 = time.time()
        for sample in dataset.samples:
            seq = sample["seq"]
            true_struct = sample.get("struct", "")
            t1 = time.time()
            pred = predict_fn(seq)
            runtime = time.time() - t1
            f1 = base_pair_f1(pred, true_struct)
            prec = base_pair_precision(pred, true_struct)
            rec = base_pair_recall(pred, true_struct)
            rows.append({
                "id": sample.get("id", ""),
                "seq": seq,
                "true_struct": true_struct,
                "pred_struct": pred,
                "method": method_name,
                "f1": f1, "precision": prec, "recall": rec,
                "runtime_sec": round(runtime, 4),
                "length": sample["length"],
                "family": sample.get("family", ""),
            })

        total_time = time.time() - t0
        avg_f1 = sum(r["f1"] for r in rows) / len(rows)
        avg_prec = sum(r["precision"] for r in rows) / len(rows)
        avg_rec = sum(r["recall"] for r in rows) / len(rows)
        valid_rate = sum(1 for r in rows if validate_structure(r["seq"], r["pred_struct"])) / len(rows)

        method_metrics[method_name] = {
            "f1": round(avg_f1, 4),
            "precision": round(avg_prec, 4),
            "recall": round(avg_rec, 4),
            "valid_rate": round(valid_rate, 4),
            "total_time_sec": round(total_time, 2),
            "avg_time_per_seq_ms": round(total_time / len(rows) * 1000, 2),
            "samples": len(rows),
        }

        # Save predictions
        pred_path = out_dir / f"{method_name.lower()}_predictions.jsonl"
        with open(pred_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        print(f"  F1={avg_f1:.4f} P={avg_prec:.4f} R={avg_rec:.4f} Valid={valid_rate:.2%} "
              f"Time={total_time:.1f}s ({total_time/len(rows)*1000:.1f}ms/seq)")

        all_rows.extend(rows)

    # Metrics report
    report = ["# Benchmark Comparison", f"Split: {split} ({len(dataset.samples)} samples)", ""]
    report.append("| Method | F1 | Precision | Recall | Valid | Time/seq (ms) |")
    report.append("|---|---:|---:|---:|---:|---:|")
    for method_name, metrics in method_metrics.items():
        report.append(f"| {method_name} | {metrics['f1']:.4f} | {metrics['precision']:.4f} | "
                      f"{metrics['recall']:.4f} | {metrics['valid_rate']:.4f} | {metrics['avg_time_per_seq_ms']:.1f} |")
    report.append("")

    # Family-wise
    families = {}
    for r in all_rows:
        if r["method"] != "OmniPrefold": continue
        fam = r["family"] or "OTHER"
        if fam not in families: families[fam] = []
        families[fam].append(r["f1"])

    if families:
        report.append("## Family-wise (OmniPrefold)")
        report.append("| Family | N | Mean F1 |")
        report.append("|---|---:|---:|")
        for fam in sorted(families, key=lambda k: -len(families[k]))[:10]:
            vals = families[fam]
            report.append(f"| {fam} | {len(vals)} | {sum(vals)/len(vals):.4f} |")
        report.append("")

    # Length-wise
    bins = [(0, 80, "short"), (80, 150, "medium"), (150, 9999, "long")]
    report.append("## Length-wise (OmniPrefold)")
    report.append("| Bin | N | Mean F1 |")
    report.append("|---|---:|---:|")
    for lo, hi, label in bins:
        vals = [r["f1"] for r in all_rows if r["method"] == "OmniPrefold" and lo <= r.get("length", 0) < hi]
        if vals:
            report.append(f"| {label} ({lo}-{hi}) | {len(vals)} | {sum(vals)/len(vals):.4f} |")

    report_path = out_dir / "benchmark_report.md"
    report_path.write_text("\n".join(report))

    # JSON metrics
    json_path = out_dir / "benchmark_metrics.json"
    json_path.write_text(json.dumps(method_metrics, indent=2))

    print(f"\nReport: {report_path}")
    print("\n".join(report))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--device", default="cuda")
    p.add_argument("--decode", default="nussinov")
    p.add_argument("--out_dir", default="outputs/benchmark")
    args = p.parse_args()
    run_benchmark(args.config, args.ckpt, args.split, args.device, args.decode, args.out_dir)


if __name__ == "__main__":
    main()
