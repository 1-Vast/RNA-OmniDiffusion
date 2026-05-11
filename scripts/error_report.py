# scripts/error_report.py — Comprehensive error taxonomy for Route Q.
"""Aggregate error analysis across pair-distance, structure-pattern,
sequence-property, calibration, and decode dimensions.

Outputs:
  --out_json: machine-readable aggregate statistics
  --out_md:   human-readable markdown report
"""

from __future__ import annotations

import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.dataset import RNAOmniDataset
from models.decode import generate_structure_seq2struct
from models.training import load_config, load_checkpoint, build_model, resolve_device
from utils.metric import base_pair_f1, base_pair_precision, base_pair_recall
from utils.struct import parse_dot_bracket, canonical_pair


def _gc(seq: str) -> float:
    return sum(1 for c in seq if c in "GC") / max(1, len(seq))


def _pair_count(struct: str) -> int:
    return sum(1 for c in struct if c in "(){}<>[]")


def _stem_count(struct: str) -> int:
    n, in_stem = 0, False
    for c in struct:
        if c in "(){}<>[]":
            if not in_stem:
                n += 1; in_stem = True
        else:
            in_stem = False
    return n


def _avg_stem_len(struct: str) -> float:
    stems, cur = [], 0
    for c in struct:
        if c in "(){}<>[]":
            cur += 1
        else:
            if cur > 0: stems.append(cur); cur = 0
    if cur > 0: stems.append(cur)
    return sum(stems) / max(1, len(stems))


def run_error_report(config_path, ckpt_path, split, device_name, decode, out_json, out_md):
    config = load_config(config_path)
    device = resolve_device(device_name)
    ckpt_config, tokenizer, checkpoint = load_checkpoint(ckpt_path, device)
    model = build_model(ckpt_config, tokenizer, device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    jsonl_key = {"train": "train_jsonl", "val": "val_jsonl", "test": "test_jsonl"}[split]
    dataset = RNAOmniDataset(config["data"][jsonl_key],
                             max_length=int(config["data"]["max_length"]))

    # Per-sample detailed results
    results = []
    for sample in dataset.samples:
        seq = sample["seq"]
        true_struct = sample.get("struct", "")
        true_pairs = set(tuple(sorted(p)) for p in sample.get("pairs", []))
        pred_struct = generate_structure_seq2struct(model, tokenizer, seq, config["decoding"], device)
        try:
            pred_pairs = set(tuple(sorted(p)) for p in parse_dot_bracket(pred_struct))
        except ValueError:
            pred_pairs = set()

        tp = true_pairs & pred_pairs
        fp = pred_pairs - true_pairs
        fn = true_pairs - pred_pairs

        # Pair-distance breakdown
        dist_groups = {"short": 0, "mid": 0, "long": 0}
        dist_tp = {"short": 0, "mid": 0, "long": 0}
        dist_fp = {"short": 0, "mid": 0, "long": 0}
        dist_fn = {"short": 0, "mid": 0, "long": 0}
        for (i, j) in true_pairs:
            d = abs(j - i)
            k = "short" if d < 16 else ("mid" if d < 48 else "long")
            dist_groups[k] += 1
            if (i, j) in tp: dist_tp[k] += 1
            else: dist_fn[k] += 1
        for (i, j) in fp:
            d = abs(j - i)
            k = "short" if d < 16 else ("mid" if d < 48 else "long")
            dist_fp[k] += 1

        f1 = base_pair_f1(pred_struct, true_struct)
        prec = base_pair_precision(pred_struct, true_struct)
        rec = base_pair_recall(pred_struct, true_struct)

        # Canonical ratio
        canon_true = sum(1 for (i,j) in true_pairs if canonical_pair(seq[i], seq[j])) / max(1, len(true_pairs))
        canon_pred = sum(1 for (i,j) in pred_pairs if i < len(seq) and j < len(seq) and canonical_pair(seq[i], seq[j])) / max(1, len(pred_pairs))

        results.append({
            "id": sample.get("id", ""),
            "seq": seq,
            "length": sample["length"],
            "f1": f1, "precision": prec, "recall": rec,
            "tp": len(tp), "fp": len(fp), "fn": len(fn),
            "true_pairs": len(true_pairs), "pred_pairs": len(pred_pairs),
            "gc": _gc(seq),
            "true_stems": _stem_count(true_struct),
            "avg_stem_len": _avg_stem_len(true_struct),
            "canon_true": canon_true, "canon_pred": canon_pred,
            "dist_groups": dist_groups, "dist_tp": dist_tp, "dist_fp": dist_fp, "dist_fn": dist_fn,
            "true_struct": true_struct, "pred_struct": pred_struct,
        })

    # ── Aggregate ──
    N = len(results)
    overall_f1 = sum(r["f1"] for r in results) / N
    overall_prec = sum(r["precision"] for r in results) / N
    overall_rec = sum(r["recall"] for r in results) / N
    total_tp = sum(r["tp"] for r in results)
    total_fp = sum(r["fp"] for r in results)
    total_fn = sum(r["fn"] for r in results)
    true_total = sum(r["true_pairs"] for r in results)
    pred_total = sum(r["pred_pairs"] for r in results)

    # A. Pair-distance
    dist_agg = {}
    for k in ["short", "mid", "long"]:
        tp = sum(r["dist_tp"][k] for r in results)
        fp = sum(r["dist_fp"][k] for r in results)
        fn = sum(r["dist_fn"][k] for r in results)
        gs = sum(r["dist_groups"][k] for r in results)
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f = 2 * p * r / max(1e-8, p + r)
        dist_agg[f"{k}_range"] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
                                   "support": gs, "fp": fp, "fn": fn}

    # B. Structure-pattern
    pattern_names = [
        "single_stem", "multi_stem", "sparse_pairs", "dense_pairs",
        "fragmented_stem", "over_pairing", "under_pairing",
    ]
    pattern_agg = {}
    for pn in pattern_names:
        group = []
        for r in results:
            if pn == "single_stem" and r["true_stems"] <= 2: group.append(r)
            elif pn == "multi_stem" and r["true_stems"] >= 4: group.append(r)
            elif pn == "sparse_pairs" and r["true_pairs"] < r["length"] * 0.15: group.append(r)
            elif pn == "dense_pairs" and r["true_pairs"] >= r["length"] * 0.30: group.append(r)
            elif pn == "fragmented_stem" and r["avg_stem_len"] <= 3.0 and r["true_stems"] > 0: group.append(r)
            elif pn == "over_pairing" and r["pred_pairs"] > r["true_pairs"] * 1.5 and r["true_pairs"] > 0: group.append(r)
            elif pn == "under_pairing" and r["pred_pairs"] < r["true_pairs"] * 0.5 and r["true_pairs"] > 0: group.append(r)
        if group:
            pattern_agg[pn] = {
                "count": len(group),
                "mean_f1": round(sum(g["f1"] for g in group) / len(group), 4),
                "mean_precision": round(sum(g["precision"] for g in group) / len(group), 4),
                "mean_recall": round(sum(g["recall"] for g in group) / len(group), 4),
            }
        else:
            pattern_agg[pn] = {"count": 0, "mean_f1": None}

    # C. Sequence-property
    seq_agg = {}
    for label, fn in [("low_gc", lambda r: r["gc"] <= 0.40),
                       ("mid_gc", lambda r: 0.40 < r["gc"] < 0.55),
                       ("high_gc", lambda r: r["gc"] >= 0.55),
                       ("short_seq", lambda r: r["length"] <= 80),
                       ("medium_seq", lambda r: 80 < r["length"] <= 150),
                       ("long_seq", lambda r: r["length"] > 150)]:
        group = [r for r in results if fn(r)]
        if group:
            seq_agg[label] = {
                "count": len(group),
                "mean_f1": round(sum(g["f1"] for g in group) / len(group), 4),
                "mean_precision": round(sum(g["precision"] for g in group) / len(group), 4),
                "mean_recall": round(sum(g["recall"] for g in group) / len(group), 4),
            }
        else:
            seq_agg[label] = {"count": 0, "mean_f1": None}

    # D. Calibration (per-true-pair metrics)
    calib = {
        "predicted_pair_count": pred_total,
        "true_pair_count": true_total,
        "pair_count_bias": round(pred_total - true_total, 1),
        "pair_count_ratio": round(pred_total / max(1, true_total), 3),
        "canonical_true_ratio": round(sum(r["canon_true"] for r in results) / N, 3),
        "canonical_pred_ratio": round(sum(r["canon_pred"] for r in results) / N, 3),
    }

    # E. Decode
    decode_agg = {
        "nussinov_f1": round(overall_f1, 4),
        "total_true_pairs": true_total,
        "total_pred_pairs": pred_total,
        "samples": N,
    }

    report = {
        "overall": {"f1": round(overall_f1, 4), "precision": round(overall_prec, 4),
                      "recall": round(overall_rec, 4), "samples": N,
                      "tp": total_tp, "fp": total_fp, "fn": total_fn},
        "pair_distance": dist_agg,
        "structure_pattern": pattern_agg,
        "sequence_property": seq_agg,
        "calibration": calib,
        "decode": decode_agg,
    }

    # Write JSON
    json_path = Path(out_json); json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2))

    # Write MD
    lines = ["# Error Taxonomy Report", f"split={split} samples={N}", ""]
    lines.append(f"## Overall: F1={overall_f1:.4f} P={overall_prec:.4f} R={overall_rec:.4f}")
    lines.append(f"TP={total_tp} FP={total_fp} FN={total_fn}")
    lines.append("")
    lines.append("## Pair-Distance")
    lines.append("| Range | F1 | Precision | Recall | Support | FP | FN |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for k in ["short_range", "mid_range", "long_range"]:
        d = dist_agg[k]
        lines.append(f"| {k} | {d['f1']:.4f} | {d['precision']:.4f} | {d['recall']:.4f} | {d['support']} | {d['fp']} | {d['fn']} |")
    lines.append("")
    lines.append("## Structure Pattern")
    lines.append("| Pattern | Count | F1 | Precision | Recall |")
    lines.append("|---|---:|---:|---:|---:|")
    for k, v in pattern_agg.items():
        f1 = f"{v['mean_f1']:.4f}" if v['mean_f1'] is not None else "—"
        lines.append(f"| {k} | {v['count']} | {f1} | {v.get('mean_precision','—')} | {v.get('mean_recall','—')} |")
    lines.append("")
    lines.append("## Sequence Property")
    lines.append("| Property | Count | F1 | Precision | Recall |")
    lines.append("|---|---:|---:|---:|---:|")
    for k, v in seq_agg.items():
        f1 = f"{v['mean_f1']:.4f}" if v['mean_f1'] is not None else "—"
        lines.append(f"| {k} | {v['count']} | {f1} | {v.get('mean_precision','—')} | {v.get('mean_recall','—')} |")
    lines.append("")
    lines.append("## Calibration")
    for k, v in calib.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Decode")
    for k, v in decode_agg.items():
        lines.append(f"- {k}: {v}")

    md_path = Path(out_md); md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--decode", default="nussinov")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_json", default="outputs/reports/mainline_error_taxonomy.json")
    p.add_argument("--out_md", default="outputs/reports/mainline_error_taxonomy.md")
    args = p.parse_args()
    run_error_report(args.config, args.ckpt, args.split, args.device, args.decode,
                     args.out_json, args.out_md)


if __name__ == "__main__":
    main()
