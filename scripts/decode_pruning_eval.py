# scripts/decode_pruning_eval.py — Eval-only experimentation for candidate pruning strategies.
"""Loads a checkpoint, runs all 6 pruning strategies on val/test data,
and writes a markdown comparison report to outputs/reports/C7_candidate_pruning.md."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from models.dataset import RNAOmniDataset
from models.decode import (
    _build_inference_batch,
    _forward_model,
    prune_and_decode,
    nussinov_decode,
)
from models.training import load_config, load_checkpoint, build_model, resolve_device
from utils.metric import (
    base_pair_f1,
    base_pair_precision,
    base_pair_recall,
    valid_structure_rate,
    canonical_pair_ratio,
)
from utils.struct import parse_dot_bracket


# ---------------------------------------------------------------------------
# Strategy definitions with parameter sweeps
# ---------------------------------------------------------------------------

STRATEGY_VARIANTS: list[dict[str, Any]] = [
    # --- baseline (no pruning) ---
    {"name": "baseline_none", "strategy": None, "params": {}},
    # --- a. canonical_only ---
    {"name": "canonical_only", "strategy": "canonical_only", "params": {"allow_wobble": True}},
    # --- b. min_loop_strict ---
    {"name": "min_loop=3", "strategy": "min_loop_strict", "params": {"min_loop": 3}},
    {"name": "min_loop=4", "strategy": "min_loop_strict", "params": {"min_loop": 4}},
    {"name": "min_loop=5", "strategy": "min_loop_strict", "params": {"min_loop": 5}},
    {"name": "min_loop=6", "strategy": "min_loop_strict", "params": {"min_loop": 6}},
    # --- c. topk_per_base ---
    {"name": "topk=1", "strategy": "topk_per_base", "params": {"k": 1}},
    {"name": "topk=2", "strategy": "topk_per_base", "params": {"k": 2}},
    {"name": "topk=3", "strategy": "topk_per_base", "params": {"k": 3}},
    {"name": "topk=5", "strategy": "topk_per_base", "params": {"k": 5}},
    # --- d. distance_bucket_top_percentile ---
    {"name": "dist_pct=5", "strategy": "distance_bucket_top_percentile", "params": {"percentile": 5}},
    {"name": "dist_pct=10", "strategy": "distance_bucket_top_percentile", "params": {"percentile": 10}},
    {"name": "dist_pct=20", "strategy": "distance_bucket_top_percentile", "params": {"percentile": 20}},
    {"name": "dist_pct=30", "strategy": "distance_bucket_top_percentile", "params": {"percentile": 30}},
    # --- e. canonical_plus_topk ---
    {"name": "canon+topk=1", "strategy": "canonical_plus_topk", "params": {"k": 1, "allow_wobble": True}},
    {"name": "canon+topk=2", "strategy": "canonical_plus_topk", "params": {"k": 2, "allow_wobble": True}},
    {"name": "canon+topk=3", "strategy": "canonical_plus_topk", "params": {"k": 3, "allow_wobble": True}},
    {"name": "canon+topk=5", "strategy": "canonical_plus_topk", "params": {"k": 5, "allow_wobble": True}},
    # --- f. canonical_plus_distance_prune ---
    {"name": "canon+dist_pct=5", "strategy": "canonical_plus_distance_prune", "params": {"percentile": 5, "allow_wobble": True}},
    {"name": "canon+dist_pct=10", "strategy": "canonical_plus_distance_prune", "params": {"percentile": 10, "allow_wobble": True}},
    {"name": "canon+dist_pct=20", "strategy": "canonical_plus_distance_prune", "params": {"percentile": 20, "allow_wobble": True}},
    {"name": "canon+dist_pct=30", "strategy": "canonical_plus_distance_prune", "params": {"percentile": 30, "allow_wobble": True}},
]


# ---------------------------------------------------------------------------
# Custom metrics
# ---------------------------------------------------------------------------

def rank_accuracy(pair_scores: np.ndarray, true_pairs: list[tuple[int, int]], seq_len: int) -> float:
    """Average normalized rank of true pairs among all row candidates.

    For each true pair (i, j), compute:
        rank = (# of k > i with score[i,k] <= score[i,j]) / (# of k > i)
    Then average over all true pairs. Returns 0..1 where 1 = best.
    """
    if not true_pairs:
        return 0.0
    scores = np.asarray(pair_scores, dtype=np.float32)[:seq_len, :seq_len]
    ranks: list[float] = []
    for i, j in true_pairs:
        if i >= seq_len or j >= seq_len:
            continue
        row_scores = scores[i, i + 1:]
        if len(row_scores) == 0:
            continue
        target_score = scores[i, j]
        # Count how many row entries are <= target_score
        better_or_equal = int(np.sum(row_scores <= target_score))
        ranks.append(better_or_equal / len(row_scores))
    if not ranks:
        return 0.0
    return float(np.mean(ranks))


def overpair_rate(pred_structs: list[str], true_structs: list[str]) -> float:
    """Fraction of samples where predicted pair count > true pair count."""
    if not pred_structs:
        return 0.0
    over = 0
    for pred, true in zip(pred_structs, true_structs):
        try:
            n_pred = len(parse_dot_bracket(pred))
        except ValueError:
            n_pred = 0
        try:
            n_true = len(parse_dot_bracket(true))
        except ValueError:
            n_true = 0
        if n_pred > n_true:
            over += 1
    return over / len(pred_structs)


def pair_count_bias(pred_structs: list[str], true_structs: list[str]) -> float:
    """Average (pred_pair_count - true_pair_count) across samples."""
    if not pred_structs:
        return 0.0
    biases: list[float] = []
    for pred, true in zip(pred_structs, true_structs):
        try:
            n_pred = len(parse_dot_bracket(pred))
        except ValueError:
            n_pred = 0
        try:
            n_true = len(parse_dot_bracket(true))
        except ValueError:
            n_true = 0
        biases.append(float(n_pred - n_true))
    return float(np.mean(biases))


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_pruning_eval(
    config_path: str,
    ckpt_path: str,
    split: str,
    device_name: str,
    max_samples: int | None = None,
) -> None:
    # --- Load ---
    config = load_config(config_path)
    device = resolve_device(device_name)
    ckpt_config, tokenizer, checkpoint = load_checkpoint(ckpt_path, device)
    # Merge data config from user config so test_jsonl is resolved correctly
    merged_config = dict(ckpt_config)
    merged_config["data"] = config.get("data", ckpt_config.get("data", {}))
    model = build_model(merged_config, tokenizer, device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    jsonl_key = {"train": "train_jsonl", "val": "val_jsonl", "test": "test_jsonl"}[split]
    dataset = RNAOmniDataset(
        merged_config["data"][jsonl_key],
        max_length=int(merged_config["data"].get("max_length", 512)),
    )
    samples = dataset.samples
    if max_samples is not None:
        samples = samples[: min(max_samples, len(samples))]
    total = len(samples)
    print(f"Loaded {total} samples from {split} split ({merged_config['data'][jsonl_key]})")

    # --- Decoding defaults (from candidate.yaml) ---
    decode_kwargs = dict(
        min_loop_length=3,
        allow_wobble=True,
        pair_threshold=0.25,
        nussinov_gamma=2.0,
        input_is_logit=True,
    )

    # --- Accumulators per variant ---
    # Each entry: lists of per-sample values
    accum: dict[str, dict[str, list[float]]] = {
        var["name"]: {
            "f1": [],
            "precision": [],
            "recall": [],
            "valid": [],
            "canonical_ratio": [],
            "pred_pair_count": [],
            "true_pair_count": [],
            "rank_acc": [],
            "time_ms": [],
        }
        for var in STRATEGY_VARIANTS
    }
    # Also store per-sample structs for aggregated metrics
    all_pred_structs: dict[str, list[str]] = {var["name"]: [] for var in STRATEGY_VARIANTS}
    all_true_structs: list[str] = []
    all_seqs: list[str] = []

    nussinov_cache: dict[int, np.ndarray] = {}  # sample_idx -> pair_logits (numpy)

    # --- Run ---
    for idx, sample in enumerate(samples):
        seq: str = sample["seq"]
        true_struct: str = sample.get("struct", "")
        true_pairs: list[tuple[int, int]] = sample.get("pairs", [])
        if not true_pairs and true_struct:
            try:
                true_pairs = parse_dot_bracket(true_struct)
            except ValueError:
                true_pairs = []

        all_seqs.append(seq)
        all_true_structs.append(true_struct)

        # --- Model inference (once per sample) ---
        t0 = time.time()
        struct_placeholder = "." * len(seq)
        batch, _, struct_positions = _build_inference_batch(
            tokenizer, "seq2struct", seq, struct_placeholder, device=device,
        )
        batch["input_ids"][:, struct_positions] = tokenizer.mask_id
        outputs = _forward_model(model, batch)
        pair_logits = outputs["pair_logits"][0, :len(seq), :len(seq)]
        pair_logits_np = pair_logits.detach().float().cpu().numpy()
        nussinov_cache[idx] = pair_logits_np
        inference_time = (time.time() - t0) * 1000  # ms

        # --- Evaluate each strategy ---
        for var_idx, var in enumerate(STRATEGY_VARIANTS):
            name = var["name"]
            strategy = var["strategy"]
            params = dict(var["params"])

            t_start = time.time()
            if strategy is None:
                # baseline: standard nussinov_decode (no pruning)
                pred = nussinov_decode(seq, pair_logits, **decode_kwargs)
            else:
                pred = prune_and_decode(
                    seq, pair_logits, strategy, params, **decode_kwargs,
                )
            elapsed = (time.time() - t_start) * 1000

            # Per-sample metrics
            f1 = base_pair_f1(pred, true_struct)
            prec = base_pair_precision(pred, true_struct)
            rec = base_pair_recall(pred, true_struct)
            try:
                n_pred_pairs = len(parse_dot_bracket(pred))
            except ValueError:
                n_pred_pairs = 0
            n_true_pairs = len(true_pairs)
            is_valid = 1.0 if n_pred_pairs >= 0 else 0.0  # will override via validate_structure below
            # canonical ratio
            cpr = canonical_pair_ratio(seq, pred, allow_wobble=True)
            # rank accuracy
            ra = rank_accuracy(pair_logits_np, true_pairs, len(seq))

            accum[name]["f1"].append(f1)
            accum[name]["precision"].append(prec)
            accum[name]["recall"].append(rec)
            accum[name]["valid"].append(1.0 if _is_valid_struct(seq, pred) else 0.0)
            accum[name]["canonical_ratio"].append(cpr)
            accum[name]["pred_pair_count"].append(float(n_pred_pairs))
            accum[name]["true_pair_count"].append(float(n_true_pairs))
            accum[name]["rank_acc"].append(ra)
            accum[name]["time_ms"].append(elapsed)

            all_pred_structs[name].append(pred)

        if (idx + 1) % max(1, total // 10) == 0 or idx == total - 1:
            print(f"  Progress: {idx + 1}/{total} samples processed")

    # --- Aggregate ---
    print("\nAggregating results...")
    report_lines: list[str] = []
    report_lines.append("# C7 — Candidate Pruning Evaluation")
    report_lines.append(f"**Split**: {split}  ")
    report_lines.append(f"**Samples**: {total}  ")
    report_lines.append(f"**Checkpoint**: `{ckpt_path}`  ")
    report_lines.append(f"**Config**: `{config_path}`  ")
    report_lines.append("")
    report_lines.append(
        "| Strategy | F1 | Precision | Recall | Valid Rate | Overpair Rate | "
        "Pair Count Bias | Rank Acc | Canon Ratio | Avg Time (ms) |"
    )
    report_lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )

    summary_rows: list[dict] = []

    for var in STRATEGY_VARIANTS:
        name = var["name"]
        data = accum[name]
        n = len(data["f1"])
        if n == 0:
            continue

        avg_f1 = float(np.mean(data["f1"]))
        avg_prec = float(np.mean(data["precision"]))
        avg_rec = float(np.mean(data["recall"]))
        avg_valid = float(np.mean(data["valid"]))
        avg_canon = float(np.mean(data["canonical_ratio"]))
        avg_rank_acc = float(np.mean(data["rank_acc"]))
        avg_time = float(np.mean(data["time_ms"]))

        # Compute overpair_rate and pair_count_bias
        pred_list = all_pred_structs[name]
        opr = overpair_rate(pred_list, all_true_structs)
        pcb = pair_count_bias(pred_list, all_true_structs)

        report_lines.append(
            f"| {name} | {avg_f1:.4f} | {avg_prec:.4f} | {avg_rec:.4f} | "
            f"{avg_valid:.4f} | {opr:.4f} | {pcb:+.2f} | "
            f"{avg_rank_acc:.4f} | {avg_canon:.4f} | {avg_time:.1f} |"
        )

        summary_rows.append({
            "strategy": name,
            "f1": round(avg_f1, 4),
            "precision": round(avg_prec, 4),
            "recall": round(avg_rec, 4),
            "valid_rate": round(avg_valid, 4),
            "overpair_rate": round(opr, 4),
            "pair_count_bias": round(pcb, 2),
            "rank_accuracy": round(avg_rank_acc, 4),
            "canonical_ratio": round(avg_canon, 4),
            "avg_time_ms": round(avg_time, 1),
            "samples": n,
        })

    report_lines.append("")
    report_lines.append(f"*Report generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*")

    # --- Write report ---
    report_dir = ROOT / "outputs" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "C7_candidate_pruning.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    # Also save JSON for programmatic consumption
    json_path = report_dir / "C7_candidate_pruning.json"
    json_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    print(f"\nReport written to: {report_path}")
    print(f"JSON written to:  {json_path}")
    print("\n" + "\n".join(report_lines))


def _is_valid_struct(seq: str, struct: str, allow_wobble: bool = True) -> bool:
    """Validate a single structure quickly."""
    from utils.struct import validate_structure
    try:
        return validate_structure(seq, struct, allow_wobble)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Eval-only pruning strategy comparison for RNA secondary structure decoding.",
    )
    parser.add_argument("--config", required=True, help="Path to config YAML (e.g., config/candidate.yaml)")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint .pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Which data split to evaluate on (default: test)")
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples (useful for quick smoke tests)")
    args = parser.parse_args()

    run_pruning_eval(
        config_path=args.config,
        ckpt_path=args.ckpt,
        split=args.split,
        device_name=args.device,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
