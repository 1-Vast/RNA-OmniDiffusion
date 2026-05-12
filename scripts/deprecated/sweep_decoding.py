"""Decoding hyperparameter sweep tool for RNA-OmniPrefold.

Loads a checkpoint and sweeps decoding parameters on a validation split
without retraining. Outputs scan results, Pareto frontier, and best config.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.decode import (
    _build_inference_batch,
    _forward_model,
    generate_structure_seq2struct,
    nussinov_decode,
)
from models.dataset import RNAOmniDataset
from models.training import (
    build_model,
    deep_update,
    load_checkpoint,
    load_config,
    resolve_device,
)
from utils.metric import evaluate_structures


def parse_float_list(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _pareto_frontier(results: List[dict], metric_keys: Sequence[str]) -> List[dict]:
    pareto: List[dict] = []
    for r in results:
        dominated = False
        for other in results:
            if r is other:
                continue
            if all(other[k] >= r[k] - 1e-9 for k in metric_keys) and any(
                other[k] > r[k] + 1e-9 for k in metric_keys
            ):
                dominated = True
                break
        if not dominated:
            pareto.append(r)
    return pareto



def main() -> None:
    parser = argparse.ArgumentParser(
        description="RNA-OmniPrefold decoding hyperparameter sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="config/candidate.yaml", help="YAML config")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Dataset split")
    parser.add_argument("--out", default="outputs/sweeps/decoding_candidate", help="Output directory")
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"], help="Device")
    parser.add_argument("--max-samples", type=int, default=512, help="Max samples to evaluate")
    parser.add_argument(
        "--pair-thresholds",
        default="0.15,0.20,0.25,0.30,0.35,0.40",
        help="Comma-separated pair thresholds",
    )
    parser.add_argument(
        "--gammas",
        default="1.0,1.5,2.0,2.5,3.0",
        help="Comma-separated Nussinov gammas",
    )
    parser.add_argument(
        "--pair-prior-alphas",
        default="0.0",
        help="Comma-separated pair prior alphas",
    )
    parser.add_argument(
        "--decode-source",
        default="pair",
        choices=["pair", "hybrid"],
        help="Decoding source",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print plan only, do not run")
    parser.add_argument("--top-k", type=int, default=10, help="Top results to display")
    parser.add_argument("--sort-by", default="pair_f1", help="Metric to sort by")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers (reserved)")
    args = parser.parse_args()

    thresholds = parse_float_list(args.pair_thresholds)
    gammas = parse_float_list(args.gammas)
    alphas = parse_float_list(args.pair_prior_alphas)

    if not thresholds:
        raise SystemExit("--pair-thresholds must contain at least one value.")
    if not gammas:
        raise SystemExit("--gammas must contain at least one value.")
    if not alphas:
        raise SystemExit("--pair-prior-alphas must contain at least one value.")

    combinations = list(itertools.product(thresholds, gammas, alphas))

    # --- Dry run (before any file I/O) --------------------------------------
    if args.dry_run:
        print(f"Plan: {len(combinations)} combinations x up to {args.max_samples} samples")
        print(f"  Config: {args.config}")
        print(f"  Checkpoint: {args.ckpt}")
        print(f"  Decode source: {args.decode_source}")
        print(f"  Split: {args.split}")
        print(f"  Device: {args.device}")
        print(f"  Output: {args.out}")
        print("  Thresholds:", thresholds)
        print("  Gammas:", gammas)
        print("  Pair-prior alphas:", alphas)
        print(f"  Sort by: {args.sort_by}")
        return

    # --- Load user config ---------------------------------------------------
    try:
        user_config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        raise SystemExit(1)

    # --- Validate dataset path ----------------------------------------------
    split_key = f"{args.split}_jsonl"
    data_section = user_config.get("data", {})
    if split_key not in data_section:
        print(f"Error: key '{split_key}' missing from config data section", file=sys.stderr)
        raise SystemExit(1)
    split_path = Path(data_section[split_key])
    if not split_path.exists():
        print(f"Error: dataset file not found: {split_path}", file=sys.stderr)
        raise SystemExit(1)

    # --- Validate checkpoint ------------------------------------------------
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"Error: checkpoint not found: {ckpt_path}", file=sys.stderr)
        raise SystemExit(1)

    # --- Device -------------------------------------------------------------
    device = resolve_device(args.device)

    # --- Load checkpoint ----------------------------------------------------
    try:
        ckpt_config, tokenizer, checkpoint = load_checkpoint(ckpt_path, device)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)

    # --- Merge configs ------------------------------------------------------
    config = dict(ckpt_config)
    config["data"] = deep_update(config.get("data", {}), user_config.get("data", {}))

    # --- Load dataset -------------------------------------------------------
    max_length = int(config["data"]["max_length"])
    dataset = RNAOmniDataset(str(split_path), max_length=max_length)
    samples = dataset.samples[: min(args.max_samples, len(dataset.samples))]
    if not samples:
        print("Error: no samples loaded from dataset", file=sys.stderr)
        raise SystemExit(1)

    # --- Build model --------------------------------------------------------
    model = build_model(config, tokenizer, device)
    try:
        model.load_state_dict(checkpoint["model_state"])
    except RuntimeError as exc:
        print(
            "Error: checkpoint incompatible with model structure. "
            "The pair head may have changed; retrain or use a matching config.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    model.eval()

    # --- Base decoding config (read-only reference) -------------------------
    base_decoding = dict(config.get("decoding", {}))
    base_decoding["decode_source"] = args.decode_source
    base_decoding["use_nussinov"] = True
    base_decoding.setdefault("min_loop_length", 3)
    base_decoding.setdefault("allow_wobble", True)
    base_decoding.setdefault("num_steps", 32)

    min_loop = int(base_decoding["min_loop_length"])
    allow_wobble = bool(base_decoding["allow_wobble"])

    # --- Pre-compute pair logits for pair source ----------------------------
    pair_logits_cache: List[np.ndarray] | None = None
    seqs = [sample["seq"] for sample in samples]
    true_structs = [sample["struct"] for sample in samples]

    if args.decode_source == "pair":
        print("Pre-computing pair logits ...")
        pair_logits_cache = []
        with torch.no_grad():
            for idx, sample in enumerate(samples):
                seq = sample["seq"]
                struct = "." * len(seq)
                batch, _, struct_positions = _build_inference_batch(
                    tokenizer, "seq2struct", seq, struct, device=device,
                )
                batch["input_ids"][:, struct_positions] = tokenizer.mask_id
                outputs = _forward_model(model, batch)
                plogits = outputs["pair_logits"]
                if plogits is None:
                    print(
                        "Error: model has no pair head; cannot run pair decoding. "
                        "Use a checkpoint trained with use_pair_head=true.",
                        file=sys.stderr,
                    )
                    raise SystemExit(1)
                pair_logits_cache.append(
                    plogits[0, : len(seq), : len(seq)].detach().float().cpu().numpy()
                )
                if (idx + 1) % 100 == 0 or idx + 1 == len(samples):
                    print(f"  cached {idx + 1}/{len(samples)} pair logit maps")
        print("Done pre-computing pair logits.")

    # --- Sweep --------------------------------------------------------------
    results: List[dict] = []
    total = len(combinations)
    t_start = time.time()
    for combo_idx, (threshold, gamma, alpha) in enumerate(combinations):
        pred_structs: List[str] = []
        if args.decode_source == "pair" and pair_logits_cache is not None:
            for i, sample in enumerate(samples):
                pred = nussinov_decode(
                    sample["seq"],
                    pair_logits_cache[i],
                    min_loop_length=min_loop,
                    allow_wobble=allow_wobble,
                    pair_threshold=threshold,
                    nussinov_gamma=gamma,
                    input_is_logit=True,
                    pair_prior=None,
                    pair_prior_alpha=alpha,
                )
                pred_structs.append(pred)
        else:
            decoding = dict(base_decoding)
            decoding["pair_threshold"] = threshold
            decoding["nussinov_gamma"] = gamma
            with torch.no_grad():
                for sample in samples:
                    pred = generate_structure_seq2struct(
                        model,
                        tokenizer,
                        sample["seq"],
                        decoding,
                        device,
                        pair_prior=None,
                        pair_prior_alpha=alpha,
                    )
                    pred_structs.append(pred)

        metrics = evaluate_structures(
            pred_structs, true_structs, seqs, allow_wobble=allow_wobble,
        )
        entry: Dict[str, Any] = {
            "pair_threshold": round(threshold, 6),
            "nussinov_gamma": round(gamma, 6),
            "pair_prior_alpha": round(alpha, 6),
            "decode_source": args.decode_source,
            "pair_f1": metrics["pair_f1"],
            "pair_precision": metrics["pair_precision"],
            "pair_recall": metrics["pair_recall"],
            "mcc": metrics["mcc"],
            "valid_structure_rate": metrics["valid_structure_rate"],
            "canonical_pair_ratio": metrics["canonical_pair_ratio"],
            "all_dot_ratio": metrics["all_dot_ratio"],
            "mean_pairs_pred": metrics["avg_pred_pair_count"],
            "mean_pairs_gold": metrics["avg_true_pair_count"],
            "pair_count_gap": metrics["pair_count_gap"],
        }
        results.append(entry)
        elapsed = time.time() - t_start
        per_combo = elapsed / (combo_idx + 1)
        remaining = per_combo * (total - combo_idx - 1)
        print(
            f"[{combo_idx + 1}/{total}] "
            f"t={threshold:.2f} g={gamma:.1f} a={alpha:.1f} "
            f"f1={metrics['pair_f1']:.4f} prec={metrics['pair_precision']:.4f} "
            f"rec={metrics['pair_recall']:.4f} "
            f"elapsed={elapsed:.1f}s rem={remaining:.0f}s"
        )

    # --- Sort ---------------------------------------------------------------
    sort_key = args.sort_by
    if sort_key not in results[0]:
        print(
            f"Warning: sort key '{sort_key}' not in results, falling back to pair_f1",
            file=sys.stderr,
        )
        sort_key = "pair_f1"
    results.sort(key=lambda r: float(r[sort_key]), reverse=True)

    # --- Output directory ---------------------------------------------------
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- scan.json ----------------------------------------------------------
    (out_dir / "scan.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8",
    )

    # --- scan.csv -----------------------------------------------------------
    csv_fields = [
        "pair_threshold",
        "nussinov_gamma",
        "pair_prior_alpha",
        "decode_source",
        "pair_f1",
        "pair_precision",
        "pair_recall",
        "mcc",
        "valid_structure_rate",
        "canonical_pair_ratio",
        "all_dot_ratio",
        "mean_pairs_pred",
        "mean_pairs_gold",
        "pair_count_gap",
    ]
    with (out_dir / "scan.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    # --- pareto.json (Pareto frontier on pair_f1, precision, recall) --------
    pareto_keys = ["pair_f1", "pair_precision", "pair_recall"]
    pareto = _pareto_frontier(results, pareto_keys)
    (out_dir / "pareto.json").write_text(
        json.dumps(pareto, indent=2) + "\n", encoding="utf-8",
    )

    # --- best.json ----------------------------------------------------------
    best = results[0]
    (out_dir / "best.json").write_text(
        json.dumps(best, indent=2) + "\n", encoding="utf-8",
    )

    # --- Console table ------------------------------------------------------
    print()
    print(f"Top {min(args.top_k, len(results))} results (sorted by {sort_key}):")
    header = (
        f"{'threshold':>10} {'gamma':>7} {'alpha':>7} "
        f"{'f1':>8} {'prec':>8} {'rec':>8} {'valid':>8} "
        f"{'all_dot':>8} {'pred_p':>7} {'true_p':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in results[: args.top_k]:
        print(
            f"{r['pair_threshold']:10.2f} {r['nussinov_gamma']:7.1f} "
            f"{r['pair_prior_alpha']:7.1f} "
            f"{r['pair_f1']:8.4f} {r['pair_precision']:8.4f} "
            f"{r['pair_recall']:8.4f} {r['valid_structure_rate']:8.4f} "
            f"{r['all_dot_ratio']:8.4f} {r['mean_pairs_pred']:7.2f} "
            f"{r['mean_pairs_gold']:7.2f}"
        )

    # --- README.md ----------------------------------------------------------
    pareto_table_rows = [
        "| threshold | gamma | alpha | pair_f1 | precision | recall | valid | all_dot |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in pareto:
        pareto_table_rows.append(
            f"| {r['pair_threshold']:.2f} | {r['nussinov_gamma']:.1f} | "
            f"{r['pair_prior_alpha']:.1f} | {r['pair_f1']:.4f} | "
            f"{r['pair_precision']:.4f} | {r['pair_recall']:.4f} | "
            f"{r['valid_structure_rate']:.4f} | {r['all_dot_ratio']:.4f} |"
        )

    readme_lines = [
        "# Decoding Hyperparameter Sweep",
        "",
        f"- Checkpoint: `{args.ckpt}`",
        f"- Config: `{args.config}`",
        f"- Split: `{args.split}` ({split_path})",
        f"- Samples evaluated: {len(samples)}",
        f"- Decode source: `{args.decode_source}`",
        f"- Device: `{device}`",
        f"- Combinations evaluated: {len(combinations)}",
        "",
        "## Sweep Ranges",
        "",
        f"- pair_thresholds: {args.pair_thresholds}",
        f"- nussinov_gammas: {args.gammas}",
        f"- pair_prior_alphas: {args.pair_prior_alphas}",
        "",
        "## Best Configuration",
        "",
        f"- pair_threshold: {best['pair_threshold']:.4f}",
        f"- nussinov_gamma: {best['nussinov_gamma']:.4f}",
        f"- pair_prior_alpha: {best['pair_prior_alpha']:.4f}",
        f"- pair_f1: {best['pair_f1']:.4f}",
        f"- pair_precision: {best['pair_precision']:.4f}",
        f"- pair_recall: {best['pair_recall']:.4f}",
        f"- valid_structure_rate: {best['valid_structure_rate']:.4f}",
        f"- all_dot_ratio: {best['all_dot_ratio']:.4f}",
        f"- mean_pairs_pred: {best['mean_pairs_pred']:.2f}",
        f"- mean_pairs_gold: {best['mean_pairs_gold']:.2f}",
        "",
        "## Pareto Frontier",
        "",
        f"Pareto-optimal configurations (non-dominated in pair_f1, precision, recall): {len(pareto)}",
        "",
        *pareto_table_rows,
        "",
        "## Output Files",
        "",
        "- `scan.json` -- all combinations with full metrics",
        "- `scan.csv` -- table format",
        "- `pareto.json` -- Pareto frontier configurations",
        "- `best.json` -- single best configuration by {sort_key}",
        "",
        "## Reproduction Command",
        "",
        "```bash",
        f"python scripts/sweep_decoding.py \\",
        f"  --config {args.config} \\",
        f"  --ckpt {args.ckpt} \\",
        f"  --split {args.split} \\",
        f'  --out {args.out} \\',
        f"  --device {args.device} \\",
        f"  --max-samples {args.max_samples} \\",
        f'  --pair-thresholds "{args.pair_thresholds}" \\',
        f'  --gammas "{args.gammas}" \\',
        f'  --pair-prior-alphas "{args.pair_prior_alphas}" \\',
        f"  --decode-source {args.decode_source} \\",
        f"  --sort-by {args.sort_by}",
        "```",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    print(f"\nOutput written to: {out_dir}")
    print(f"  {out_dir / 'scan.json'}")
    print(f"  {out_dir / 'scan.csv'}")
    print(f"  {out_dir / 'pareto.json'}")
    print(f"  {out_dir / 'best.json'}")
    print(f"  {out_dir / 'README.md'}")


if __name__ == "__main__":
    main()
