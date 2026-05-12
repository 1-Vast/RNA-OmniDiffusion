"""Evaluate energy-inspired pair priors on the validation set.

Usage::

    conda run -n DL python scripts/prior_eval.py \
        --config config/candidate.yaml \
        --ckpt outputs/candidate/best.pt \
        --device cuda

The script tests a range of *lambda_prior* values (0.1, 0.25, 0.5, 1.0),
computes per-sample metrics, and writes a Markdown report to
``outputs/reports/C8_energy_inspired_prior.md``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import build_model, load_checkpoint, load_config, resolve_device
from models.dataset import RNAOmniDataset
from models.decode import (
    _build_inference_batch,
    _forward_model,
    generate_structure_seq2struct,
)
from utils.metric import (
    base_pair_f1,
    base_pair_precision,
    base_pair_recall,
    evaluate_structures,
)
from utils.rna_priors import build_pair_prior_matrix
from utils.struct import parse_dot_bracket


# ---------------------------------------------------------------------------
# logit-level diagnostics
# ---------------------------------------------------------------------------


def _compute_logit_metrics(
    pair_logits: np.ndarray,
    true_pairs: list[tuple[int, int]],
    L: int,
) -> dict[str, float]:
    """Per-sample metrics derived from raw pair logits."""

    logits: np.ndarray = pair_logits.astype(np.float32)[:L, :L]
    true_set: set[tuple[int, int]] = {(min(i, j), max(i, j)) for i, j in true_pairs}

    # Collect upper-triangle values
    neg_count = 0
    neg_gt_025 = 0
    neg_gt_05 = 0
    true_values: list[float] = []
    all_upper: list[tuple[float, int, int]] = []  # (logit, i, j)

    for i in range(L):
        for j in range(i + 1, L):
            val = float(logits[i, j])
            all_upper.append((val, i, j))
            if (i, j) in true_set:
                true_values.append(val)
            else:
                neg_count += 1
                prob = float(1.0 / (1.0 + np.exp(-val)))
                if prob > 0.25:
                    neg_gt_025 += 1
                if prob > 0.5:
                    neg_gt_05 += 1

    neg_gt_025_ratio = neg_gt_025 / max(1, neg_count)
    neg_gt_05_ratio = neg_gt_05 / max(1, neg_count)

    # Rank accuracy: what fraction of true pairs appear in the top-K
    # where K = number of true pairs
    all_upper.sort(key=lambda t: t[0], reverse=True)
    k = max(1, len(true_set))
    top_k_positions = {(i, j) for _, i, j in all_upper[:k]}
    rank_acc = len(true_set & top_k_positions) / max(1, len(true_set))

    return {
        "neg_gt_025": float(neg_gt_025_ratio),
        "neg_gt_05": float(neg_gt_05_ratio),
        "rank_accuracy": float(rank_acc),
    }


# ---------------------------------------------------------------------------
# batch logit extraction
# ---------------------------------------------------------------------------


@torch.no_grad()
def _extract_batch_logits(
    model: torch.nn.Module,
    tokenizer,
    samples: list[dict],
    device: torch.device,
) -> torch.Tensor:
    """Forward a batch of samples and return pair logits as (B, L, L)."""

    max_len = max(int(s["length"]) for s in samples)
    batch_ids: list[list[int]] = []
    segment_rows: list[list[int]] = []
    seq_positions_rows: list[list[int]] = []
    struct_positions_rows: list[list[int]] = []

    for sample in samples:
        tokens: list[str] = []
        segments: list[int] = []
        seq_pos: list[int] = []
        struct_pos: list[int] = []

        tokens.append(tokenizer.task_token("seq2struct"))
        segments.append(0)

        tokens.append("<SEQ>")
        segments.append(1)
        for base in sample["seq"]:
            seq_pos.append(len(tokens))
            tokens.append(base)
            segments.append(1)
        tokens.append("</SEQ>")
        segments.append(1)

        tokens.append("<STRUCT>")
        segments.append(2)
        for _ in sample["seq"]:
            struct_pos.append(len(tokens))
            tokens.append("<MASK>")
            segments.append(2)
        tokens.append("</STRUCT>")
        segments.append(2)

        batch_ids.append(tokenizer.encode(tokens))
        segment_rows.append(segments)
        seq_positions_rows.append(seq_pos)
        struct_positions_rows.append(struct_pos)

    B = len(samples)
    T = max(len(ids) for ids in batch_ids)
    input_ids = torch.full((B, T), tokenizer.pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((B, T), dtype=torch.long, device=device)
    segment_ids = torch.zeros((B, T), dtype=torch.long, device=device)
    seq_pos_t = torch.full((B, max_len), -1, dtype=torch.long, device=device)

    for idx, ids in enumerate(batch_ids):
        input_ids[idx, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[idx, : len(ids)] = 1
        segment_ids[idx, : len(ids)] = torch.tensor(
            segment_rows[idx], dtype=torch.long, device=device
        )
        seq_pos_t[idx, : len(seq_positions_rows[idx])] = torch.tensor(
            seq_positions_rows[idx], dtype=torch.long, device=device
        )

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "segment_ids": segment_ids,
        "task_ids": torch.full(
            (B,), tokenizer.task_to_id["seq2struct"], dtype=torch.long, device=device
        ),
        "time_steps": torch.ones(B, dtype=torch.float32, device=device),
        "seq_positions": seq_pos_t,
    }

    outputs = _forward_model(model, batch)
    pair = outputs.get("pair_logits")
    if pair is None:
        raise SystemExit("Model did not return pair_logits.")
    return pair[:, :max_len, :max_len]


# ---------------------------------------------------------------------------
# main evaluation
# ---------------------------------------------------------------------------


def run_prior_eval(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    config, tokenizer, checkpoint = load_checkpoint(args.ckpt, device)
    user_config = load_config(args.config)
    split_key = f"{args.split}_jsonl"
    if split_key not in user_config.get("data", {}):
        raise SystemExit(f"Split '{args.split}' not found in config data section (looked for '{split_key}').")
    split_path = Path(user_config["data"][split_key])
    if not split_path.exists():
        raise SystemExit(f"Split file not found: {split_path}")

    dataset = RNAOmniDataset(
        split_path, max_length=int(user_config["data"]["max_length"])
    )
    if args.limit:
        dataset.samples = dataset.samples[: int(args.limit)]

    model = build_model(config, tokenizer, device)
    try:
        model.load_state_dict(checkpoint["model_state"])
    except RuntimeError as exc:
        raise SystemExit(
            "Checkpoint incompatible with current model structure."
        ) from exc
    model.eval()

    decoding_config = config.get("decoding", {})
    batch_size = int(args.batch or config.get("training", {}).get("batch_size", 8))

    lambda_values = [0.1, 0.25, 0.5, 1.0]
    report_rows: list[dict] = []

    # Pre-extract all pair logits for logit-level metrics
    print(f"Extracting pair logits for {len(dataset.samples)} samples ...")
    all_logits_np: list[np.ndarray] = []
    for start in range(0, len(dataset.samples), batch_size):
        batch_samples = dataset.samples[start : start + batch_size]
        pair_batch = _extract_batch_logits(model, tokenizer, batch_samples, device)
        for i in range(len(batch_samples)):
            L = int(batch_samples[i]["length"])
            all_logits_np.append(
                pair_batch[i, :L, :L].detach().float().cpu().numpy()
            )

    for lam in lambda_values:
        print(f"\n--- lambda_prior = {lam} ---")
        t0 = time.time()

        all_preds: list[str] = []
        all_trues: list[str] = []
        all_seqs: list[str] = []
        logit_metrics_agg: dict[str, list[float]] = {
            "neg_gt_025": [],
            "neg_gt_05": [],
            "rank_accuracy": [],
        }

        for idx, sample in enumerate(dataset.samples):
            seq = sample["seq"]
            true_struct = sample["struct"]
            L = len(seq)

            true_pairs = parse_dot_bracket(true_struct)

            # --- logit-level metrics (pre-extracted) ---
            lm = _compute_logit_metrics(all_logits_np[idx], true_pairs, L)
            for key in logit_metrics_agg:
                logit_metrics_agg[key].append(lm[key])

            # --- prediction with prior ---
            pair_prior = build_pair_prior_matrix(seq, canonical_weight=1.0)

            try:
                pred_struct = generate_structure_seq2struct(
                    model,
                    tokenizer,
                    seq,
                    decoding_config,
                    device,
                    pair_prior=pair_prior,
                    pair_prior_alpha=lam,
                )
            except Exception:
                pred_struct = "." * L

            all_preds.append(pred_struct)
            all_trues.append(true_struct)
            all_seqs.append(seq)

        # --- aggregate ---
        overall = evaluate_structures(all_preds, all_trues, all_seqs)

        # overpair rate: ratio where pred_pairs > true_pairs
        overpair = 0
        for pred, true in zip(all_preds, all_trues):
            pred_n = len(parse_dot_bracket(pred))
            true_n = len(parse_dot_bracket(true))
            if pred_n > true_n:
                overpair += 1
        overpair_rate = overpair / max(1, len(all_preds))

        elapsed = time.time() - t0

        row = {
            "lambda_prior": lam,
            "f1": overall["pair_f1"],
            "precision": overall["pair_precision"],
            "recall": overall["pair_recall"],
            "valid_rate": overall["valid_structure_rate"],
            "overpair_rate": overpair_rate,
            "neg_gt_025": float(np.mean(logit_metrics_agg["neg_gt_025"])),
            "neg_gt_05": float(np.mean(logit_metrics_agg["neg_gt_05"])),
            "rank_accuracy": float(np.mean(logit_metrics_agg["rank_accuracy"])),
            "all_dot_ratio": overall["all_dot_ratio"],
            "avg_pred_pairs": overall["avg_pred_pair_count"],
            "avg_true_pairs": overall["avg_true_pair_count"],
            "seconds": elapsed,
        }
        report_rows.append(row)

        print(
            f"  F1={row['f1']:.4f}  Prec={row['precision']:.4f}  "
            f"Rec={row['recall']:.4f}  Valid={row['valid_rate']:.4f}  "
            f"RankAcc={row['rank_accuracy']:.4f}  Time={elapsed:.1f}s"
        )

    # --- write markdown report ---
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# C8 — Energy-Inspired Pair Prior Evaluation",
        "",
        f"**Checkpoint:** `{args.ckpt}`",
        f"**Split:** `{args.split}`",
        f"**Samples:** {len(dataset.samples)}",
        f"**Decoding:** `{decoding_config.get('decode_source', 'pair')}` "
        f"(thresh={decoding_config.get('pair_threshold', 0.5)}, "
        f"gamma={decoding_config.get('nussinov_gamma', 1.0)})",
        "",
        "## Results",
        "",
        "| λ_prior | F1 ↑ | Prec ↑ | Rec ↑ | Valid ↑ | Overpair ↓ | "
        "neg>.25 ↓ | neg>.5 ↓ | RankAcc ↑ | All-dot ↓ | ΔPairs | Time |",
        "|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in report_rows:
        dp = r["avg_pred_pairs"] - r["avg_true_pairs"]
        lines.append(
            f"| {r['lambda_prior']:.2f} "
            f"| {r['f1']:.4f} "
            f"| {r['precision']:.4f} "
            f"| {r['recall']:.4f} "
            f"| {r['valid_rate']:.4f} "
            f"| {r['overpair_rate']:.4f} "
            f"| {r['neg_gt_025']:.4f} "
            f"| {r['neg_gt_05']:.4f} "
            f"| {r['rank_accuracy']:.4f} "
            f"| {r['all_dot_ratio']:.4f} "
            f"| {dp:+.1f} "
            f"| {r['seconds']:.0f}s |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- **neg>.25 / neg>.5**: fraction of non-true-pair positions "
            "where sigmoid(logit) exceeds the threshold.",
            "- **RankAcc**: fraction of true pairs appearing in the top-*k* "
            "logit positions (*k* = number of true pairs).",
            "- **Overpair**: fraction of samples where predicted pairs > true pairs.",
            "- Priors are constructed via `utils/rna_priors.py` "
            "(canonical type + loop closure).",
            "- The prior is added to the Nussinov score matrix with weight "
            "`pair_prior_alpha` = λ_prior.",
        ]
    )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate energy-inspired pair priors on validation set.",
    )
    parser.add_argument(
        "--config", default="config/candidate.yaml", help="YAML config path"
    )
    parser.add_argument("--ckpt", required=True, help="Model checkpoint path")
    parser.add_argument(
        "--split", default="val", choices=["train", "val", "test"]
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"]
    )
    parser.add_argument("--limit", type=int, help="Limit number of samples")
    parser.add_argument("--batch", type=int, help="Batch size for logit extraction")
    parser.add_argument(
        "--out",
        default="outputs/reports/C8_energy_inspired_prior.md",
        help="Output report path",
    )
    args = parser.parse_args()
    run_prior_eval(args)


if __name__ == "__main__":
    main()
