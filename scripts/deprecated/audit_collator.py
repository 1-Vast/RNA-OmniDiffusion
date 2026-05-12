from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.training import load_config
from models.dataset import RNAOmniDataset
from models.collator import RNAOmniCollator
from models.token import RNAOmniTokenizer


def collect_batch_stats(
    batch: dict,
    task_counter: Counter,
    seq_masked_list: list,
    struct_masked_list: list,
    pos_pair_list: list,
    neg_pair_list: list,
    length_list: list,
) -> None:
    bsz = len(batch["raw_seq"])
    lengths = batch["lengths"].tolist()
    labels = batch["labels"]
    segment_ids = batch["segment_ids"]
    task_names = batch["task_names"]
    pos_counts = batch["pair_positive_counts"].tolist()
    neg_counts = batch["pair_negative_counts"].tolist()

    for i in range(bsz):
        task_counter[task_names[i]] += 1
        length_list.append(lengths[i])
        pos_pair_list.append(pos_counts[i])
        neg_pair_list.append(neg_counts[i])

        masked_mask = labels[i] != -100
        seq_masked = int((masked_mask & (segment_ids[i] == 1)).sum().item())
        struct_masked = int((masked_mask & (segment_ids[i] == 2)).sum().item())
        seq_masked_list.append(seq_masked)
        struct_masked_list.append(struct_masked)


def make_dataloader(
    samples: list,
    tokenizer: RNAOmniTokenizer,
    config: dict,
    batch_size: int,
) -> DataLoader:
    training_cfg = config["training"]
    collator = RNAOmniCollator(
        tokenizer=tokenizer,
        task_ratios=config["tasks"],
        pair_negative_ratio=int(
            training_cfg.get("pair_negative_ratio", training_cfg.get("pairRatio", 3))
        ),
        seed=int(training_cfg.get("seed", 42)),
        ablation=config.get("ablation", {}),
    )
    return DataLoader(samples, batch_size=batch_size, shuffle=False, collate_fn=collator)


def task_name_sort_key(name: str) -> int:
    order = {"seq2struct": 0, "invfold": 1, "inpaint": 2, "motif_control": 3}
    return order.get(name, 99)


def run_audit(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    split = args.split
    jsonl_key = f"{split}_jsonl" if split != "train" else "train_jsonl"
    if jsonl_key not in config.get("data", {}):
        jsonl_key = "train_jsonl"
    jsonl_path = config["data"][jsonl_key]
    max_len = int(config["data"].get("max_length", 512))
    batch_size = min(int(config["training"].get("batch_size", 8)), args.samples)
    num_samples = args.samples

    dataset = RNAOmniDataset(jsonl_path, max_length=max_len)
    total_available = len(dataset)
    take = min(num_samples, total_available)
    if take < total_available:
        rng = np.random.default_rng(int(config["training"].get("seed", 42)))
        indices = rng.choice(total_available, size=take, replace=False).tolist()
        subset = Subset(dataset, indices)
        samples_selected = [dataset[i] for i in indices]
    else:
        subset = dataset
        samples_selected = list(dataset.samples)

    tokenizer = RNAOmniTokenizer.from_samples(samples_selected)

    loader = make_dataloader(
        subset, tokenizer, config, batch_size=batch_size,
    )

    task_counter: Counter = Counter()
    seq_masked_all: list = []
    struct_masked_all: list = []
    pos_pair_all: list = []
    neg_pair_all: list = []
    lengths_all: list = []

    for batch in loader:
        collect_batch_stats(
            batch,
            task_counter,
            seq_masked_all,
            struct_masked_all,
            pos_pair_all,
            neg_pair_all,
            lengths_all,
        )

    total = sum(task_counter.values())
    family_counter = Counter(
        s.get("family", "OTHER") or "OTHER" for s in samples_selected
    )
    motif_counter = Counter()
    for s in samples_selected:
        motif_counter[len(s.get("motifs", []))] += 1

    seq2struct_share = task_counter.get("seq2struct", 0) / max(1, total)

    sum_pos = sum(pos_pair_all)
    sum_neg = sum(neg_pair_all)
    pair_positive_ratio = sum_pos / max(1, sum_pos + sum_neg)

    length_arr = np.array(lengths_all, dtype=np.float64) if lengths_all else np.array([0.0])
    seq_masked_arr = np.array(seq_masked_all, dtype=np.float64) if seq_masked_all else np.array([0.0])
    struct_masked_arr = np.array(struct_masked_all, dtype=np.float64) if struct_masked_all else np.array([0.0])

    mask_ratio_seq = float(seq_masked_arr.mean() / max(1.0, length_arr.mean()))
    mask_ratio_struct = float(struct_masked_arr.mean() / max(1.0, length_arr.mean()))
    mask_ratio_estimate = (mask_ratio_seq + mask_ratio_struct) / 2.0

    stats = {
        "config": str(args.config),
        "split": split,
        "jsonl_path": str(jsonl_path),
        "total_samples_in_dataset": total_available,
        "sampled": take,
        "batches": total,
        "task_distribution": {
            name: task_counter.get(name, 0) for name in RNAOmniCollator.task_names
        },
        "seq2struct_share": round(seq2struct_share, 4),
        "mask_ratio_estimate": round(mask_ratio_estimate, 4),
        "mask_ratio_seq_mean": round(mask_ratio_seq, 4),
        "mask_ratio_struct_mean": round(mask_ratio_struct, 4),
        "seq_masked_tokens_per_sample": {
            "min": int(seq_masked_arr.min()) if len(seq_masked_arr) else 0,
            "max": int(seq_masked_arr.max()) if len(seq_masked_arr) else 0,
            "mean": round(float(seq_masked_arr.mean()), 2),
            "median": round(float(np.median(seq_masked_arr)), 2),
        },
        "struct_masked_tokens_per_sample": {
            "min": int(struct_masked_arr.min()) if len(struct_masked_arr) else 0,
            "max": int(struct_masked_arr.max()) if len(struct_masked_arr) else 0,
            "mean": round(float(struct_masked_arr.mean()), 2),
            "median": round(float(np.median(struct_masked_arr)), 2),
        },
        "pair_counts": {
            "positive_total": sum_pos,
            "negative_total": sum_neg,
            "positive_ratio": round(pair_positive_ratio, 4),
            "positive_per_sample_mean": round(float(np.mean(pos_pair_all)), 2) if pos_pair_all else 0.0,
            "negative_per_sample_mean": round(float(np.mean(neg_pair_all)), 2) if neg_pair_all else 0.0,
        },
        "seq_length": {
            "min": int(length_arr.min()),
            "max": int(length_arr.max()),
            "mean": round(float(length_arr.mean()), 2),
            "median": round(float(np.median(length_arr)), 2),
        },
        "family_top10": [
            {"family": fam, "count": cnt}
            for fam, cnt in family_counter.most_common(10)
        ],
        "motif_count_distribution": {
            "min": min(motif_counter.keys()) if motif_counter else 0,
            "max": max(motif_counter.keys()) if motif_counter else 0,
            "mean": round(
                sum(k * v for k, v in motif_counter.items()) / max(1, sum(motif_counter.values())), 2
            ),
            "distribution": {
                str(k): v for k, v in sorted(motif_counter.items())
            },
        },
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "collator_audit.json").open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    suggestions = []
    if seq2struct_share < 0.3:
        suggestions.append("Consider increasing tasks.seq2struct in a trial config")
    if pair_positive_ratio < 0.1:
        suggestions.append("Check pair loss weight or data quality")
    if stats["seq_length"]["mean"] > 200:
        suggestions.append("Consider length-based batching")

    md_lines = [
        "# Collator Audit Report",
        "",
        f"Config: `{args.config}`",
        f"Split: `{split}`",
        f"JSONL: `{jsonl_path}`",
        f"Dataset size: {total_available} samples, sampled {take}",
        f"Batches processed: {total}",
        "",
        "## Task Distribution",
        "",
        "| Task | Count | Share |",
        "|------|-------|-------|",
    ]
    for name in sorted(RNAOmniCollator.task_names, key=task_name_sort_key):
        cnt = task_counter.get(name, 0)
        share = cnt / max(1, total)
        md_lines.append(f"| {name} | {cnt} | {share:.3f} |")
    md_lines += [
        "",
        "## Masking Statistics",
        "",
        f"Mask ratio estimate: {mask_ratio_estimate:.4f}",
        f"Mask ratio (seq): {mask_ratio_seq:.4f}",
        f"Mask ratio (struct): {mask_ratio_struct:.4f}",
        "",
        "| Metric | Min | Max | Mean | Median |",
        "|--------|-----|-----|------|--------|",
        f"| Seq masked tokens | {stats['seq_masked_tokens_per_sample']['min']} | {stats['seq_masked_tokens_per_sample']['max']} | {stats['seq_masked_tokens_per_sample']['mean']} | {stats['seq_masked_tokens_per_sample']['median']} |",
        f"| Struct masked tokens | {stats['struct_masked_tokens_per_sample']['min']} | {stats['struct_masked_tokens_per_sample']['max']} | {stats['struct_masked_tokens_per_sample']['mean']} | {stats['struct_masked_tokens_per_sample']['median']} |",
        "",
        "## Pair Statistics",
        "",
        f"Positive pairs total: {sum_pos}",
        f"Negative pairs total: {sum_neg}",
        f"Positive ratio: {pair_positive_ratio:.4f}",
        f"Positive per sample (mean): {stats['pair_counts']['positive_per_sample_mean']}",
        f"Negative per sample (mean): {stats['pair_counts']['negative_per_sample_mean']}",
        "",
        "## Sequence Length",
        "",
        "| Min | Max | Mean | Median |",
        "|-----|-----|------|--------|",
        f"| {stats['seq_length']['min']} | {stats['seq_length']['max']} | {stats['seq_length']['mean']} | {stats['seq_length']['median']} |",
        "",
        "## Family Distribution (Top 10)",
        "",
    ]
    if stats["family_top10"]:
        md_lines.append("| Family | Count |")
        md_lines.append("|--------|-------|")
        for entry in stats["family_top10"]:
            md_lines.append(f"| {entry['family']} | {entry['count']} |")
    else:
        md_lines.append("(no families found)")
    md_lines += [
        "",
        "## Motif Count Distribution",
        "",
        f"Min motifs per sample: {stats['motif_count_distribution']['min']}",
        f"Max motifs per sample: {stats['motif_count_distribution']['max']}",
        f"Mean motifs per sample: {stats['motif_count_distribution']['mean']}",
    ]
    md_lines += [
        "",
        "## Suggestions",
        "",
    ]
    if suggestions:
        for s in suggestions:
            md_lines.append(f"- {s}")
    else:
        md_lines.append("(none)")

    with (out_dir / "collator_audit.md").open("w", encoding="utf-8") as fh:
        fh.write("\n".join(md_lines) + "\n")

    csv_lines = ["task,count"]
    for name in sorted(RNAOmniCollator.task_names, key=task_name_sort_key):
        csv_lines.append(f"{name},{task_counter.get(name, 0)}")
    with (out_dir / "task_distribution.csv").open("w", encoding="utf-8") as fh:
        fh.write("\n".join(csv_lines) + "\n")

    print(f"Audit complete -> {out_dir}")
    print(f"  collator_audit.json")
    print(f"  collator_audit.md")
    print(f"  task_distribution.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collator/masking statistics audit tool for RNA-OmniPrefold."
    )
    parser.add_argument(
        "--config", required=True, type=str,
        help="YAML config path (required)",
    )
    parser.add_argument(
        "--split", default="train", type=str, choices=["train", "val", "test"],
        help="Dataset split (default: train)",
    )
    parser.add_argument(
        "--samples", default=512, type=int,
        help="Number of samples (default: 512)",
    )
    parser.add_argument(
        "--out", default="outputs/audit/collator_candidate", type=str,
        help="Output directory (default: outputs/audit/collator_candidate)",
    )
    parser.add_argument(
        "--device", default="cpu", type=str, choices=["auto", "cpu", "cuda"],
        help="Device (default: cpu)",
    )
    args = parser.parse_args()
    run_audit(args)


if __name__ == "__main__":
    main()
