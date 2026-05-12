# scripts/diagnose_bottleneck.py — Where is the model stuck?
"""Comprehensive bottleneck diagnosis for RNA-OmniPrefold.
Analyzes pair logits, loss dynamics, decode quality, length effects."""

from __future__ import annotations

import json, sys, time
from collections import defaultdict
from pathlib import Path

import torch
import RNA  # ViennaRNA
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.dataset import RNAOmniDataset
from models.decode import generate_structure_seq2struct, nussinov_decode
from models.training import load_config, load_checkpoint, build_model, resolve_device
from utils.metric import base_pair_f1, base_pair_precision, base_pair_recall
from utils.struct import parse_dot_bracket, canonical_pair

def main():
    config = load_config("config/mainline_strongest.yaml")
    device = resolve_device("cuda")
    ckpt_config, tokenizer, checkpoint = load_checkpoint("outputs/mainline_lr0010/best.pt", device)
    model = build_model(ckpt_config, tokenizer, device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = RNAOmniDataset("dataset/archive/test.jsonl", max_length=512)
    samples = dataset.samples
    print(f"Test samples: {len(samples)}")

    # ── 1. Training loss dynamics ──────────────────────────────────────
    log_path = Path("outputs/mainline_lr0010/trainlog.jsonl")
    if log_path.exists():
        entries = [json.loads(l) for l in log_path.read_text().strip().splitlines() if l.strip()]
        losses = [(e["epoch"], e.get("train_pair_loss", 0), e.get("val_pair_loss", 0),
                    e.get("val_pair_f1", 0), e.get("gap", 0),
                    e.get("positive_pair_logit_mean", 0), e.get("negative_pair_logit_mean", 0))
                  for e in entries]
        print(f"\n=== Loss Dynamics ===")
        print(f"{'Epoch':>5} {'TrainPair':>10} {'ValPair':>10} {'ValF1':>7} {'PosLogit':>8} {'NegLogit':>8} {'Margin':>8}")
        for e, tp, vp, f1, gap, pl, nl in losses:
            print(f"{e:>5} {tp:>10.4f} {vp:>10.4f} {f1:>7.4f} {pl:>8.4f} {nl:>8.4f} {gap:>8.4f}")

    # ── 2. Pair logit analysis ─────────────────────────────────────────
    print(f"\n=== Pair Logit Analysis ===")
    from models.training import forward_model, move_batch_to_device
    from models.collator import RNAOmniCollator
    from models.token import RNAOmniTokenizer

    tokenizer2 = RNAOmniTokenizer.from_samples(samples)
    collator = RNAOmniCollator(tokenizer2,
                                {"seq2struct": 0.5, "invfold": 0.25, "inpaint": 0.25},
                                pair_negative_ratio=5)

    all_pos_logits, all_neg_logits = [], []
    all_gt_pos, all_pred_pos = [], []

    with torch.no_grad():
        for i in range(0, len(samples), 8):
            batch_samples = samples[i:i+8]
            batch = collator(batch_samples)
            batch = move_batch_to_device(batch, device)
            outputs = forward_model(model, batch)
            pair_logits = outputs["pair_logits"]
            pair_labels = batch["pair_labels"]
            lengths = batch["lengths"]

            for b in range(pair_logits.shape[0]):
                L = int(lengths[b])
                if L < 2: continue
                logits = pair_logits[b, :L, :L]
                labels = pair_labels[b, :L, :L]
                upper = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
                logits_u = logits[upper]
                labels_u = labels[upper]

                pos = logits_u[labels_u > 0.5]
                neg = logits_u[labels_u <= 0.5]

                if pos.numel():
                    all_pos_logits.extend(pos.cpu().tolist())
                    all_gt_pos.append(len(pos))

                if neg.numel():
                    all_neg_logits.extend(neg.cpu().tolist())

            # Also decode for F1
            for b_idx, sample in enumerate(batch_samples):
                seq = sample["seq"]
                true_struct = sample.get("struct", "")
                pred = generate_structure_seq2struct(model, tokenizer, seq, config["decoding"], device)
                f1 = base_pair_f1(pred, true_struct)
                try:
                    pp = parse_dot_bracket(pred)
                except:
                    pp = []
                all_pred_pos.append(len(pp))

    pos = torch.tensor(all_pos_logits)
    neg = torch.tensor(all_neg_logits)

    print(f"  Positive pairs (all): {pos.numel():,}")
    print(f"  Negative pairs (sampled): {neg.numel():,}")
    print(f"  Pos logit mean: {pos.mean():.4f} ± {pos.std():.4f}")
    print(f"  Neg logit mean: {neg.mean():.4f} ± {neg.std():.4f}")
    print(f"  Pos prob mean: {pos.sigmoid().mean():.4f}")
    print(f"  Neg prob mean: {neg.sigmoid().mean():.4f}")
    print(f"  Margin (pos - neg): {pos.mean() - neg.mean():.4f}")
    # Use random subset for rank accuracy
    sample_n = min(1000, pos.numel(), neg.numel())
    pos_sample = pos[torch.randperm(pos.numel())[:sample_n]]
    neg_sample = neg[torch.randperm(neg.numel())[:sample_n]]
    rank_acc = (pos_sample.unsqueeze(1) > neg_sample.unsqueeze(0)).float().mean()
    print(f"  Pos logit > Neg logit (rank acc, {sample_n}x{sample_n}): {rank_acc:.4f}")

    # Percentile analysis
    print(f"  Pos logit percentiles: P10={pos.float().quantile(0.1):.3f} P50={pos.median():.3f} P90={pos.float().quantile(0.9):.3f}")
    print(f"  Neg logit percentiles: P10={neg.float().quantile(0.1):.3f} P50={neg.median():.3f} P90={neg.float().quantile(0.9):.3f}")

    # Threshold crosser analysis
    pos_prob = pos.sigmoid()
    neg_prob = neg.sigmoid()
    for t in [0.25, 0.5, 0.75]:
        neg_above = (neg_prob > t).float().mean()
        pos_above = (pos_prob > t).float().mean()
        print(f"  at t={t:.2f}: neg above={neg_above:.4f} ({neg_above*100:.1f}%) pos above={pos_above:.4f} ({pos_above*100:.1f}%)")

    # ── 3. Nussinov threshold sweep ─────────────────────────────────────
    print(f"\n=== Nussinov Threshold Sweep ===")
    best_f1, best_thresh = 0, 0.25
    for thresh in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
        decode_cfg = dict(config["decoding"])
        decode_cfg["pair_threshold"] = thresh
        f1s = []
        for s in samples[:50]:
            pred = generate_structure_seq2struct(model, tokenizer, s["seq"], decode_cfg, device)
            f1s.append(base_pair_f1(pred, s.get("struct", "")))
        avg = sum(f1s) / len(f1s)
        if avg > best_f1: best_f1, best_thresh = avg, thresh
        print(f"  threshold={thresh:.2f}: F1={avg:.4f}")
    print(f"  Best: threshold={best_thresh:.2f} F1={best_f1:.4f}")

    # ── 4. Logit → Nussinov quality ────────────────────────────────────
    print(f"\n=== Nussinov Output vs Pair Logits ===")
    for s in samples[:5]:
        seq = s["seq"]
        true = s.get("struct", "")
        pred = generate_structure_seq2struct(model, tokenizer, seq, config["decoding"], device)
        rnafold_pred = RNA.fold_compound(seq).mfe()[0]
        f1_omi = base_pair_f1(pred, true)
        f1_rna = base_pair_f1(rnafold_pred, true)
        print(f"  {s.get('id','?')[:30]} L={len(seq)}: Omi={f1_omi:.2f} RNAfold={f1_rna:.2f} | True={true[:40]} Pred={pred[:40]} RNAfold={rnafold_pred[:40]}")

    # ── 5. Length-wise bottleneck ──────────────────────────────────────
    print(f"\n=== Length-Wise Breakdown ===")
    bins = [(0, 60, "0-60"), (60, 80, "60-80"), (80, 120, "80-120"),
            (120, 200, "120-200"), (200, 350, "200-350"), (350, 999, "350+")]
    for lo, hi, label in bins:
        group = [s for s in samples if lo <= s["length"] < hi]
        if not group: continue
        f1s_omi, f1s_rna = [], []
        for s in group:
            pred = generate_structure_seq2struct(model, tokenizer, s["seq"], config["decoding"], device)
            rnadb = RNA.fold_compound(s["seq"]).mfe()[0]
            f1s_omi.append(base_pair_f1(pred, s.get("struct", "")))
            f1s_rna.append(base_pair_f1(rnadb, s.get("struct", "")))
        print(f"  {label:>8} (N={len(group):>3}): OmniF1={np.mean(f1s_omi):.4f} RNAfoldF1={np.mean(f1s_rna):.4f} Δ={np.mean(f1s_rna)-np.mean(f1s_omi):.3f}")

    # ── 6. Pair count analysis ─────────────────────────────────────────
    print(f"\n=== Pair Count Analysis ===")
    to_under, to_over, to_correct = 0, 0, 0
    for s in samples[:100]:
        true_pairs = len(s.get("pairs", []))
        pred = generate_structure_seq2struct(model, tokenizer, s["seq"], config["decoding"], device)
        try:
            pred_pairs = len(parse_dot_bracket(pred))
        except:
            pred_pairs = 0
        if pred_pairs < true_pairs * 0.8: to_under += 1
        elif pred_pairs > true_pairs * 1.2: to_over += 1
        else: to_correct += 1
    print(f"  Under-pairing (<80% of true): {to_under}%")
    print(f"  Over-pairing (>120% of true): {to_over}%")
    print(f"  Within range (80-120%): {to_correct}%")

    # ── 7. Training data size check ─────────────────────────────────────
    train_data = RNAOmniDataset("dataset/archive/train.jsonl", max_length=512)
    train_total_pairs = sum(len(s.get("pairs", [])) for s in train_data.samples)
    avg_pairs = train_total_pairs / len(train_data.samples)
    print(f"\n=== Training Data ===")
    print(f"  Train samples: {len(train_data.samples)}")
    print(f"  Total true pairs: {train_total_pairs:,}")
    print(f"  Avg pairs/sample: {avg_pairs:.1f}")
    print(f"  Avg length: {np.mean([s['length'] for s in train_data.samples]):.0f}")
    print(f"  Max length: {max(s['length'] for s in train_data.samples)}")

if __name__ == "__main__":
    main()
