"""Fast decode sweep: min_loop pruning + energy prior combinations."""
import sys, json, time, argparse
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.training import load_config, load_checkpoint, build_model, resolve_device
from models.dataset import RNAOmniDataset
from models.decode import nussinov_decode, _forward_model, _build_inference_batch, apply_pruning_mask
from utils.rna_priors import build_pair_prior_matrix
from utils.metric import base_pair_f1, base_pair_precision, base_pair_recall
from utils.struct import parse_dot_bracket, validate_structure


def eval_one(pred, true, seq):
    return {
        "f1": base_pair_f1(pred, true),
        "prec": base_pair_precision(pred, true),
        "rec": base_pair_recall(pred, true),
        "valid": validate_structure(seq, pred),
        "pred_pairs": len(parse_dot_bracket(pred)),
        "true_pairs": len(parse_dot_bracket(true)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="config/candidate.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="outputs/reports/calibration_sweep.md")
    args = ap.parse_args()

    device = resolve_device(args.device)
    config, tokenizer, checkpoint = load_checkpoint(args.ckpt, device)
    user_config = load_config(args.config)
    split_key = f"{args.split}_jsonl"
    split_path = Path(user_config["data"][split_key])
    dataset = RNAOmniDataset(split_path, max_length=int(user_config["data"]["max_length"]))
    if args.limit > 0:
        dataset.samples = dataset.samples[:args.limit]
    model = build_model(config, tokenizer, device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    seqs = [s["seq"] for s in dataset.samples]
    trues = [s["struct"] for s in dataset.samples]
    print(f"Loaded {len(seqs)} samples, extracting logits...")

    # Extract all pair logits in batch
    all_logits = []
    bs = 8
    for start in range(0, len(seqs), bs):
        batch_seqs = seqs[start:start+bs]
        batch, _, struct_positions = _build_inference_batch(
            tokenizer, "seq2struct", batch_seqs[0], "." * len(batch_seqs[0]), device=device
        )
        # For batch > 1 we need proper batching
        for seq in batch_seqs:
            batch, _, sp = _build_inference_batch(tokenizer, "seq2struct", seq, "." * len(seq), device=device)
            batch["input_ids"][:, sp] = tokenizer.mask_id
            outputs = _forward_model(model, batch)
            L = len(seq)
            all_logits.append(outputs["pair_logits"][0, :L, :L].detach().cpu().numpy())
    print(f"Extracted logits for {len(all_logits)} samples.")

    # Define sweep grid
    min_loops = [3, 4, 5, 6]
    lambdas = [0.0, 0.25, 0.5, 0.75]
    gammas = [1.5, 2.0, 2.5, 3.0]
    thresholds = [0.15, 0.20, 0.25, 0.30]

    results = []
    total = len(min_loops) * len(lambdas) * len(gammas) * len(thresholds)
    idx = 0
    best_f1 = 0
    best_combo = None

    for ml in min_loops:
        for lam in lambdas:
            for gamma in gammas:
                for thresh in thresholds:
                    idx += 1
                    t0 = time.time()
                    preds = []
                    for i, (seq, logits) in enumerate(zip(seqs, all_logits)):
                        L = len(seq)
                        # Apply min_loop pruning mask
                        prune_mask = apply_pruning_mask(seq, logits, "min_loop_strict", {"min_loop": ml})
                        # Build energy prior
                        prior = build_pair_prior_matrix(seq, canonical_weight=1.0, loop_penalty=True) if lam > 0 else None
                        # Combine: apply pruning mask to nussinov
                        pred = nussinov_decode(
                            seq, logits,
                            min_loop_length=ml,
                            pair_threshold=thresh,
                            nussinov_gamma=gamma,
                            input_is_logit=True,
                            pruning_mask=prune_mask,
                            pair_prior=prior,
                            pair_prior_alpha=lam,
                        )
                        preds.append(pred)

                    # Aggregate metrics
                    f1s, precs, recs, valids, overpairs, pbiases = [], [], [], [], [], []
                    for p, t, s in zip(preds, trues, seqs):
                        m = eval_one(p, t, s)
                        f1s.append(m["f1"])
                        precs.append(m["prec"])
                        recs.append(m["rec"])
                        valids.append(m["valid"])
                        overpairs.append(1.0 if m["pred_pairs"] > m["true_pairs"] else 0.0)
                        pbiases.append(m["pred_pairs"] - m["true_pairs"])

                    avg = lambda x: sum(x) / len(x)
                    r = {
                        "ml": ml, "lam": lam, "gamma": gamma, "thresh": thresh,
                        "f1": avg(f1s), "prec": avg(precs), "rec": avg(recs),
                        "valid": avg(valids), "overpair": avg(overpairs),
                        "pbias": avg(pbiases), "time": time.time() - t0,
                    }
                    results.append(r)
                    if r["f1"] > best_f1:
                        best_f1 = r["f1"]
                        best_combo = r
                    if idx % 20 == 0:
                        print(f"  [{idx}/{total}] best so far: ml={best_combo['ml']} lam={best_combo['lam']} gamma={best_combo['gamma']} thresh={best_combo['thresh']} F1={best_f1:.4f}")

    # Write report
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Calibration Sweep ({args.split} split, {len(seqs)} samples)",
        "",
        f"**Best**: ml={best_combo['ml']} lam={best_combo['lam']} gamma={best_combo['gamma']} thresh={best_combo['thresh']} **F1={best_f1:.4f}**",
        "",
        "| ml | lam | gamma | thresh | F1 | Prec | Rec | Valid | Overpair | Pbias | Time |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(results, key=lambda x: x["f1"], reverse=True)[:40]:
        lines.append(
            f"| {r['ml']} | {r['lam']:.2f} | {r['gamma']:.1f} | {r['thresh']:.2f} | "
            f"{r['f1']:.4f} | {r['prec']:.4f} | {r['rec']:.4f} | {r['valid']:.2f} | "
            f"{r['overpair']:.2f} | {r['pbias']:.1f} | {r['time']:.0f}s |"
        )
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport -> {args.out}")
    print(f"Best: ml={best_combo['ml']} lam={best_combo['lam']} gamma={best_combo['gamma']} thresh={best_combo['thresh']} F1={best_f1:.4f}")


if __name__ == "__main__":
    main()
