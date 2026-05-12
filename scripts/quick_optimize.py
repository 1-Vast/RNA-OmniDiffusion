"""Quick eval: min_loop pruning + energy prior on full test set."""
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
from utils.metric import evaluate_structures
from utils.struct import parse_dot_bracket


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="config/candidate.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_samples", type=int, default=0)
    args = ap.parse_args()

    print("Loading model...", flush=True)
    device = resolve_device(args.device)
    config, tokenizer, checkpoint = load_checkpoint(args.ckpt, device)
    user_config = load_config(args.config)
    dataset = RNAOmniDataset(
        Path(user_config["data"]["test_jsonl"]),
        max_length=int(user_config["data"]["max_length"]),
    )
    if args.max_samples > 0:
        dataset.samples = dataset.samples[:args.max_samples]
    N = len(dataset.samples)
    seqs = [s["seq"] for s in dataset.samples]
    trues = [s["struct"] for s in dataset.samples]
    print(f"Loading model weights...", flush=True)
    model = build_model(config, tokenizer, device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Extracting pair logits for {N} samples...", flush=True)
    
    # Extract logits (per-sample, single forward each)
    all_logits = []
    for i, seq in enumerate(seqs):
        batch, _, sp = _build_inference_batch(tokenizer, "seq2struct", seq, "." * len(seq), device=device)
        batch["input_ids"][:, sp] = tokenizer.mask_id
        with torch.no_grad():
            outputs = _forward_model(model, batch)
        L = len(seq)
        all_logits.append(outputs["pair_logits"][0, :L, :L].detach().cpu().numpy())
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{N}", flush=True)

    print(f"Running decode sweep...", flush=True)
    
    configs = [
        # (name, min_loop, gamma, threshold, lam, use_prune)
        ("baseline", 3, 2.0, 0.25, 0.0, False),
        ("ml=4", 4, 2.0, 0.25, 0.0, True),
        ("ml=5", 5, 2.0, 0.25, 0.0, True),
        ("ml=6", 6, 2.0, 0.25, 0.0, True),
        ("prior_lam=0.5", 3, 2.0, 0.25, 0.5, False),
        ("ml4+prior_lam=0.5", 4, 2.0, 0.25, 0.5, True),
        ("ml4+prior_lam=0.25", 4, 2.0, 0.25, 0.25, True),
        ("ml4_gamma=3.0", 4, 3.0, 0.25, 0.0, True),
        ("ml4_thresh=0.15", 4, 2.0, 0.15, 0.0, True),
        ("ml4_thresh=0.35", 4, 2.0, 0.35, 0.0, True),
        ("ml4_g3_th15_lam5", 4, 3.0, 0.15, 0.5, True),
    ]
    
    results = []
    for name, ml, gamma, thresh, lam, use_prune in configs:
        t0 = time.time()
        preds = []
        for seq, logits in zip(seqs, all_logits):
            prune_mask = apply_pruning_mask(seq, logits, "min_loop_strict", {"min_loop": ml}) if use_prune else None
            prior = build_pair_prior_matrix(seq, canonical_weight=1.0, loop_penalty=True) if lam > 0 else None
            pred = nussinov_decode(
                seq, logits, min_loop_length=ml if use_prune else 3,
                pair_threshold=thresh, nussinov_gamma=gamma,
                input_is_logit=True, pruning_mask=prune_mask,
                pair_prior=prior, pair_prior_alpha=lam,
            )
            preds.append(pred)
        
        metrics = evaluate_structures(preds, trues, seqs, allow_wobble=True)
        elapsed = time.time() - t0
        
        overpair = sum(1 for p, t in zip(preds, trues) if len(parse_dot_bracket(p)) > len(parse_dot_bracket(t))) / N
        pbias = metrics.get("avg_pred_pair_count", 0) - metrics.get("avg_true_pair_count", 0)
        
        r = {
            "name": name, "f1": metrics["pair_f1"], "prec": metrics["pair_precision"],
            "rec": metrics["pair_recall"], "valid": metrics["valid_structure_rate"],
            "overpair": overpair, "pbias": pbias,
            "canon": metrics.get("canonical_pair_ratio", 0),
            "time": elapsed,
        }
        results.append(r)
        print(f"  {name}: F1={r['f1']:.4f} Prec={r['prec']:.4f} Rec={r['rec']:.4f} Overpair={r['overpair']:.2f} ({elapsed:.1f}s)", flush=True)

    # Write report
    out = Path("outputs/reports/calibration_optimize.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Calibration Optimization ({N} test samples)",
        "",
        "| Config | F1 | Precision | Recall | Valid | Overpair | Pbias | Canon | Time |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(results, key=lambda x: x["f1"], reverse=True):
        lines.append(
            f"| {r['name']} | {r['f1']:.4f} | {r['prec']:.4f} | {r['rec']:.4f} | "
            f"{r['valid']:.2f} | {r['overpair']:.2f} | {r['pbias']:.1f} | "
            f"{r['canon']:.2f} | {r['time']:.0f}s |"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport -> {out}")


if __name__ == "__main__":
    main()
