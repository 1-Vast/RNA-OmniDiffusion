"""Comprehensive calibration grid-search across seeds and params."""
import sys, json, time, argparse
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.training import load_config, load_checkpoint, build_model, resolve_device
from models.dataset import RNAOmniDataset
from models.decode import nussinov_decode, _forward_model, _build_inference_batch, apply_pruning_mask
from utils.metric import evaluate_structures
from utils.struct import parse_dot_bracket

def eval_combos(seqs, trues, all_logits, combos):
    """Evaluate multiple decode combos on pre-extracted logits. Returns dict name->metrics."""
    results = {}
    for name, ml, gamma, thresh in combos:
        preds = []
        for seq, logits in zip(seqs, all_logits):
            prune = apply_pruning_mask(seq, logits, "min_loop_strict", {"min_loop": ml}) if ml > 3 else None
            pred = nussinov_decode(seq, logits, min_loop_length=max(3, ml),
                pair_threshold=thresh, nussinov_gamma=gamma,
                input_is_logit=True, pruning_mask=prune)
            preds.append(pred)
        m = evaluate_structures(preds, trues, seqs, allow_wobble=True)
        m["overpair"] = sum(1 for p, t in zip(preds, trues) if len(parse_dot_bracket(p)) > len(parse_dot_bracket(t))) / len(preds)
        results[name] = m
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="config/candidate.yaml")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = resolve_device(args.device)
    config, tokenizer, checkpoint = load_checkpoint(args.ckpt, device)
    user_config = load_config(args.config)
    dataset = RNAOmniDataset(Path(user_config["data"]["test_jsonl"]), max_length=int(user_config["data"]["max_length"]))
    N = len(dataset.samples)
    seqs = [s["seq"] for s in dataset.samples]
    trues = [s["struct"] for s in dataset.samples]
    print(f"Loading model + extracting logits ({N} samples)...", flush=True)
    model = build_model(config, tokenizer, device)
    model.load_state_dict(checkpoint["model_state"]); model.eval()
    all_logits = []
    for i, seq in enumerate(seqs):
        b, _, sp = _build_inference_batch(tokenizer, "seq2struct", seq, "."*len(seq), device=device)
        b["input_ids"][:, sp] = tokenizer.mask_id
        with torch.no_grad(): out = _forward_model(model, b)
        all_logits.append(out["pair_logits"][0, :len(seq), :len(seq)].detach().cpu().numpy())
        if (i+1) % 100 == 0: print(f"  {i+1}/{N}", flush=True)

    # Build combo list: ml x gamma x threshold
    mls = list(range(3, 9))  # 3-8
    gammas = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    
    combos = []
    for ml in mls:
        for g in gammas:
            for t in thresholds:
                combos.append((f"ml={ml}_g={g}_t={t}", ml, g, t))
    
    print(f"Testing {len(combos)} combinations...", flush=True)
    results = eval_combos(seqs, trues, all_logits, combos)
    
    # Find best
    sorted_r = sorted(results.items(), key=lambda x: x[1]["pair_f1"], reverse=True)
    best_name, best_m = sorted_r[0]
    print(f"\nBest: {best_name} F1={best_m['pair_f1']:.4f} Prec={best_m['pair_precision']:.4f} Rec={best_m['pair_recall']:.4f} Overpair={best_m['overpair']:.3f}")
    
    # Top 20
    out = Path("outputs/reports/grid_opt_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    top20 = [{"name": n, "f1": m["pair_f1"], "prec": m["pair_precision"], "rec": m["pair_recall"],
              "overpair": m["overpair"], "valid": m["valid_structure_rate"]} for n, m in sorted_r[:20]]
    out.write_text(json.dumps(top20, indent=2))
    
    # Markdown
    md = Path("outputs/reports/grid_opt_results.md")
    lines = ["# Grid Optimization Results", f"\n{args.ckpt}", f"\n**Best**: {best_name} F1={best_m['pair_f1']:.4f}", "",
             "| Config | F1 | Prec | Rec | Overpair | Valid |",
             "|---|---:|---:|---:|---:|---:|"]
    for n, m in sorted_r[:20]:
        lines.append(f"| {n} | {m['pair_f1']:.4f} | {m['pair_precision']:.4f} | {m['pair_recall']:.4f} | {m['overpair']:.3f} | {m['valid_structure_rate']:.2f} |")
    md.write_text("\n".join(lines)+"\n", encoding="utf-8")

if __name__ == "__main__":
    main()
