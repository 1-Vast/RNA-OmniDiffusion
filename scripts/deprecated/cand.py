"""Candidate-structure generator for preference optimization.

Generates multiple candidate dot-bracket structures per RNA sequence by:
1. Sweeping decode hyperparameters (Nussinov gamma and pair-logit threshold).
2. (Optional) Adding Gaussian noise to pair logits for diverse candidates.

Usage:
  conda run -n DL python scripts/cand.py --config config/candidate.yaml --ckpt outputs/candidate/best.pt --input dataset/archive/train.jsonl --out outputs/pref/cand_train.jsonl --limit 500 --device cuda --noise_scales 0.5 1.0 2.0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

from models.decode import generate_structure_seq2struct, nussinov_decode, _build_inference_batch
from models.training import build_model, load_checkpoint, resolve_device, load_config as _load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate candidate RNA structures")
    parser.add_argument("--config", required=True, help="YAML training config")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path (.pt)")
    parser.add_argument("--input", required=True, help="JSONL file with RNA sequences")
    parser.add_argument("--out", required=True, help="Output candidate JSONL path")
    parser.add_argument("--limit", type=int, default=0, help="Max samples (0 = all)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--gammas", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0, 8.0])
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.05, 0.15, 0.25, 0.4, 0.6])
    parser.add_argument("--max_candidates", type=int, default=8)
    parser.add_argument("--noise_scales", nargs="+", type=float, default=[],
                        help="Gaussian noise std for perturbed candidates (e.g. 0.5 1.0 2.0)")
    return parser.parse_args()


def load_samples(path: str, limit: int) -> list[dict]:
    samples = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            samples.append(json.loads(line))
            if limit and len(samples) >= limit:
                break
    return samples


def main() -> None:
    args = parse_args()
    if not Path(args.ckpt).exists():
        print(f"Error: checkpoint not found: {args.ckpt}", file=sys.stderr)
        raise SystemExit(1)

    device = resolve_device(args.device)
    config, tokenizer, checkpoint = load_checkpoint(args.ckpt, device)
    model = build_model(config, tokenizer, device)
    try:
        model.load_state_dict(checkpoint["model_state"], strict=False)
    except RuntimeError as exc:
        print(f"Error loading checkpoint: {exc}", file=sys.stderr)
        raise SystemExit(1)
    model.eval()

    user_config = _load_config(args.config)
    decode_cfg = {**config.get("decoding", {}), **user_config.get("decoding", {})}

    samples = load_samples(args.input, args.limit)
    if not samples:
        print("No samples loaded.", file=sys.stderr)
        raise SystemExit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)

    with out_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            seq = sample["seq"]
            candidates: list[dict] = []

            # --- Sweep-based candidates (gamma x threshold) ---
            for gamma in args.gammas:
                for thresh in args.thresholds:
                    if len(candidates) >= args.max_candidates * 2:
                        break
                    local_cfg = dict(decode_cfg)
                    local_cfg["nussinov_gamma"] = gamma
                    local_cfg["pair_threshold"] = thresh
                    with torch.no_grad():
                        try:
                            struct = generate_structure_seq2struct(model, tokenizer, seq, local_cfg, device)
                        except Exception:
                            continue
                    if struct is None:
                        continue
                    from utils.reward import dotbracket_to_pairs, score_struct
                    pairs = dotbracket_to_pairs(struct)
                    candidates.append({
                        "cid": f"s{len(candidates)}",
                        "struct": struct,
                        "pairs": [[int(i), int(j)] for i, j in pairs],
                        "features": score_struct(seq, struct),
                    })
                if len(candidates) >= args.max_candidates * 2:
                    break

            # --- Noise-perturbed candidates ---
            if args.noise_scales and len(candidates) < args.max_candidates * 2:
                with torch.no_grad():
                    try:
                        batch, _, _ = _build_inference_batch(tokenizer, "seq2struct", seq, "." * len(seq), device=device)
                        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                        outputs = model(
                            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                            segment_ids=batch["segment_ids"], task_ids=batch["task_ids"],
                            time_steps=batch["time_steps"], seq_positions=batch["seq_positions"],
                        )
                        pair_logits = outputs.get("pair_logits")
                        if pair_logits is not None:
                            L = len(seq)
                            base = pair_logits[0, :L, :L].cpu().clone()
                            for ns in args.noise_scales:
                                if len(candidates) >= args.max_candidates * 2:
                                    break
                                noisy = base + torch.randn_like(base) * ns
                                try:
                                    nstruct = nussinov_decode(
                                        seq, noisy,
                                        min_loop_length=int(decode_cfg.get("min_loop_length", 3)),
                                        allow_wobble=bool(decode_cfg.get("allow_wobble", True)),
                                        pair_threshold=float(decode_cfg.get("pair_threshold", 0.25)),
                                        nussinov_gamma=float(decode_cfg.get("nussinov_gamma", 2.0)),
                                        input_is_logit=True,
                                    )
                                except Exception:
                                    continue
                                if nstruct is None:
                                    continue
                                npairs = dotbracket_to_pairs(nstruct)
                                candidates.append({
                                    "cid": f"n{len(candidates)}",
                                    "struct": nstruct,
                                    "pairs": [[int(i), int(j)] for i, j in npairs],
                                    "features": score_struct(seq, nstruct),
                                })
                    except Exception:
                        pass

            # Deduplicate and subsample
            seen = set()
            unique = [c for c in candidates if c["struct"] not in seen and not seen.add(c["struct"])]
            if len(unique) > args.max_candidates:
                unique = rng.sample(unique, args.max_candidates)
            for idx, c in enumerate(unique):
                c["cid"] = f"c{idx}"

            entry = {"id": sample.get("id", ""), "seq": seq, "candidates": unique}
            handle.write(json.dumps(entry) + "\n")

    print(f"Candidates written to {out_path}  samples: {len(samples)}")


if __name__ == "__main__":
    main()
