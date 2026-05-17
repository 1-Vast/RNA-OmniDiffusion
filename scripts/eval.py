from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.dataset import RNAOmniDataset
from main import build_model, load_checkpoint, load_config, resolve_device
from models.decode import (
    GREEDY_DECODE_WARNING,
    batched_greedy_decode_gpu,
    generate_structure_seq2struct,
    nussinov_decode,
    pairs_matrix_to_dotbracket_batch_with_stats,
)
from utils.metric import base_pair_f1, base_pair_precision, base_pair_recall, canonical_pair_ratio, evaluate_structures
from utils.struct import canonical_pair, parse_dot_bracket, pairs_to_dot_bracket, validate_structure


def random_valid(seq: str, loop: int = 3) -> str:
    rng = random.Random(42 + len(seq))
    candidates = [(i, j) for i in range(len(seq)) for j in range(i + loop, len(seq)) if canonical_pair(seq[i], seq[j])]
    rng.shuffle(candidates)
    pairs = []
    used = set()
    for i, j in candidates:
        if i in used or j in used:
            continue
        if any(i < a < j < b or a < i < b < j for a, b in pairs):
            continue
        pairs.append((i, j))
        used.update([i, j])
        if len(pairs) >= max(1, len(seq) // 12):
            break
    return pairs_to_dot_bracket(pairs, len(seq))


def row(method: str, sample: dict, pred: str, sample_index: int | None = None) -> dict:
    try:
        pred_pairs = parse_dot_bracket(pred)
    except ValueError:
        pred_pairs = []
    return {
        "method": method,
        "sample_index": sample_index,
        "id": sample.get("id", ""),
        "seq": sample["seq"],
        "true_struct": sample["struct"],
        "pred_struct": pred,
        "true_pairs": sample.get("pairs", []),
        "pred_pairs": pred_pairs,
        "pair_precision": base_pair_precision(pred, sample["struct"]),
        "pair_recall": base_pair_recall(pred, sample["struct"]),
        "pair_f1": base_pair_f1(pred, sample["struct"]),
        "valid": validate_structure(sample["seq"], pred),
        "canonical_pair_ratio": canonical_pair_ratio(sample["seq"], pred),
        "family": sample.get("family", "OTHER"),
        "length": sample["length"],
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    return evaluate_structures(
        [item["pred_struct"] for item in rows],
        [item["true_struct"] for item in rows],
        [item["seq"] for item in rows],
    )


def build_seq2struct_batch(tokenizer, samples: list[dict], device: torch.device) -> dict:
    token_rows, segment_rows, seq_positions, struct_positions = [], [], [], []
    max_tokens = 0
    max_len = max(int(sample["length"]) for sample in samples)
    for sample in samples:
        tokens = [tokenizer.task_token("seq2struct"), "<SEQ>"]
        segments = [0, 1]
        seq_pos = []
        for base in sample["seq"]:
            seq_pos.append(len(tokens))
            tokens.append(base)
            segments.append(1)
        tokens += ["</SEQ>", "<STRUCT>"]
        segments += [1, 2]
        struct_pos = []
        for _ in sample["seq"]:
            struct_pos.append(len(tokens))
            tokens.append("<MASK>")
            segments.append(2)
        tokens.append("</STRUCT>")
        segments.append(2)
        token_rows.append(tokenizer.encode(tokens))
        segment_rows.append(segments)
        seq_positions.append(seq_pos)
        struct_positions.append(struct_pos)
        max_tokens = max(max_tokens, len(tokens))

    batch_size = len(samples)
    input_ids = torch.full((batch_size, max_tokens), tokenizer.pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_tokens), dtype=torch.long, device=device)
    segment_ids = torch.zeros((batch_size, max_tokens), dtype=torch.long, device=device)
    seq_pos_tensor = torch.full((batch_size, max_len), -1, dtype=torch.long, device=device)
    struct_pos_tensor = torch.full((batch_size, max_len), -1, dtype=torch.long, device=device)
    for idx, ids in enumerate(token_rows):
        input_ids[idx, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[idx, : len(ids)] = 1
        segment_ids[idx, : len(ids)] = torch.tensor(segment_rows[idx], dtype=torch.long, device=device)
        seq_pos_tensor[idx, : len(seq_positions[idx])] = torch.tensor(seq_positions[idx], dtype=torch.long, device=device)
        struct_pos_tensor[idx, : len(struct_positions[idx])] = torch.tensor(struct_positions[idx], dtype=torch.long, device=device)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "segment_ids": segment_ids,
        "task_ids": torch.full((batch_size,), tokenizer.task_to_id["seq2struct"], dtype=torch.long, device=device),
        "time_steps": torch.ones(batch_size, dtype=torch.float32, device=device),
        "seq_positions": seq_pos_tensor,
        "struct_positions": struct_pos_tensor,
    }


def forward_pair_logits(model, tokenizer, samples: list[dict], device: torch.device) -> torch.Tensor:
    batch = build_seq2struct_batch(tokenizer, samples, device)
    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            segment_ids=batch["segment_ids"],
            task_ids=batch["task_ids"],
            time_steps=batch["time_steps"],
            seq_positions=batch["seq_positions"],
        )
    pair_logits = outputs.get("pair_logits")
    if pair_logits is None:
        raise SystemExit("Model did not return pair_logits; greedy decode requires use_pair_head=true.")
    max_len = max(int(sample["length"]) for sample in samples)
    if pair_logits.ndim != 3 or pair_logits.size(1) < max_len or pair_logits.size(2) < max_len:
        raise SystemExit(f"Unexpected pair_logits shape {tuple(pair_logits.shape)} for max RNA length {max_len}.")
    return pair_logits[:, :max_len, :max_len]


def forward_token_structures(model, tokenizer, samples: list[dict], device: torch.device) -> tuple[list[str], float]:
    batch = build_seq2struct_batch(tokenizer, samples, device)
    allowed_ids = torch.tensor([tokenizer.token_id(token) for token in tokenizer.structure_tokens], dtype=torch.long, device=device)
    start = time.time()
    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            segment_ids=batch["segment_ids"],
            task_ids=batch["task_ids"],
            time_steps=batch["time_steps"],
            seq_positions=batch["seq_positions"],
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    forward_seconds = time.time() - start
    structs: list[str] = []
    struct_logits = outputs["structure_logits"]
    for idx, sample in enumerate(samples):
        length = int(sample["length"])
        positions = batch["struct_positions"][idx, :length]
        restricted = struct_logits[idx, positions].index_select(-1, allowed_ids)
        pred_ids = allowed_ids[restricted.argmax(dim=-1)].detach().cpu().tolist()
        structs.append("".join(tokenizer.decode(pred_ids)))
    return structs, forward_seconds


def token_invalid_reasons(seq: str, pred: str) -> list[str]:
    legal = set(".()[]{}")
    reasons: list[str] = []
    if len(pred) != len(seq):
        reasons.append("length_mismatch")
    if any(char not in legal for char in pred):
        reasons.append("illegal_token")
    if pred and set(pred) <= {"."}:
        reasons.append("all_dot")
    balance = 0
    prefix_error = False
    for char in pred:
        if char in "([{":
            balance += 1
        elif char in ")]}":
            balance -= 1
        if balance < 0:
            prefix_error = True
            break
    if prefix_error:
        reasons.append("prefix_error")
    if balance != 0:
        reasons.append("unbalanced")
    if not validate_structure(seq, pred) and not reasons:
        reasons.append("unbalanced")
    return reasons


def token_diagnostic_row(sample: dict, pred: str) -> dict:
    true_struct = sample["struct"]
    bracket_chars = set("()[]{}")
    dot_positions = [idx for idx, char in enumerate(true_struct) if char == "."]
    bracket_positions = [idx for idx, char in enumerate(true_struct) if char in bracket_chars]
    pred_open = sum(1 for char in pred if char in "([{")
    pred_close = sum(1 for char in pred if char in ")]}")
    true_open = sum(1 for char in true_struct if char in "([{")
    true_close = sum(1 for char in true_struct if char in ")]}")
    reasons = token_invalid_reasons(sample["seq"], pred)
    return {
        "id": sample.get("id", ""),
        "seq": sample["seq"],
        "true_struct": true_struct,
        "pred_struct": pred,
        "valid": validate_structure(sample["seq"], pred),
        "reasons": reasons,
        "reason": reasons[0] if reasons else "",
        "open_count": pred_open,
        "close_count": pred_close,
        "true_open_count": true_open,
        "true_close_count": true_close,
        "bracket_token_correct": sum(1 for idx in bracket_positions if idx < len(pred) and pred[idx] == true_struct[idx]),
        "bracket_token_total": len(bracket_positions),
        "dot_token_correct": sum(1 for idx in dot_positions if idx < len(pred) and pred[idx] == "."),
        "dot_token_total": len(dot_positions),
        "illegal_token_count": sum(1 for char in pred if char not in ".()[]{}"),
    }


def run_token(args: argparse.Namespace) -> None:
    user_config = load_config(args.config)
    split_path = Path(user_config["data"][f"{args.split}_jsonl"])
    if not split_path.exists():
        raise SystemExit(f"Input JSONL does not exist: {split_path}")
    if not Path(args.ckpt).exists():
        raise SystemExit(f"Checkpoint not found: {args.ckpt}")
    device = resolve_device(args.device)
    config, tokenizer, checkpoint = load_checkpoint(args.ckpt, device)
    dataset = RNAOmniDataset(split_path, max_length=int(user_config["data"]["max_length"]))
    if args.limit:
        dataset.samples = dataset.samples[: int(args.limit)]
    model = build_model(config, tokenizer, device)
    try:
        model.load_state_dict(checkpoint["model_state"])
    except RuntimeError as exc:
        raise SystemExit("Checkpoint is not compatible with the current model structure.") from exc
    model.eval()

    batch_size = int(args.batch or config.get("training", {}).get("batch_size", 8))
    rows: list[dict] = []
    for start in range(0, len(dataset.samples), batch_size):
        batch_samples = dataset.samples[start : start + batch_size]
        preds, _ = forward_token_structures(model, tokenizer, batch_samples, device)
        rows.extend(token_diagnostic_row(sample, pred) for sample, pred in zip(batch_samples, preds))

    samples = len(rows)
    invalid_counts = {key: 0 for key in ["all_dot", "unbalanced", "prefix_error", "illegal_token", "length_mismatch"]}
    for item in rows:
        for reason in item["reasons"]:
            if reason in invalid_counts:
                invalid_counts[reason] += 1
    total_len = sum(len(item["pred_struct"]) for item in rows)
    bracket_total = sum(item["bracket_token_total"] for item in rows)
    dot_total = sum(item["dot_token_total"] for item in rows)
    avg_true_pairs = sum(item["true_open_count"] for item in rows) / max(1, samples)
    report = {
        "samples": samples,
        "valid_structure_rate": sum(1 for item in rows if item["valid"]) / max(1, samples),
        "all_dot_ratio": invalid_counts["all_dot"] / max(1, samples),
        "avg_pred_open_count": sum(item["open_count"] for item in rows) / max(1, samples),
        "avg_pred_close_count": sum(item["close_count"] for item in rows) / max(1, samples),
        "avg_true_open_count": avg_true_pairs,
        "avg_true_close_count": sum(item["true_close_count"] for item in rows) / max(1, samples),
        "open_close_balance_error": sum(abs(item["open_count"] - item["close_count"]) for item in rows) / max(1, samples),
        "prefix_error_rate": invalid_counts["prefix_error"] / max(1, samples),
        "final_balance_error_rate": invalid_counts["unbalanced"] / max(1, samples),
        "illegal_token_rate": sum(item["illegal_token_count"] for item in rows) / max(1, total_len),
        "bracket_token_accuracy": sum(item["bracket_token_correct"] for item in rows) / max(1, bracket_total),
        "dot_token_accuracy": sum(item["dot_token_correct"] for item in rows) / max(1, dot_total),
        "invalid_reason_counts": invalid_counts,
    }
    unusable = report["valid_structure_rate"] < 0.5
    all_dot = report["all_dot_ratio"] > 0.8
    imbalanced = report["open_close_balance_error"] > max(2.0, 0.25 * max(1.0, avg_true_pairs))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    bad = [item for item in rows if not item["valid"]][:20]
    with (out / "examples.jsonl").open("w", encoding="utf-8") as handle:
        for item in bad:
            handle.write(
                json.dumps(
                    {
                        "id": item["id"],
                        "seq": item["seq"],
                        "true_struct": item["true_struct"],
                        "pred_struct": item["pred_struct"],
                        "reason": item["reason"],
                        "open_count": item["open_count"],
                        "close_count": item["close_count"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    dominant_reason = max(invalid_counts.items(), key=lambda kv: kv[1])[0] if samples else ""
    lines = [
        "# Token Decode Diagnostic",
        "",
        f"- samples: {samples}",
        f"- valid_structure_rate: {report['valid_structure_rate']:.4f}",
        f"- all_dot_ratio: {report['all_dot_ratio']:.4f}",
        f"- dominant invalid reason: {dominant_reason}",
        f"- avg predicted opens/closes: {report['avg_pred_open_count']:.2f} / {report['avg_pred_close_count']:.2f}",
        f"- avg true opens/closes: {report['avg_true_open_count']:.2f} / {report['avg_true_close_count']:.2f}",
        f"- open_close_balance_error: {report['open_close_balance_error']:.4f}",
        f"- bracket_token_accuracy: {report['bracket_token_accuracy']:.4f}",
        f"- dot_token_accuracy: {report['dot_token_accuracy']:.4f}",
        "",
        "## Judgment",
        "",
        f"- token head completely unusable: {'yes' if unusable else 'no'}",
        f"- mainly all-dot: {'yes' if all_dot else 'no'}",
        f"- severe bracket imbalance: {'yes' if imbalanced else 'no'}",
        f"- bracket balancing postprocess needed: {'yes' if imbalanced or report['prefix_error_rate'] > 0 else 'no'}",
        "- tokenfallback as strict ablation metric: no",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"token diagnostic -> {out / 'report.md'}")


def stage_logits(
    model,
    tokenizer,
    samples: list[dict],
    config: dict,
    config_path: str,
    ckpt_path: str,
    split: str,
    logits_file: Path,
    device: torch.device,
    batch_size: int,
) -> dict:
    start = time.time()
    max_len = max(int(sample["length"]) for sample in samples)
    pair_chunks = []
    token_score_chunks = []
    dot_id = tokenizer.token_id(".")
    open_ids = [tokenizer.token_id(token) for token in ["(", "[", "{"]]
    close_ids = [tokenizer.token_id(token) for token in [")", "]", "}"]]
    forward_seconds = 0.0
    with torch.no_grad():
        for start_idx in range(0, len(samples), batch_size):
            batch_samples = samples[start_idx : start_idx + batch_size]
            batch = build_seq2struct_batch(tokenizer, batch_samples, device)
            t0 = time.time()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                segment_ids=batch["segment_ids"],
                task_ids=batch["task_ids"],
                time_steps=batch["time_steps"],
                seq_positions=batch["seq_positions"],
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            forward_seconds += time.time() - t0
            pair = outputs.get("pair_logits")
            if pair is None:
                raise SystemExit("Model did not return pair_logits; strict benchmark requires pair head.")
            padded_pair = torch.zeros((len(batch_samples), max_len, max_len), dtype=torch.float16, device="cpu")
            padded_scores = torch.zeros((len(batch_samples), max_len, 3), dtype=torch.float16, device="cpu")
            struct_logits = outputs["structure_logits"]
            for local_idx, sample in enumerate(batch_samples):
                length = int(sample["length"])
                padded_pair[local_idx, :length, :length] = pair[local_idx, :length, :length].detach().to("cpu", dtype=torch.float16)
                positions = batch["struct_positions"][local_idx, :length]
                logits = struct_logits[local_idx, positions].float()
                probs = torch.softmax(logits, dim=-1)
                scores = torch.stack(
                    [
                        probs[:, open_ids].sum(dim=-1),
                        probs[:, close_ids].sum(dim=-1),
                        probs[:, dot_id],
                    ],
                    dim=-1,
                )
                padded_scores[local_idx, :length] = scores.detach().to("cpu", dtype=torch.float16)
            pair_chunks.append(padded_pair)
            token_score_chunks.append(padded_scores)
    payload = {
        "ids": [sample.get("id", "") for sample in samples],
        "seqs": [sample["seq"] for sample in samples],
        "true_structs": [sample["struct"] for sample in samples],
        "families": [sample.get("family", "OTHER") for sample in samples],
        "lengths": [int(sample["length"]) for sample in samples],
        "pair_logits": torch.cat(pair_chunks, dim=0),
        "struct_token_scores": torch.cat(token_score_chunks, dim=0),
        "config": config_path,
        "ckpt": ckpt_path,
        "split": split,
        "dtype": "float16",
        "created_at": datetime.now().isoformat(),
        "stage_forward_seconds": forward_seconds,
        "stage_total_seconds": time.time() - start,
    }
    logits_file.parent.mkdir(parents=True, exist_ok=True)
    save_start = time.time()
    torch.save(payload, logits_file)
    payload["stage_save_seconds"] = time.time() - save_start
    return payload


def load_staged_logits(logits_file: Path) -> dict:
    if not logits_file.exists():
        raise SystemExit(f"logits_file does not exist: {logits_file}. Run with --stage_logits first.")
    try:
        return torch.load(logits_file, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(logits_file, map_location="cpu")


def slice_stage(stage: dict, limit: int | None) -> dict:
    if not limit:
        return stage
    count = min(int(limit), len(stage.get("ids", [])))
    sliced = dict(stage)
    for key in ("ids", "seqs", "true_structs", "families", "lengths"):
        sliced[key] = list(stage[key])[:count]
    for key in ("pair_logits", "struct_token_scores"):
        if key in stage and stage[key] is not None:
            sliced[key] = stage[key][:count]
    return sliced


def token_prior_matrix(scores: torch.Tensor) -> torch.Tensor:
    open_score = scores[:, 0].float()
    close_score = scores[:, 1].float()
    dot_score = scores[:, 2].float()
    return open_score[:, None] * close_score[None, :] - 0.5 * (dot_score[:, None] + dot_score[None, :])


def decode_one_nussinov_worker(payload: dict) -> dict:
    start = time.time()
    sample_index = int(payload["sample_index"])
    seq = payload["seq"]
    true_struct = payload["true_struct"]
    length = int(payload["length"])
    try:
        torch.set_num_threads(1)
        pair_logits = torch.tensor(payload["pair_logits"], dtype=torch.float32)[:length, :length]
        if payload.get("source", "pair") == "hybrid":
            if payload.get("struct_token_scores") is None:
                raise ValueError("Hybrid decode requires struct_token_logits in staged logits.")
            prior = token_prior_matrix(torch.tensor(payload["struct_token_scores"], dtype=torch.float32)[:length])
        else:
            prior = None
        pred = nussinov_decode(
            seq,
            pair_logits,
            min_loop_length=int(payload.get("min_loop_length", 3)),
            allow_wobble=bool(payload.get("allow_wobble", True)),
            pair_threshold=float(payload.get("threshold", 0.25)),
            nussinov_gamma=float(payload.get("gamma", 2.0)),
            max_pair_fraction=payload.get("max_pair_fraction"),
            token_pair_compatibility=prior,
            token_alpha=float(payload.get("token_alpha", 0.0)),
            input_is_logit=True,
            pair_prior_alpha=float(payload.get("pair_prior_alpha", 0.0)),
        )
        result = row(
            "model",
            {
                "id": payload.get("id", ""),
                "seq": seq,
                "struct": true_struct,
                "pairs": payload.get("pairs", []),
                "family": payload.get("family", "OTHER"),
                "length": length,
            },
            pred,
            sample_index,
        )
        result["decode_seconds"] = time.time() - start
        result["error"] = ""
        result["pred_pair_count"] = len(result.get("pred_pairs", []))
        result["true_pair_count"] = len(parse_dot_bracket(true_struct))
        return result
    except Exception as exc:
        return {
            "method": "model",
            "sample_index": sample_index,
            "id": payload.get("id", ""),
            "seq": seq,
            "true_struct": true_struct,
            "pred_struct": "." * length,
            "family": payload.get("family", "OTHER"),
            "length": length,
            "pair_precision": 0.0,
            "pair_recall": 0.0,
            "pair_f1": 0.0,
            "valid": False,
            "canonical_pair_ratio": 0.0,
            "decode_seconds": time.time() - start,
            "error": str(exc),
            "pred_pair_count": 0,
            "true_pair_count": 0,
        }


def write_benchmark_csv(path: Path, summary: dict) -> None:
    fieldnames = [
        "method",
        "pair_precision",
        "pair_recall",
        "pair_f1",
        "valid_structure_rate",
        "canonical_pair_ratio",
        "all_dot_ratio",
        "avg_pred_pair_count",
        "avg_true_pair_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method, metrics in summary.get("overall", {}).items():
            writer.writerow({"method": method, **{key: metrics.get(key, 0.0) for key in fieldnames if key != "method"}})


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return float(ordered[index])


def decode_staged_nussinov(stage: dict, args: argparse.Namespace, config: dict, out_dir: Path) -> tuple[list[dict], dict]:
    lengths = [int(value) for value in stage["lengths"]]
    count = len(lengths)
    threshold = float(args.threshold if args.threshold is not None else config["decoding"].get("pair_threshold", 0.25))
    gamma = float(args.gamma if args.gamma is not None else config["decoding"].get("nussinov_gamma", 2.0))
    max_pair_fraction = config["decoding"].get("max_pair_fraction")
    source = str(args.source or "pair")
    if source == "hybrid" and "struct_token_scores" not in stage:
        raise SystemExit("Hybrid decode requires struct_token_logits in staged logits.")
    pair_logits = stage["pair_logits"]
    struct_scores = stage.get("struct_token_scores")
    payloads = []
    for idx in range(count):
        length = lengths[idx]
        pair_payload = pair_logits[idx, :length, :length].float().numpy()
        struct_payload = None if struct_scores is None else struct_scores[idx, :length].float().numpy()
        payloads.append(
            {
                "sample_index": idx,
                "id": stage["ids"][idx],
                "seq": stage["seqs"][idx],
                "true_struct": stage["true_structs"][idx],
                "family": stage["families"][idx],
                "length": length,
                "pair_logits": pair_payload,
                "struct_token_scores": struct_payload,
                "min_loop_length": int(config["decoding"].get("min_loop_length", 3)),
                "allow_wobble": bool(config["decoding"].get("allow_wobble", True)),
                "threshold": threshold,
                "gamma": gamma,
                "max_pair_fraction": max_pair_fraction,
                "source": source,
                "token_alpha": float(args.token_alpha),
                "pair_prior_alpha": float(args.pair_prior_alpha) if args.pair_prior == "auto" else 0.0,
            }
        )
    start = time.time()
    rows: list[dict] = []
    if int(args.workers) > 0:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=int(args.workers), mp_context=context) as executor:
            rows = list(executor.map(decode_one_nussinov_worker, payloads, chunksize=max(1, int(args.chunksize))))
    else:
        rows = [decode_one_nussinov_worker(payload) for payload in payloads]
    decode_seconds = time.time() - start
    errors = [row for row in rows if row.get("error")]
    if errors:
        with (out_dir / "errors.jsonl").open("w", encoding="utf-8") as handle:
            for item in errors:
                handle.write(json.dumps(item) + "\n")
    decode_times = [float(row.get("decode_seconds", 0.0)) for row in rows]
    slowest = sorted(rows, key=lambda item: float(item.get("decode_seconds", 0.0)), reverse=True)[:20]
    timing = {
        "nussinov_decode_seconds": decode_seconds,
        "workers": int(args.workers),
        "chunksize": int(args.chunksize),
        "p50_decode_seconds": percentile(decode_times, 0.50),
        "p90_decode_seconds": percentile(decode_times, 0.90),
        "p95_decode_seconds": percentile(decode_times, 0.95),
        "max_decode_seconds": max(decode_times) if decode_times else 0.0,
        "slowest_samples": [
            {
                "sample_index": item.get("sample_index"),
                "id": item.get("id", ""),
                "length": item.get("length"),
                "decode_seconds": item.get("decode_seconds", 0.0),
                "true_pair_count": item.get("true_pair_count", 0),
                "pred_pair_count": item.get("pred_pair_count", 0),
            }
            for item in slowest
        ],
        "decode_error_count": len(errors),
    }
    return rows, timing


def run_scan(args: argparse.Namespace, config: dict, stage: dict, out_dir: Path, split_path: Path, dataset_samples: list[dict]) -> None:
    scan_path = Path(args.scan)
    if not scan_path.exists():
        raise SystemExit(f"Scan file does not exist: {scan_path}")
    specs = json.loads(scan_path.read_text(encoding="utf-8"))
    if not isinstance(specs, list):
        raise SystemExit("Scan file must contain a JSON list of decode settings.")
    summary_rows = []
    scan_root = out_dir / "scan"
    scan_root.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        name = str(spec.get("name") or f"scan{len(summary_rows)}")
        item_out = scan_root / name
        item_out.mkdir(parents=True, exist_ok=True)
        scan_args = argparse.Namespace(**vars(args))
        scan_args.scan = None
        scan_args.decode = "nussinov"
        scan_args.fast = False
        scan_args.threshold = spec.get("threshold", args.threshold)
        scan_args.gamma = spec.get("gamma", args.gamma)
        scan_args.source = spec.get("source", args.source or "pair")
        scan_args.token_alpha = float(spec.get("token_alpha", args.token_alpha))
        scan_args.pair_prior = spec.get("pair_prior", args.pair_prior)
        scan_args.pair_prior_alpha = float(spec.get("pair_prior_alpha", args.pair_prior_alpha))
        scan_args.out = str(item_out / "benchmark.json")
        scan_args.workers = args.workers
        scan_args.chunksize = args.chunksize
        timing_start = time.time()
        rows, timing = decode_staged_nussinov(stage, scan_args, config, item_out)
        timing.update(
            {
                "stage_forward_seconds": float(stage.get("stage_forward_seconds", 0.0)),
                "stage_save_seconds": float(stage.get("stage_save_seconds", 0.0)),
                "device": args.device,
                "cuda": args.device == "cuda",
                "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" and torch.cuda.is_available() else "",
            }
        )
        result = finalize_benchmark(
            rows=rows,
            dataset_samples=dataset_samples,
            indexed_samples=list(enumerate(dataset_samples)),
            args=scan_args,
            config=config,
            split_path=split_path,
            out_dir=item_out,
            bench_path=item_out / "benchmark.json",
            pred_path=item_out / "predictions.jsonl",
            decode_method="nussinov",
            start_time=timing_start,
            timing=timing,
        )
        metrics = result.get("overall", {}).get("model", {})
        avg_true = float(metrics.get("avg_true_pair_count", 0.0))
        summary_rows.append(
            {
                "name": name,
                "source": scan_args.source,
                "threshold": float(scan_args.threshold if scan_args.threshold is not None else config["decoding"].get("pair_threshold", 0.25)),
                "gamma": float(scan_args.gamma if scan_args.gamma is not None else config["decoding"].get("nussinov_gamma", 2.0)),
                "alpha": float(scan_args.token_alpha),
                "pair_f1": float(metrics.get("pair_f1", 0.0)),
                "pair_precision": float(metrics.get("pair_precision", 0.0)),
                "pair_recall": float(metrics.get("pair_recall", 0.0)),
                "pair_ratio": float(metrics.get("avg_pred_pair_count", 0.0)) / max(1e-8, avg_true),
                "all_dot": float(metrics.get("all_dot_ratio", 0.0)),
                "valid": float(metrics.get("valid_structure_rate", 0.0)),
                "seconds": float(result.get("benchmark_seconds", 0.0)),
            }
        )
    (scan_root / "summary.json").write_text(json.dumps(summary_rows, indent=2) + "\n", encoding="utf-8")
    with (scan_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()) if summary_rows else ["name"])
        writer.writeheader()
        writer.writerows(summary_rows)
    lines = [
        "| Name | Source | Threshold | Gamma | Alpha | Pair F1 | Precision | Recall | Pair Ratio | All-dot | Valid | Seconds |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary_rows:
        lines.append(
            f"| {item['name']} | {item['source']} | {item['threshold']:.2f} | {item['gamma']:.2f} | {item['alpha']:.2f} | "
            f"{item['pair_f1']:.4f} | {item['pair_precision']:.4f} | {item['pair_recall']:.4f} | "
            f"{item['pair_ratio']:.4f} | {item['all_dot']:.4f} | {item['valid']:.4f} | {item['seconds']:.2f} |"
        )
    (scan_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"scan summary -> {scan_root / 'summary.md'}")


def finalize_benchmark(
    *,
    rows: list[dict],
    dataset_samples: list[dict],
    indexed_samples: list[tuple[int, dict]],
    args: argparse.Namespace,
    config: dict,
    split_path: Path,
    out_dir: Path,
    bench_path: Path,
    pred_path: Path,
    decode_method: str,
    start_time: float,
    timing: dict,
    resumed: bool = False,
    skipped_crossing_counts: list[int] | None = None,
) -> dict:
    metric_start = time.time()
    all_rows = [row("all", sample, "." * sample["length"], idx) for idx, sample in indexed_samples]
    random_rows = [
        row("random", sample, random_valid(sample["seq"], int(config["decoding"].get("min_loop_length", 3))), idx)
        for idx, sample in indexed_samples
    ]
    metric_seconds = time.time() - metric_start
    write_start = time.time()
    with pred_path.open("w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item) + "\n")
    skipped = skipped_crossing_counts or [0] * len(rows)
    total_seconds = time.time() - start_time
    summary = {
        "decode_method": decode_method,
        "partial": bool(args.limit or args.samples),
        "samples": len(rows),
        "pair_head_available": bool(timing.get("pair_head_available", decode_method not in {"token", "tokenfallback"})),
        "decode_warning": GREEDY_DECODE_WARNING if decode_method == "greedy" else "",
        "skipped_crossing_pairs_total": int(sum(skipped)),
        "skipped_crossing_pairs_avg": float(sum(skipped) / max(1, len(skipped))),
        "crossing_sample_ratio": float(sum(1 for value in skipped if value > 0) / max(1, len(skipped))),
        "decode_error_count": int(timing.get("decode_error_count", 0)),
        "device": timing.get("device", "unknown"),
        "cuda": bool(timing.get("cuda", False)),
        "gpu": timing.get("gpu", ""),
        "benchmark_seconds": total_seconds,
        "samples_per_sec": len(rows) / max(1e-8, total_seconds),
        "overall": {"model": summarize(rows), "all": summarize(all_rows), "random": summarize(random_rows)},
    }
    benchmeta = {
        "config": args.config,
        "ckpt": args.ckpt,
        "split": args.split,
        "input_jsonl": str(split_path),
        "decode_method": decode_method,
        "pair_head_available": summary["pair_head_available"],
        "total_samples": len(dataset_samples),
        "completed_samples": len(rows),
        "partial": bool(args.limit or args.samples),
        "resumed": resumed,
        "device": summary["device"],
        "cuda": summary["cuda"],
        "gpu": summary["gpu"],
        "batch": int(args.batch or config.get("training", {}).get("batch_size", 8)),
        "workers": int(args.workers),
        "chunksize": int(args.chunksize),
        "threshold": args.threshold if args.threshold is not None else config["decoding"].get("pair_threshold", 0.25),
        "gamma": args.gamma if args.gamma is not None else config["decoding"].get("nussinov_gamma", 2.0),
        "max_pair_fraction": config["decoding"].get("max_pair_fraction"),
        "source": args.source or "pair",
        "token_alpha": float(args.token_alpha),
        "pair_prior": args.pair_prior,
        "pair_prior_alpha": float(args.pair_prior_alpha) if args.pair_prior == "auto" else 0.0,
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.now().isoformat(),
    }
    benchtime = {
        "total_seconds": total_seconds,
        "stage_forward_seconds": float(timing.get("stage_forward_seconds", 0.0)),
        "stage_save_seconds": float(timing.get("stage_save_seconds", 0.0)),
        "forward_seconds": float(timing.get("forward_seconds", 0.0)),
        "decode_seconds": float(timing.get("decode_seconds", timing.get("nussinov_decode_seconds", 0.0))),
        "nussinov_decode_seconds": float(timing.get("nussinov_decode_seconds", 0.0)),
        "metric_seconds": metric_seconds,
        "write_seconds": time.time() - write_start,
        "samples_per_sec": summary["samples_per_sec"],
        "workers": int(args.workers),
        "chunksize": int(args.chunksize),
        "p50_decode_seconds": float(timing.get("p50_decode_seconds", 0.0)),
        "p90_decode_seconds": float(timing.get("p90_decode_seconds", 0.0)),
        "p95_decode_seconds": float(timing.get("p95_decode_seconds", 0.0)),
        "max_decode_seconds": float(timing.get("max_decode_seconds", 0.0)),
        "length_bucket_speed": {},
        "slowest_samples": timing.get("slowest_samples", []),
    }
    bench_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_benchmark_csv(bench_path.with_suffix(".csv"), summary)
    (out_dir / "benchmeta.json").write_text(json.dumps(benchmeta, indent=2) + "\n", encoding="utf-8")
    (out_dir / "benchtime.json").write_text(json.dumps(benchtime, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["overall"], indent=2))
    print(f"benchmark -> {bench_path}")
    print(f"predictions -> {pred_path}")
    if args.profile:
        print(json.dumps(benchtime, indent=2))
    return summary


def run_bench(args: argparse.Namespace) -> None:
    start_time = time.time()
    user_config = load_config(args.config)
    split_path = Path(args.input) if args.input else Path(user_config["data"][f"{args.split}_jsonl"])
    if not split_path.exists():
        raise SystemExit(f"Input JSONL does not exist: {split_path}")
    if not Path(args.ckpt).exists():
        raise SystemExit(
            "Checkpoint not found. Run: conda run -n DL python scripts\\run.py potential "
            "--config config/candidate.yaml --mode full --device cuda"
        )
    device = resolve_device(args.device)
    config, tokenizer, checkpoint = load_checkpoint(args.ckpt, device)
    config["decoding"] = {**config.get("decoding", {}), **user_config.get("decoding", {})}
    dataset = RNAOmniDataset(split_path, max_length=int(user_config["data"]["max_length"]))
    limit = args.limit if args.limit is not None else args.samples
    if limit:
        dataset.samples = dataset.samples[: int(limit)]

    requested_decode = "greedy" if args.fast else args.decode
    ablation = config.get("ablation", {})
    pair_head_available = bool(ablation.get("use_pair_head", True) and config.get("model", {}).get("use_pair_head", True))
    if requested_decode == "nussinov" and not pair_head_available:
        decode_method = "tokenfallback"
    elif requested_decode == "nussinov" and not bool(config.get("decoding", {}).get("use_nussinov", True)):
        decode_method = "token"
    else:
        decode_method = requested_decode
    batch_size = int(args.batch or config.get("training", {}).get("batch_size", 8))
    out_dir = Path(args.out).parent if args.out else Path(config["training"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = Path(args.out) if args.out else out_dir / "benchmark"
    bench_path = prefix if prefix.suffix == ".json" else prefix.with_suffix(".json")
    pred_path = out_dir / "predictions.jsonl"
    tmp_path = out_dir / "predictions.tmp.jsonl"

    if args.threshold is not None:
        config.setdefault("decoding", {})["pair_threshold"] = float(args.threshold)
    if args.gamma is not None:
        config.setdefault("decoding", {})["nussinov_gamma"] = float(args.gamma)

    need_model = decode_method in {"greedy", "token", "tokenfallback", "greedyfallback"} or not args.decode_only
    model = None
    if need_model:
        model = build_model(config, tokenizer, device)
        try:
            model.load_state_dict(checkpoint["model_state"])
        except RuntimeError as exc:
            raise SystemExit(
                "Checkpoint is not compatible with the current model structure. "
                "The pair head was changed to an MLP; retrain the checkpoint or use a matching config."
            ) from exc
        model.eval()

    if decode_method == "nussinov":
        logits_file = Path(args.logits_file) if args.logits_file else out_dir / "logits.pt"
        if args.decode_only:
            stage = slice_stage(load_staged_logits(logits_file), limit)
        else:
            assert model is not None
            stage = stage_logits(
                model,
                tokenizer,
                dataset.samples,
                config,
                args.config,
                args.ckpt,
                args.split,
                logits_file,
                device,
                batch_size,
            )
            stage = slice_stage(stage, limit)
        if args.scan:
            run_scan(args, config, stage, out_dir, split_path, dataset.samples)
            return
        rows, timing = decode_staged_nussinov(stage, args, config, out_dir)
        timing.update(
            {
                "stage_forward_seconds": float(stage.get("stage_forward_seconds", 0.0)),
                "stage_save_seconds": float(stage.get("stage_save_seconds", 0.0)),
                "device": device.type,
                "cuda": bool(device.type == "cuda"),
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
                "pair_head_available": pair_head_available,
            }
        )
        finalize_benchmark(
            rows=rows,
            dataset_samples=dataset.samples,
            indexed_samples=list(enumerate(dataset.samples)),
            args=args,
            config=config,
            split_path=split_path,
            out_dir=out_dir,
            bench_path=bench_path,
            pred_path=pred_path,
            decode_method="nussinov",
            start_time=start_time,
            timing=timing,
        )
        return

    completed_indices = set()
    model_rows: list[dict] = []
    resumed = False
    meta_path = out_dir / "benchmeta.json"
    if args.resume and pred_path.exists() and meta_path.exists():
        old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if old_meta.get("decode_method") != decode_method:
            print(
                f"Resume ignored because existing decode_method={old_meta.get('decode_method')} "
                f"does not match requested decode_method={decode_method}.",
                file=sys.stderr,
            )
        else:
            resumed = True
    elif args.resume and pred_path.exists():
        print("Resume ignored because benchmeta.json is missing.", file=sys.stderr)
    if resumed:
        for line in pred_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                if "sample_index" in item and item["sample_index"] is not None:
                    completed_indices.add(int(item["sample_index"]))
                    model_rows.append(item)
                else:
                    resumed = False
                    completed_indices.clear()
                    model_rows.clear()
                    print("Resume ignored because existing predictions do not contain sample_index.", file=sys.stderr)
                    break
    indexed_samples = list(enumerate(dataset.samples))
    remaining = [(idx, sample) for idx, sample in indexed_samples if idx not in completed_indices]

    forward_seconds = decode_seconds = metric_seconds = write_seconds = 0.0
    skipped_crossing_counts: list[int] = []
    handle = tmp_path.open("a" if resumed else "w", encoding="utf-8")
    if resumed:
        for item in model_rows:
            handle.write(json.dumps(item) + "\n")
    try:
        for start in range(0, len(remaining), batch_size):
            batch_items = remaining[start : start + batch_size]
            batch_indices = [idx for idx, _ in batch_items]
            batch_samples = [sample for _, sample in batch_items]
            if decode_method in {"token", "tokenfallback"}:
                assert model is not None
                structs, elapsed = forward_token_structures(model, tokenizer, batch_samples, device)
                forward_seconds += elapsed
                skipped_crossing_counts.extend([0] * len(batch_samples))
            elif decode_method == "greedy":
                t0 = time.time()
                assert model is not None
                pair_logits = forward_pair_logits(model, tokenizer, batch_samples, device)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                forward_seconds += time.time() - t0
                t0 = time.time()
                pair_matrix = batched_greedy_decode_gpu(
                    pair_logits,
                    seqs=[sample["seq"] for sample in batch_samples],
                    min_loop_length=int(config["decoding"].get("min_loop_length", 3)),
                    pair_threshold=float(config["decoding"].get("pair_threshold", 0.25)),
                    allow_wobble=bool(config["decoding"].get("allow_wobble", True)),
                    canonical_only=True,
                    prevent_crossing=False,
                )
                structs, skipped = pairs_matrix_to_dotbracket_batch_with_stats(
                    pair_matrix,
                    [int(sample["length"]) for sample in batch_samples],
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                decode_seconds += time.time() - t0
                skipped_crossing_counts.extend(skipped)
            else:
                structs = []
                for sample in batch_samples:
                    t0 = time.time()
                    structs.append(generate_structure_seq2struct(model, tokenizer, sample["seq"], config["decoding"], device))
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    decode_seconds += time.time() - t0
                    skipped_crossing_counts.append(0)

            t0 = time.time()
            for sample_index, sample, pred in zip(batch_indices, batch_samples, structs):
                item = row("model", sample, pred, sample_index)
                model_rows.append(item)
                handle.write(json.dumps(item) + "\n")
            metric_seconds += time.time() - t0
            if len(model_rows) % max(1, int(args.save_every)) == 0:
                t0 = time.time()
                handle.flush()
                write_seconds += time.time() - t0
    finally:
        handle.close()
    tmp_path.replace(pred_path)

    all_rows = [row("all", sample, "." * sample["length"], idx) for idx, sample in indexed_samples]
    random_rows = [
        row("random", sample, random_valid(sample["seq"], int(config["decoding"].get("min_loop_length", 3))), idx)
        for idx, sample in indexed_samples
    ]
    total_seconds = time.time() - start_time
    summary = {
        "decode_method": decode_method,
        "partial": bool(limit),
        "samples": len(model_rows),
        "decode_warning": GREEDY_DECODE_WARNING if decode_method == "greedy" else "",
        "skipped_crossing_pairs_total": int(sum(skipped_crossing_counts)),
        "skipped_crossing_pairs_avg": float(sum(skipped_crossing_counts) / max(1, len(skipped_crossing_counts))),
        "crossing_sample_ratio": float(sum(1 for value in skipped_crossing_counts if value > 0) / max(1, len(skipped_crossing_counts))),
        "device": device.type,
        "cuda": bool(device.type == "cuda"),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
        "pair_head_available": pair_head_available,
        "benchmark_seconds": total_seconds,
        "samples_per_sec": len(model_rows) / max(1e-8, total_seconds),
        "overall": {"model": summarize(model_rows), "all": summarize(all_rows), "random": summarize(random_rows)},
    }
    benchmeta = {
        "config": args.config,
        "ckpt": args.ckpt,
        "split": args.split,
        "input_jsonl": str(split_path),
        "decode_method": decode_method,
        "pair_head_available": pair_head_available,
        "total_samples": len(dataset.samples),
        "completed_samples": len(model_rows),
        "partial": bool(limit),
        "resumed": resumed,
        "device": summary["device"],
        "cuda": summary["cuda"],
        "gpu": summary["gpu"],
        "batch": batch_size,
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.now().isoformat(),
    }
    batch_count = max(1, (len(remaining) + batch_size - 1) // batch_size)
    benchtime = {
        "total_seconds": total_seconds,
        "forward_seconds": forward_seconds,
        "decode_seconds": decode_seconds,
        "metric_seconds": metric_seconds,
        "write_seconds": write_seconds,
        "samples_per_sec": summary["samples_per_sec"],
        "avg_forward_seconds_per_batch": forward_seconds / batch_count,
        "avg_decode_seconds_per_batch": decode_seconds / batch_count,
        "length_bucket_speed": {},
    }
    bench_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_benchmark_csv(bench_path.with_suffix(".csv"), summary)
    (out_dir / "benchmeta.json").write_text(json.dumps(benchmeta, indent=2) + "\n", encoding="utf-8")
    (out_dir / "benchtime.json").write_text(json.dumps(benchtime, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["overall"], indent=2))
    print(f"benchmark -> {bench_path}")
    print(f"predictions -> {pred_path}")
    if args.profile:
        print(json.dumps(benchtime, indent=2))


def run_export(args: argparse.Namespace) -> None:
    args.split = "test"
    args.limit = args.samples
    args.decode = "greedy"
    args.fast = True
    args.profile = False
    args.resume = False
    args.save_every = 100
    args.batch = None
    args.workers = 0
    args.chunksize = 4
    args.stage_logits = False
    args.logits_file = None
    args.decode_only = False
    args.threshold = None
    args.gamma = None
    args.source = "pair"
    args.token_alpha = 0.0
    args.pair_prior = "none"
    args.pair_prior_alpha = 0.0
    args.scan = None
    run_bench(args)


def run_analyze(args: argparse.Namespace) -> None:
    path = Path(args.log)
    if not path.exists():
        raise SystemExit(f"Training log does not exist: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if rows and any("val_pair_f1" in row for row in rows):
        best = max(rows, key=lambda r: float(r.get("val_pair_f1", 0.0)))
    elif rows:
        best = min(rows, key=lambda r: float(r.get("val_loss", float("inf"))))
    else:
        best = {}
    result = {
        "epochs": len(rows),
        "best": best,
        "gap": float(best.get("gap", best.get("positive_pair_logit_mean", 0.0) - best.get("negative_pair_logit_mean", 0.0))),
        "rankAcc": best.get("rankAcc"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"analysis -> {out}")


def run_diagnose(args: argparse.Namespace) -> None:
    path = Path(args.pred)
    if not path.exists():
        raise SystemExit(f"Prediction file does not exist: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    worst = sorted(rows, key=lambda r: float(r.get("pair_f1", 0.0)))[:20]
    result = {"count": len(rows), "worst": worst, "allDot": sum(1 for r in rows if set(r.get("pred_struct", "")) <= {"."})}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"diagnosis -> {out}")


def run_compare(args: argparse.Namespace) -> None:
    rows = []
    for name, file in zip(args.names, args.inputs):
        data = json.loads(Path(file).read_text(encoding="utf-8"))
        metrics = data.get("overall", {}).get("model", {})
        rows.append({"name": name, **metrics})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        keys = sorted({k for item in rows for k in item})
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| Variant | Pair F1 | All-dot | Valid | Pair Ratio |", "|---|---:|---:|---:|---:|"]
    for item in rows:
        ratio = float(item.get("avg_pred_pair_count", 0.0)) / max(1e-8, float(item.get("avg_true_pair_count", 0.0)))
        lines.append(f"| {item['name']} | {float(item.get('pair_f1', 0.0)):.4f} | {float(item.get('all_dot_ratio', 0.0)):.4f} | {float(item.get('valid_structure_rate', 0.0)):.4f} | {ratio:.4f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"compare -> {out / 'summary.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RNA-OmniPrefold.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    bench = sub.add_parser("bench")
    bench.add_argument("--config", default="config/candidate.yaml")
    bench.add_argument("--ckpt", required=True)
    bench.add_argument("--split", default="test", choices=["train", "val", "test"])
    bench.add_argument("--input")
    bench.add_argument("--out")
    bench.add_argument("--samples", type=int, help=argparse.SUPPRESS)
    bench.add_argument("--limit", type=int)
    bench.add_argument("--decode", choices=["nussinov", "greedy", "token", "tokenfallback", "greedyfallback"], default="nussinov")
    bench.add_argument("--batch", type=int)
    bench.add_argument("--fast", action="store_true")
    bench.add_argument("--profile", action="store_true")
    bench.add_argument("--resume", action="store_true")
    bench.add_argument("--save_every", type=int, default=100)
    bench.add_argument("--workers", type=int, default=0)
    bench.add_argument("--chunksize", type=int, default=4)
    bench.add_argument("--stage_logits", action="store_true")
    bench.add_argument("--logits_file")
    bench.add_argument("--decode_only", action="store_true")
    bench.add_argument("--threshold", type=float)
    bench.add_argument("--gamma", type=float)
    bench.add_argument("--source", choices=["pair", "hybrid"], default="pair")
    bench.add_argument("--token_alpha", type=float, default=0.0)
    bench.add_argument("--pair_prior", choices=["none", "auto"], default="none")
    bench.add_argument("--pair_prior_alpha", type=float, default=0.0)
    bench.add_argument("--scan")
    bench.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    bench.set_defaults(func=run_bench)
    export = sub.add_parser("export")
    export.add_argument("--config", default="config/candidate.yaml")
    export.add_argument("--ckpt", required=True)
    export.add_argument("--input", required=True)
    export.add_argument("--out", required=True)
    export.add_argument("--samples", type=int)
    export.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    export.set_defaults(func=run_export)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--log", required=True)
    analyze.add_argument("--out", required=True)
    analyze.set_defaults(func=run_analyze)
    diagnose = sub.add_parser("diagnose")
    diagnose.add_argument("--pred", required=True)
    diagnose.add_argument("--out", required=True)
    diagnose.set_defaults(func=run_diagnose)
    compare = sub.add_parser("compare")
    compare.add_argument("--inputs", nargs="+", required=True)
    compare.add_argument("--names", nargs="+", required=True)
    compare.add_argument("--out", required=True)
    compare.set_defaults(func=run_compare)
    token = sub.add_parser("token")
    token.add_argument("--config", default="config/candidate.yaml")
    token.add_argument("--ckpt", required=True)
    token.add_argument("--split", default="test", choices=["train", "val", "test"])
    token.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    token.add_argument("--limit", type=int)
    token.add_argument("--batch", type=int)
    token.add_argument("--out", required=True)
    token.set_defaults(func=run_token)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
