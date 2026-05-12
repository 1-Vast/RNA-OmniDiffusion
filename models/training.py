from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import BatchSampler, DataLoader, Sampler, Subset

from models.collator import RNAOmniCollator
from models.dataset import RNAOmniDataset
from models.token import RNAOmniTokenizer
from models.decode import generate_sequence_invfold, generate_structure_seq2struct
from models.omni import RNAOmniDiffusion, compute_omni_loss
from models.rank import local_rank_loss
from utils.metric import base_pair_f1, evaluate_structures
from models.display import (
    checkpoint_saved,
    early_stopping_summary,
    epoch_line,
    inference_header,
    inference_result_invfold,
    inference_result_seq2struct,
    train_startup,
    training_complete,
)


def load_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config = normalize_config(config or {})
    return apply_ablation_settings(config)


def normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    data = config.setdefault("data", {})
    if "train" in data:
        data["train_jsonl"] = data["train"]
    if "val" in data:
        data["val_jsonl"] = data["val"]
    if "test" in data:
        data["test_jsonl"] = data["test"]
    if "maxlen" in data:
        data["max_length"] = data["maxlen"]

    training = config.setdefault("training", {})
    aliases = {
        "out": "output_dir",
        "lambdaPair": "lambda_pair",
        "lambdaStruct": "lambda_struct",
        "lambdaSeq": "lambda_seq",
        "pairRatio": "pair_negative_ratio",
        "pairWeight": "pair_positive_weight",
        "clip": "grad_clip",
    }
    for src, dst in aliases.items():
        if src in training:
            training[dst] = training[src]
    training.setdefault("lambda_pair", training.get("lambdaPair", 0.5))
    training.setdefault("pair_positive_weight", training.get("pairWeight", "auto"))
    training.setdefault("pair_negative_ratio", training.get("pairRatio", 3))
    training.setdefault("output_dir", training.get("out", "outputs/archive"))

    decoding = config.setdefault("decoding", {})
    if "threshold" in decoding:
        decoding["pair_threshold"] = decoding["threshold"]
    if "gamma" in decoding:
        decoding["nussinov_gamma"] = decoding["gamma"]
    if "source" in decoding:
        decoding["decode_source"] = decoding["source"]

    model = config.setdefault("model", {})
    model.setdefault("pairhead", "mlp")
    model.setdefault("pairhidden", model.get("hidden_size", 512))
    model.setdefault("pairdrop", model.get("dropout", 0.1))
    model.setdefault("distbias", False)
    model.setdefault("distbuckets", 32)
    model.setdefault("distmax", data.get("max_length", 512))
    model.setdefault("invalidlogit", -20.0)
    model.setdefault("pairrefine", False)
    model.setdefault("pairrefinechannels", 16)
    model.setdefault("pairrefineblocks", 1)
    model.setdefault("pairrefinetype", "conv2d")
    model.setdefault("pairrefinedrop", 0.0)
    tasks = config.setdefault("tasks", {})
    if "seq_denoise" in tasks and "denoise" not in tasks:
        tasks["denoise"] = tasks["seq_denoise"]
    pref = config.setdefault("preference", {})
    pref.setdefault("enabled", False)
    pref.setdefault("buffer_path", None)
    pref.setdefault("beta", 0.05)
    pref.setdefault("min_confidence", 0.6)
    pref.setdefault("start_epoch_ratio", 0.3)
    pref.setdefault("loss_type", "ranking")
    typed = config.setdefault("typed", {})
    typed.setdefault("enabled", False)
    typed.setdefault("spec_path", None)
    policy = config.setdefault("policy", {})
    policy.setdefault("enabled", False)
    policy.setdefault("path", None)
    relmask = config.setdefault("relation_mask", {})
    relmask.setdefault("enabled", False)
    relmask.setdefault("total_ratio", 0.25)
    relmask.setdefault("hard_negative_ratio", 0.25)
    relmask.setdefault("stem_span_ratio", 0.35)
    relmask.setdefault("global_ratio", 0.40)
    relmask.setdefault("stem_span_len", 4)
    relmask.setdefault("long_range_threshold", 64)
    relmask.setdefault("log_stats", True)
    plp = config.setdefault("pair_loss_policy", {})
    plp.setdefault("enabled", False)
    plp.setdefault("positive_weight", 1.0)
    plp.setdefault("negative_weight", 1.0)
    plp.setdefault("hard_negative_weight", 1.0)
    plp.setdefault("long_range_positive_weight", 1.0)
    plp.setdefault("long_range_negative_weight", 1.0)
    plp.setdefault("pair_ratio_target", None)
    plp.setdefault("pair_ratio_weight", 0.0)
    plp.setdefault("isolated_pair_penalty", 0.0)
    plp.setdefault("stem_continuity_weight", 0.0)
    plp.setdefault("long_range_threshold", 64)
    plp.setdefault("min_loop", 4)
    plp.setdefault("start_epoch_ratio", 0.0)
    plp.setdefault("schedule", "constant")
    plp.setdefault("max_aux_loss", 5.0)
    plp.setdefault("log_stats", True)
    pa = config.setdefault("pair_arch", {})
    pa.setdefault("enabled", False)
    pa.setdefault("type", "base")
    pa.setdefault("dist_buckets", 32)
    pa.setdefault("distance_threshold", 48)
    pa.setdefault("dropout", 0.0)
    pr = config.setdefault("pair_residual", {})
    pr.setdefault("enabled", False)
    pr.setdefault("type", "conv2d")
    pr.setdefault("kernel_size", 3)
    pr.setdefault("residual_scale", 0.1)
    pr.setdefault("init", "zero")
    psb = config.setdefault("pair_specific_bias", {})
    psb.setdefault("enabled", False)
    psb.setdefault("type_bias", True)
    psb.setdefault("distance_bias", True)
    psb.setdefault("init_from_prior", True)
    psb.setdefault("learnable", True)
    lrk = config.setdefault("local_rank_loss", {})
    lrk.setdefault("enabled", False)
    lrk.setdefault("margin", 1.0)
    lrk.setdefault("weight", 0.1)
    lrk.setdefault("negatives_per_positive", 4)
    lrk.setdefault("warmup_steps", 100)
    return config


def resolve_device(name: str | None = "auto") -> torch.device:
    requested = (name or "auto").lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested with --device cuda, but torch.cuda.is_available() is False.")
    device = torch.device(requested)
    if device.type == "cuda":
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    return device


def apply_ablation_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    ablation = config.get("ablation", {}) or {}
    decoding = config.setdefault("decoding", {})
    if "use_nussinov" in ablation:
        decoding["use_nussinov"] = bool(ablation.get("use_nussinov", True))
    if "decode_source" in ablation:
        decoding["decode_source"] = str(ablation.get("decode_source", decoding.get("decode_source", "pair")))
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def synthetic_samples() -> list[dict]:
    return [
        {
            "id": "toy_0001",
            "seq": "GCAUAGC",
            "struct": "((...))",
            "family": "miRNA",
            "motifs": [{"type": "STEM", "start": 0, "end": 6}, {"type": "HAIRPIN", "start": 2, "end": 4}],
        },
        {
            "id": "toy_0002",
            "seq": "GGGAAACCC",
            "struct": "(((...)))",
            "family": "riboswitch",
            "motifs": [{"type": "STEM", "start": 0, "end": 8}, {"type": "HAIRPIN", "start": 3, "end": 5}],
        },
        {
            "id": "toy_0003",
            "seq": "AUGCAU",
            "struct": "(....)",
            "family": "tRNA",
            "motifs": [{"type": "HAIRPIN", "start": 1, "end": 4}],
        },
        {
            "id": "toy_0004",
            "seq": "GCUAAGC",
            "struct": "((...))",
            "family": "miRNA",
        },
    ]


def create_tiny_jsonl_dataset(config: Dict[str, Any], overwrite: bool = False) -> None:
    samples = synthetic_samples()
    data_cfg = config["data"]
    splits = {
        "train_jsonl": samples,
        "val_jsonl": samples[:2],
        "test_jsonl": samples[2:],
    }
    for key, rows in splits.items():
        path = Path(data_cfg[key])
        if path.exists() and not overwrite:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")


def ensure_dataset_paths(config: Dict[str, Any], create_if_missing: bool) -> None:
    required = [Path(config["data"]["train_jsonl"]), Path(config["data"]["val_jsonl"])]
    if all(path.exists() for path in required):
        return
    if create_if_missing:
        print("Dataset JSONL files not found; creating a tiny synthetic dataset under dataset/processed/.")
        create_tiny_jsonl_dataset(config, overwrite=False)
        return
    missing = ", ".join(str(path) for path in required if not path.exists())
    raise FileNotFoundError(f"Missing required dataset file(s): {missing}")


def build_datasets_and_tokenizer(config: Dict[str, Any]) -> tuple[RNAOmniDataset, RNAOmniDataset, RNAOmniTokenizer]:
    max_length = int(config["data"]["max_length"])
    allow_unlabeled = bool(config["data"].get("allow_unlabeled", False))
    train_dataset = RNAOmniDataset(config["data"]["train_jsonl"], max_length=max_length, allow_unlabeled=allow_unlabeled)
    val_dataset = RNAOmniDataset(config["data"]["val_jsonl"], max_length=max_length, allow_unlabeled=allow_unlabeled)
    tokenizer = RNAOmniTokenizer.from_samples(train_dataset.samples + val_dataset.samples)
    return train_dataset, val_dataset, tokenizer


def build_model(config: Dict[str, Any], tokenizer: RNAOmniTokenizer, device: torch.device) -> RNAOmniDiffusion:
    model_cfg = config["model"]
    model = RNAOmniDiffusion(
        vocab_size=tokenizer.vocab_size,
        hidden_size=int(model_cfg["hidden_size"]),
        num_layers=int(model_cfg["num_layers"]),
        num_heads=int(model_cfg["num_heads"]),
        dropout=float(model_cfg["dropout"]),
        max_position_embeddings=int(model_cfg["max_position_embeddings"]),
        num_tasks=len(tokenizer.task_to_id),
        use_pair_head=bool(config.get("ablation", {}).get("use_pair_head", True)),
        pairhead=str(model_cfg.get("pairhead", "mlp")),
        pairhidden=int(model_cfg.get("pairhidden", model_cfg["hidden_size"])),
        pairdrop=float(model_cfg.get("pairdrop", model_cfg.get("dropout", 0.1))),
        distbias=bool(model_cfg.get("distbias", False)),
        distbuckets=int(model_cfg.get("distbuckets", 32)),
        distmax=int(model_cfg.get("distmax", config.get("data", {}).get("max_length", 512))),
        invalidlogit=float(model_cfg.get("invalidlogit", -20.0)),
        pairrefine=bool(model_cfg.get("pairrefine", False)),
        pairrefinechannels=int(model_cfg.get("pairrefinechannels", 16)),
        pairrefineblocks=int(model_cfg.get("pairrefineblocks", 1)),
        pairrefinedrop=float(model_cfg.get("pairrefinedrop", 0.0)),
        pairrefinetype=str(model_cfg.get("pairrefinetype", "conv2d")),
        pair_arch=str(config.get("pair_arch", {}).get("type", "base")) if (config.get("pair_arch", {}).get("enabled", False) or config.get("pair_residual", {}).get("enabled", False)) else None,
        pair_logit_offset=float(config.get("training", {}).get("pair_logit_offset", 0.0)),
        pair_specific_bias_enabled=bool(config.get("pair_specific_bias", {}).get("enabled", False)),
        pair_specific_bias_init_prior=bool(config.get("pair_specific_bias", {}).get("init_from_prior", True)),
        pair_specific_bias_learnable=bool(config.get("pair_specific_bias", {}).get("learnable", True)),
        pair_specific_bias_use_type=bool(config.get("pair_specific_bias", {}).get("type_bias", True)),
        pair_specific_bias_use_distance=bool(config.get("pair_specific_bias", {}).get("distance_bias", True)),
    )
    return model.to(device)


ENCODER_PRETRAIN_PREFIXES = (
    "token_embedding.",
    "position_embedding.",
    "segment_embedding.",
    "task_embedding.",
    "time_mlp.",
    "encoder.",
    "norm.",
)


def load_encoder_only_pretrain(model: RNAOmniDiffusion, ckpt_path: str | Path, device: torch.device) -> dict:
    # Encoder-only loading skips pair and task-specific heads.
    # Used for RNA-OmniPrefold Stage 1 -> Stage 2 transition.
    _, _, checkpoint = load_checkpoint(ckpt_path, device)
    source = checkpoint["model_state"]
    target = model.state_dict()
    selected = {}
    skipped = []
    for key, value in source.items():
        if not key.startswith(ENCODER_PRETRAIN_PREFIXES):
            continue
        if key not in target:
            skipped.append(key)
            continue
        if tuple(target[key].shape) != tuple(value.shape):
            skipped.append(key)
            continue
        selected[key] = value
    missing, unexpected = model.load_state_dict(selected, strict=False)
    return {
        "loaded": sorted(selected),
        "skipped": sorted(skipped),
        "missing": sorted(missing),
        "unexpected": sorted(unexpected),
    }


def make_loader(
    dataset: RNAOmniDataset | Subset,
    tokenizer: RNAOmniTokenizer,
    config: Dict[str, Any],
    shuffle: bool,
) -> DataLoader:
    training_cfg = config["training"]
    data_cfg = config.get("data", {})
    num_workers = int(data_cfg.get("num_workers", 0))
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": bool(data_cfg.get("pin_memory", False)),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 2))
    collator = RNAOmniCollator(
        tokenizer=tokenizer,
        task_ratios=config["tasks"],
        pair_negative_ratio=int(training_cfg.get("pair_negative_ratio", training_cfg.get("pairRatio", 3))),
        seed=int(training_cfg.get("seed", 42)),
        ablation=config.get("ablation", {}),
        relation_mask=config.get("relation_mask", {}),
    )
    if config.get("data", {}).get("length_grouping", False):
        batch_sampler = LengthGroupedBatchSampler(
            get_dataset_lengths(dataset),
            batch_size=int(training_cfg["batch_size"]),
            bucket_size=int(config.get("data", {}).get("length_bucket_size", 50)),
            shuffle=shuffle,
            seed=int(training_cfg.get("seed", 42)),
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collator,
            **loader_kwargs,
        )
    return DataLoader(
        dataset,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=shuffle,
        collate_fn=collator,
        **loader_kwargs,
    )


def get_dataset_lengths(dataset: RNAOmniDataset | Subset) -> list[int]:
    if isinstance(dataset, Subset):
        return [int(dataset.dataset.samples[idx]["length"]) for idx in dataset.indices]
    return [int(sample["length"]) for sample in dataset.samples]


class LengthGroupedBatchSampler(Sampler[list[int]]):
    def __init__(self, lengths: Sequence[int], batch_size: int, bucket_size: int, shuffle: bool, seed: int) -> None:
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.bucket_size = max(1, bucket_size)
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(indices)
        bucketed = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start : start + self.bucket_size]
            bucket.sort(key=lambda idx: self.lengths[idx])
            if self.shuffle:
                rng.shuffle(bucket)
            bucketed.extend(bucket)
        batches = [bucketed[start : start + self.batch_size] for start in range(0, len(bucketed), self.batch_size)]
        if self.shuffle:
            rng.shuffle(batches)
        self.epoch += 1
        return iter(batches)

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def save_checkpoint(
    path: Path,
    model: RNAOmniDiffusion,
    tokenizer: RNAOmniTokenizer,
    config: Dict[str, Any],
    epoch: int,
    metrics: dict,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    global_step: int = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state": model.state_dict(),
        "tokenizer": tokenizer.to_dict(),
        "config": config,
        "epoch": epoch,
        "metrics": metrics,
        "global_step": global_step,
    }
    if optimizer is not None:
        state["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state"] = scheduler.state_dict()
    torch.save(state, path)


def load_checkpoint(ckpt_path: str | Path, device: torch.device) -> tuple[Dict[str, Any], RNAOmniTokenizer, dict]:
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Run `python main.py train --config config/candidate.yaml` first, "
            "or pass an existing --ckpt path."
        )
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    config = apply_ablation_settings(normalize_config(checkpoint["config"]))
    tokenizer = RNAOmniTokenizer.from_dict(checkpoint["tokenizer"])
    return config, tokenizer, checkpoint


def forward_model(model: RNAOmniDiffusion, batch: dict) -> dict:
    return model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        segment_ids=batch["segment_ids"],
        task_ids=batch["task_ids"],
        time_steps=batch["time_steps"],
        seq_positions=batch["seq_positions"],
        raw_seq=batch.get("raw_seq"),
    )


def estimate_loss_options(config: Dict[str, Any], dataset: RNAOmniDataset, tokenizer: RNAOmniTokenizer) -> dict:
    training = config["training"]
    token_weights = torch.ones(tokenizer.vocab_size, dtype=torch.float32)
    struct_counts = {token: 0 for token in tokenizer.structure_tokens}
    positive_pairs = 0
    sampled_negatives = 0
    for sample in dataset.samples:
        for char in sample["struct"]:
            if char in struct_counts:
                struct_counts[char] += 1
        pair_count = len(sample.get("pairs", []))
        positive_pairs += pair_count
        sampled_negatives += max(pair_count * int(training.get("pair_negative_ratio", 3)), int(sample["length"]) if pair_count == 0 else 0)

    if str(training.get("structure_positive_weight", "auto")).lower() == "auto":
        dot_count = max(1, struct_counts.get(".", 0))
        bracket_count = max(1, sum(count for token, count in struct_counts.items() if token != "."))
        bracket_weight = min(10.0, max(1.0, dot_count / bracket_count))
    else:
        bracket_weight = float(training.get("structure_positive_weight", 1.0))
    for token in ["(", ")", "[", "]", "{", "}"]:
        token_weights[tokenizer.token_id(token)] = float(bracket_weight)

    pair_weight_cfg = str(training.get("pair_positive_weight", "auto")).lower()
    if pair_weight_cfg == "auto":
        pair_pos_weight = float(sampled_negatives / max(1, positive_pairs)) if positive_pairs else 1.0
    else:
        pair_pos_weight = float(training.get("pair_positive_weight", 1.0))

    return {
        "lambda_pair": float(training.get("lambda_pair", 0.5)),
        "lambda_seq": float(training.get("lambda_seq", 1.0)),
        "lambda_struct": float(training.get("lambda_struct", 1.0)),
        "token_id_weights": token_weights,
        "pair_pos_weight": pair_pos_weight,
        "pair_options": {
            "pairWeight": training.get("pair_positive_weight", training.get("pairWeight", "auto")),
            "pairRatio": int(training.get("pair_negative_ratio", training.get("pairRatio", 3))),
            "pairUpper": bool(training.get("pairUpper", True)),
            "pairLoop": int(training.get("pairLoop", 3)),
            "pairDiag": bool(training.get("pairDiag", False)),
            "pairFloat": bool(training.get("pairFloat", True)),
            "sampleNegOnGpu": bool(training.get("sampleNegOnGpu", True)),
            "lambdaConflict": float(training.get("lambdaConflict", 0.0)),
            "conflictMargin": float(training.get("conflictMargin", 1.0)),
            "conflictUseProb": bool(training.get("conflictUseProb", True)),
        },
        "use_pair_loss": bool(config.get("ablation", {}).get("use_pair_loss", True))
        and bool(config.get("ablation", {}).get("use_pair_head", True)),
        "structure_bracket_weight": float(bracket_weight),
        "pair_loss_policy": config.get("pair_loss_policy", {}),
        "local_rank_loss": config.get("local_rank_loss", {}),
    }


def loss_from_batch(outputs: dict, batch: dict, loss_options: dict) -> dict:
    result = compute_omni_loss(
        outputs,
        batch,
        lambda_pair=loss_options["lambda_pair"],
        lambda_seq=loss_options["lambda_seq"],
        lambda_struct=loss_options["lambda_struct"],
        token_id_weights=loss_options["token_id_weights"],
        pair_pos_weight=loss_options["pair_pos_weight"],
        use_pair_loss=loss_options["use_pair_loss"],
        pair_options=loss_options.get("pair_options"),
        pair_loss_policy=loss_options.get("pair_loss_policy"),
    )
    # Local pair ranking loss (optional)
    rank_cfg = loss_options.get("local_rank_loss", {})
    rank_loss = result["loss"].new_zeros(())
    if rank_cfg.get("enabled", False):
        pair_logits = outputs.get("pair_logits")
        pair_labels = batch.get("pair_labels")
        raw_seq = batch.get("raw_seq")
        global_step = loss_options.get("global_step", 0)
        warmup = int(rank_cfg.get("warmup_steps", 100))
        if pair_logits is not None and pair_labels is not None and raw_seq is not None:
            # Warmup: linearly ramp weight from 0 to full
            weight = float(rank_cfg.get("weight", 0.1))
            if warmup > 0 and global_step < warmup:
                weight = weight * (global_step + 1) / warmup
            if weight > 0:
                rank_loss = local_rank_loss(
                    pair_logits.float(),
                    pair_labels.to(pair_logits.device).float(),
                    raw_seq,
                    margin=float(rank_cfg.get("margin", 1.0)),
                    negatives_per_positive=int(rank_cfg.get("negatives_per_positive", 4)),
                )
                result["loss"] = result["loss"] + weight * rank_loss
    result["rank_loss"] = rank_loss.detach()
    return result


def update_running(total: dict, loss_dict: dict, batch_size: int) -> None:
    total["samples"] += batch_size
    for key in ("loss", "token_loss", "pair_loss", "conflict_loss", "rank_loss"):
        total[key] += float(loss_dict[key].detach().cpu()) * batch_size
    # preference metrics
    for key in ("pref_loss", "pref_covered"):
        val = loss_dict.get(key)
        if val is not None:
            if torch.is_tensor(val):
                total[key] += float(val.detach().cpu())
            elif key == "pref_covered":
                total[key] += int(val)
    for key in (
        "pos",
        "neg",
        "weight",
        "posLogit",
        "negLogit",
        "gap",
        "posProb",
        "negProb",
        "rankAcc",
        "pos_pair_count",
        "neg_pair_count",
        "pair_positive_weight_used",
        "positive_pair_logit_mean",
        "negative_pair_logit_mean",
        "pair_logit_gap",
        "positive_pair_prob_mean",
        "negative_pair_prob_mean",
        "pair_ranking_accuracy_sampled",
        "lambdaConflict",
        "mean_row_pair_prob_sum",
        "max_row_pair_prob_sum",
    ):
        value = loss_dict.get(key)
        if value is None:
            continue
        if torch.is_tensor(value):
            value = float(value.detach().cpu())
        total.setdefault(f"{key}_sum", 0.0)
        total.setdefault(f"{key}_count", 0)
        total[f"{key}_sum"] += float(value)
        total[f"{key}_count"] += 1


def averages(total: dict, prefix: str) -> dict:
    denom = max(1, total["samples"])
    result = {
        f"{prefix}_loss": total["loss"] / denom,
        f"{prefix}_token_loss": total["token_loss"] / denom,
        f"{prefix}_pair_loss": total["pair_loss"] / denom,
        f"{prefix}_conflict_loss": total["conflict_loss"] / denom,
        f"{prefix}_rank_loss": total["rank_loss"] / denom,
    }
    if total.get("pref_covered", 0) > 0:
        result[f"{prefix}_pref_loss"] = total.get("pref_loss", 0.0) / total["pref_covered"]
        result[f"{prefix}_pref_coverage"] = total["pref_covered"] / max(1, total["samples"])
    else:
        result[f"{prefix}_pref_loss"] = 0.0
        result[f"{prefix}_pref_coverage"] = 0.0
    for key in (
        "pos",
        "neg",
        "weight",
        "posLogit",
        "negLogit",
        "gap",
        "posProb",
        "negProb",
        "rankAcc",
        "pos_pair_count",
        "neg_pair_count",
        "pair_positive_weight_used",
        "positive_pair_logit_mean",
        "negative_pair_logit_mean",
        "pair_logit_gap",
        "positive_pair_prob_mean",
        "negative_pair_prob_mean",
        "pair_ranking_accuracy_sampled",
        "lambdaConflict",
        "mean_row_pair_prob_sum",
        "max_row_pair_prob_sum",
    ):
        count = total.get(f"{key}_count", 0)
        if count:
            result[key] = total[f"{key}_sum"] / count
    return result


def decode_batch_tokens(tokenizer: RNAOmniTokenizer, logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=-1)


def collect_pair_diagnostics(outputs: dict, batch: dict, diagnostics: dict) -> None:
    pair_logits = outputs.get("pair_logits")
    if pair_logits is None:
        return
    pair_logits = pair_logits.detach().float().cpu()
    pair_labels = batch["pair_labels"].detach().float().cpu()
    pair_mask = batch["pair_mask"].detach().cpu()
    lengths = batch["lengths"].detach().cpu().tolist()

    if pair_mask.any():
        selected_logits = pair_logits[pair_mask]
        selected_labels = pair_labels[pair_mask]
        pos = selected_logits[selected_labels > 0.5]
        neg = selected_logits[selected_labels <= 0.5]
        if pos.numel():
            diagnostics["positive_pair_logit_sum"] += float(pos.sum())
            diagnostics["positive_pair_logit_count"] += int(pos.numel())
        if neg.numel():
            diagnostics["negative_pair_logit_sum"] += float(neg.sum())
            diagnostics["negative_pair_logit_count"] += int(neg.numel())

    for batch_idx, length in enumerate(lengths):
        if length < 2:
            continue
        logits = pair_logits[batch_idx, :length, :length]
        upper = torch.triu(torch.ones((length, length), dtype=torch.bool), diagonal=1)
        probs = torch.sigmoid(logits[upper])
        if probs.numel() == 0:
            continue
        diagnostics["pair_prob_sum"] += float(probs.sum())
        diagnostics["pair_prob_count"] += int(probs.numel())
        topk = min(10, probs.numel())
        diagnostics["pair_prob_topk_sum"] += float(torch.topk(probs, k=topk).values.mean())
        diagnostics["pair_prob_topk_count"] += 1


def finalize_pair_diagnostics(diagnostics: dict) -> dict:
    return {
        "positive_pair_logit_mean": diagnostics["positive_pair_logit_sum"] / max(1, diagnostics["positive_pair_logit_count"]),
        "negative_pair_logit_mean": diagnostics["negative_pair_logit_sum"] / max(1, diagnostics["negative_pair_logit_count"]),
        "pair_prob_mean": diagnostics["pair_prob_sum"] / max(1, diagnostics["pair_prob_count"]),
        "pair_prob_topk_mean": diagnostics["pair_prob_topk_sum"] / max(1, diagnostics["pair_prob_topk_count"]),
    }


def evaluate_model(
    model: RNAOmniDiffusion,
    loader: DataLoader,
    samples: Sequence[dict],
    tokenizer: RNAOmniTokenizer,
    config: Dict[str, Any],
    device: torch.device,
    loss_options: dict,
    max_batches: int | None = None,
    decode_structures: bool = True,
) -> dict:
    model.eval()
    totals = {"samples": 0, "loss": 0.0, "token_loss": 0.0, "pair_loss": 0.0, "conflict_loss": 0.0, "rank_loss": 0.0}
    token_correct = 0
    token_count = 0
    pair_diag = {
        "positive_pair_logit_sum": 0.0,
        "positive_pair_logit_count": 0,
        "negative_pair_logit_sum": 0.0,
        "negative_pair_logit_count": 0,
        "pair_prob_sum": 0.0,
        "pair_prob_count": 0,
        "pair_prob_topk_sum": 0.0,
        "pair_prob_topk_count": 0,
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = move_batch_to_device(batch, device)
            outputs = forward_model(model, batch)
            loss_dict = loss_from_batch(outputs, batch, loss_options)
            update_running(totals, loss_dict, int(batch["input_ids"].size(0)))

            labels = batch["labels"]
            supervised = labels != -100
            if supervised.any():
                preds = decode_batch_tokens(tokenizer, outputs["token_logits"])
                token_correct += int((preds[supervised] == labels[supervised]).sum().detach().cpu())
                token_count += int(supervised.sum().detach().cpu())
            collect_pair_diagnostics(outputs, batch, pair_diag)

            if max_batches is not None and batch_idx + 1 >= max_batches:
                break

    metrics = averages(totals, "val")
    metrics["val_token_acc"] = token_correct / max(1, token_count)
    if decode_structures:
        seqs = [sample["seq"] for sample in samples]
        true_structs = [sample["struct"] for sample in samples]
        pred_structs = [
            generate_structure_seq2struct(model, tokenizer, seq, config["decoding"], device)
            for seq in seqs
        ]
        struct_metrics = evaluate_structures(
            pred_structs,
            true_structs,
            seqs,
            allow_wobble=bool(config.get("decoding", {}).get("allow_wobble", True)),
        )
        metrics.update({f"val_{key}": value for key, value in struct_metrics.items()})
    metrics.update(finalize_pair_diagnostics(pair_diag))
    return metrics


def print_pair_batch_debug(batch: dict) -> None:
    print("Pair batch debug:")
    print(f"  raw_seq[0]: {batch['raw_seq'][0]}")
    print(f"  raw_struct[0]: {batch['raw_struct'][0]}")
    print(f"  parsed pairs[0]: {batch['raw_pairs'][0]}")
    print(f"  positive pairs: {int(batch['pair_positive_counts'].sum())}")
    print(f"  sampled negative pairs: {int(batch['pair_negative_counts'].sum())}")
    print(f"  seq_positions count[0]: {int((batch['seq_positions'][0] >= 0).sum())}")
    print(f"  struct_positions count[0]: {int((batch['struct_positions'][0] >= 0).sum())}")
    print(f"  input_ids shape: {tuple(batch['input_ids'].shape)}")
    print(f"  pair_labels shape: {tuple(batch['pair_labels'].shape)}")


def format_epoch_metrics(metrics: dict) -> str:
    keys = [
        "epoch",
        "train_loss",
        "train_token_loss",
        "train_pair_loss",
        "train_conflict_loss",
        "val_loss",
        "val_token_acc",
        "val_pair_precision",
        "val_pair_recall",
        "val_pair_f1",
        "val_valid_structure_rate",
        "val_canonical_pair_ratio",
        "val_all_dot_ratio",
        "learning_rate",
        "gap",
        "rankAcc",
        "mean_row_pair_prob_sum",
        "max_row_pair_prob_sum",
        "epoch_time",
    ]
    parts = []
    for key in keys:
        if key not in metrics:
            continue
        value = metrics[key]
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def warn_if_collapsed(metrics: dict) -> None:
    if float(metrics.get("val_all_dot_ratio", 0.0)) > 0.8:
        print(
            "Warning: model may collapse to all-dot structures. Check pair loss weight, pair labels, "
            "Nussinov threshold, or positive/negative pair imbalance."
        )


def train_model(
    config: Dict[str, Any],
    max_steps: int | None = None,
    train_subset: int | None = None,
    device_name: str | None = "auto",
) -> dict:
    set_seed(int(config["training"].get("seed", 42)))
    ensure_dataset_paths(config, create_if_missing=True)
    train_dataset, val_dataset, tokenizer = build_datasets_and_tokenizer(config)
    loss_options = estimate_loss_options(config, train_dataset, tokenizer)
    if train_subset is not None:
        train_dataset_for_loader = Subset(train_dataset, list(range(min(train_subset, len(train_dataset)))))
    else:
        train_dataset_for_loader = train_dataset

    device = resolve_device(device_name)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    model = build_model(config, tokenizer, device)
    pretrain_load_report = None
    init_from_pretrain = config["training"].get("init_from_pretrain")
    if init_from_pretrain:
        if bool(config["training"].get("load_encoder_only", False)):
            report = load_encoder_only_pretrain(model, init_from_pretrain, device)
            pretrain_load_report = report
            print(f"Loaded encoder-only pretrain from {init_from_pretrain}: {len(report['loaded'])} tensors.")
        else:
            _, _, checkpoint = load_checkpoint(init_from_pretrain, device)
            model.load_state_dict(checkpoint["model_state"], strict=False)
            pretrain_load_report = {"loaded": list(checkpoint["model_state"]), "skipped": []}
            print(f"Loaded pretrain weights from {init_from_pretrain}.")
    train_loader = make_loader(train_dataset_for_loader, tokenizer, config, shuffle=True)
    val_loader = make_loader(val_dataset, tokenizer, config, shuffle=False)

    if config.get("debug", {}).get("check_pair_batch", False):
        print_pair_batch_debug(next(iter(train_loader)))

    optimizer = AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"].get("weight_decay", 0.01)),
    )
    warmup_steps = int(config["training"].get("warmup_steps", 0))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, (step + 1) / warmup_steps)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    start_epoch = 1
    global_step = 0
    resume_path = config["training"].get("resume_from")
    if resume_path:
        _, _, checkpoint = load_checkpoint(resume_path, device)
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        print(f"Resumed training from {resume_path} at epoch {start_epoch}.")
    use_amp = bool(config["training"].get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config_used.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    log_path = output_dir / "trainlog.jsonl"
    old_log_path = output_dir / "train_log.jsonl"
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    if not resume_path:
        if log_path.exists():
            log_path.unlink()
        if old_log_path.exists():
            old_log_path.unlink()
    best_score = -math.inf
    best_epoch = 0
    best_raw_value = 0.0
    best_metric_name = str(config["training"].get("save_best_by", "val_pair_f1"))
    history = []
    log_every = int(config["training"].get("log_every", 20))
    patience = int(config["training"].get("early_stopping_patience", 10))
    min_delta = float(config["training"].get("min_delta", 0.0001))
    stale_epochs = 0
    device_name = device.type
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    train_startup(config, device_name, gpu_name)

    for epoch in range(start_epoch, int(config["training"]["epochs"]) + 1):
        epoch_start = time.time()
        model.train()
        train_totals = {"samples": 0, "loss": 0.0, "token_loss": 0.0, "pair_loss": 0.0, "conflict_loss": 0.0, "rank_loss": 0.0, "pref_loss": 0.0, "pref_covered": 0}
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = forward_model(model, batch)
                loss_options["global_step"] = global_step
                loss_dict = loss_from_batch(outputs, batch, loss_options)
                loss = loss_dict["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"].get("grad_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            update_running(train_totals, loss_dict, int(batch["input_ids"].size(0)))
            if global_step == 1 or global_step % log_every == 0:
                print(
                    f"step={global_step} epoch={epoch} "
                    f"loss={float(loss.detach().cpu()):.4f} "
                    f"token={float(loss_dict['token_loss'].cpu()):.4f} "
                    f"pair={float(loss_dict['pair_loss'].cpu()):.4f} "
                    f"rank={float(loss_dict.get('rank_loss', loss.new_zeros(()).detach()).cpu()):.4f} "
                    f"conflict={float(loss_dict['conflict_loss'].cpu()):.4f}"
                )
            if max_steps is not None and global_step >= max_steps:
                break

        val_eval_samples = val_dataset.samples
        val_decode_samples = int(config["training"].get("val_decode_samples", len(val_eval_samples)))
        if val_decode_samples > 0:
            val_eval_samples = val_eval_samples[: min(val_decode_samples, len(val_eval_samples))]
        if max_steps is not None:
            val_eval_samples = val_eval_samples[: min(32, len(val_eval_samples))]
        val_max_batches = config["training"].get("val_max_batches")
        if val_max_batches is not None:
            val_max_batches = int(val_max_batches)
        train_decode_structures = bool(config["training"].get("train_decode_structures", True))
        val_metrics = evaluate_model(
            model,
            val_loader,
            val_eval_samples,
            tokenizer,
            config,
            device,
            loss_options,
            max_batches=2 if max_steps else val_max_batches,
            decode_structures=train_decode_structures,
        )
        epoch_metrics = {
            "epoch": epoch,
            "step": global_step,
            **averages(train_totals, "train"),
            **val_metrics,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "device": device.type,
            "cuda": bool(device.type == "cuda"),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
            "amp": bool(use_amp),
            "memory": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
            "epoch_time": time.time() - epoch_start,
            "pairrefine": bool(config.get("model", {}).get("pairrefine", False)),
            "conflict_loss": averages(train_totals, "train").get("train_conflict_loss", 0.0),
            "lambdaConflict": float(config["training"].get("lambdaConflict", 0.0)),
            "init_from_pretrain": str(init_from_pretrain or ""),
            "load_encoder_only": bool(config["training"].get("load_encoder_only", False)),
            "pretrain_loaded_keys": len(pretrain_load_report["loaded"]) if pretrain_load_report else 0,
            "pretrain_skipped_keys": len(pretrain_load_report["skipped"]) if pretrain_load_report else 0,
        }
        history.append(epoch_metrics)
        print(epoch_line(epoch_metrics, epoch, int(config["training"]["epochs"])))
        warn_if_collapsed(epoch_metrics)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_metrics) + "\n")
        save_checkpoint(last_path, model, tokenizer, config, epoch, epoch_metrics, optimizer, scheduler, global_step)
        checkpoint_saved(str(last_path), is_best=False)
        raw_score = float(epoch_metrics.get(best_metric_name, math.inf if best_metric_name.endswith("loss") else -math.inf))
        score = -raw_score if best_metric_name.endswith("loss") else raw_score
        if score > best_score + min_delta:
            best_score = score
            best_epoch = epoch
            best_raw_value = raw_score
            stale_epochs = 0
            save_checkpoint(best_path, model, tokenizer, config, epoch, epoch_metrics, optimizer, scheduler, global_step)
            checkpoint_saved(str(best_path), is_best=True, metric_name=best_metric_name, metric_value=raw_score)
        else:
            stale_epochs += 1
            if max_steps is None and stale_epochs >= patience:
                print()
                early_stopping_summary(best_epoch, best_metric_name, best_raw_value, patience)
                break
        if max_steps is not None and global_step >= max_steps:
            break

    return {
        "model": model,
        "tokenizer": tokenizer,
        "config": config,
        "best_path": best_path,
        "last_path": last_path,
        "history": history,
    }


def run_train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    config["training"]["_config_path"] = args.config
    if args.resume:
        config["training"]["resume_from"] = args.resume
    result = train_model(config, max_steps=args.max_steps, train_subset=args.train_subset, device_name=args.device)
    training_complete(
        best_path=str(result["best_path"]),
        last_path=str(result["last_path"]),
        log_path=str(Path(config["training"]["output_dir"]) / "trainlog.jsonl"),
        output_dir=config["training"]["output_dir"],
    )


def run_eval(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    try:
        ckpt_config, tokenizer, checkpoint = load_checkpoint(args.ckpt, device)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
    user_config = load_config(args.config)
    config = dict(ckpt_config)
    config["data"] = user_config.get("data", config.get("data", {}))
    config["decoding"] = deep_update(config.get("decoding", {}), user_config.get("decoding", {}))
    val_dataset = RNAOmniDataset(config["data"]["val_jsonl"], max_length=int(config["data"]["max_length"]))
    model = build_model(config, tokenizer, device)
    try:
        model.load_state_dict(checkpoint["model_state"])
    except RuntimeError as exc:
        raise SystemExit(
            "Checkpoint is not compatible with the current model structure. "
            "The pair head was changed to an MLP; retrain the checkpoint or use a matching config."
        ) from exc
    loader = make_loader(val_dataset, tokenizer, config, shuffle=False)
    loss_options = estimate_loss_options(config, val_dataset, tokenizer)
    metrics = evaluate_model(model, loader, val_dataset.samples, tokenizer, config, device, loss_options)
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, float):
            print(f"{key}={value:.6f}")
        else:
            print(f"{key}={value}")
    warn_if_collapsed(metrics)


def run_infer(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    try:
        config, tokenizer, checkpoint = load_checkpoint(args.ckpt, device)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
    model = build_model(config, tokenizer, device)
    try:
        model.load_state_dict(checkpoint["model_state"])
    except RuntimeError as exc:
        raise SystemExit(
            "Checkpoint is not compatible with the current model structure. "
            "The pair head was changed to an MLP; retrain the checkpoint or use a matching config."
        ) from exc
    model.eval()

    inference_header(args.task, args.ckpt, args.config, args.device if args.device != "auto" else str(device))

    if args.task == "seq2struct":
        if not args.seq:
            raise ValueError("--seq is required for seq2struct inference.")
        struct = generate_structure_seq2struct(model, tokenizer, args.seq, config["decoding"], device)
        inference_result_seq2struct(args.seq, struct, len(args.seq))
    elif args.task == "invfold":
        if not args.struct:
            raise ValueError("--struct is required for invfold inference.")
        seq = generate_sequence_invfold(model, tokenizer, args.struct, config["decoding"], device)
        inference_result_invfold(args.struct, seq, len(args.struct))
    else:
        raise ValueError("Minimal CLI inference supports --task seq2struct or --task invfold.")


def run_smoke(args: argparse.Namespace) -> None:
    config = load_config("config/candidate.yaml")
    config = deep_update(
        config,
        {
            "data": {"max_length": 64, "num_workers": 0},
            "model": {
                "hidden_size": 64,
                "num_layers": 2,
                "num_heads": 4,
                "dropout": 0.1,
                "max_position_embeddings": 256,
            },
            "training": {
                "batch_size": 2,
                "epochs": 1,
                "lr": 0.001,
                "warmup_steps": 1,
                "amp": False,
                "output_dir": "outputs/smoke",
                "seed": 42,
                "log_every": 1,
            },
            "decoding": {"num_steps": 4, "use_nussinov": True, "decode_source": "pair"},
        },
    )
    create_tiny_jsonl_dataset(config, overwrite=False)
    result = train_model(config, max_steps=2, device_name="auto")
    model = result["model"]
    tokenizer = result["tokenizer"]
    model.eval()
    device = next(model.parameters()).device
    seq = synthetic_samples()[0]["seq"]
    pred_struct = generate_structure_seq2struct(model, tokenizer, seq, config["decoding"], device)
    true_struct = synthetic_samples()[0]["struct"]
    print(f"smoke_losses={[round(item['train_loss'], 4) for item in result['history']]}")
    print(f"seq={seq}")
    print(f"pred_struct={pred_struct}")
    print(f"bp_f1_vs_toy={base_pair_f1(pred_struct, true_struct):.4f}")
    print("smoke_ok")

