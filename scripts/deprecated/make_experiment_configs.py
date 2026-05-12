"""Generate compact RNA-OmniPrefold sequence-pretraining configs."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def make_pretrain(train_jsonl: str, val_jsonl: str, seed: int, epochs: int, output_dir: str) -> dict:
    return {
        "data": {
            "train_jsonl": train_jsonl,
            "val_jsonl": val_jsonl,
            "allow_unlabeled": True,
            "max_length": 512,
            "num_workers": 4,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 2,
            "length_grouping": True,
            "length_bucket_size": 50,
        },
        "tasks": {"seq_denoise": 1.0},
        "model": {
            "hidden_size": 512,
            "num_layers": 8,
            "num_heads": 8,
            "dropout": 0.1,
            "max_position_embeddings": 2048,
        },
        "training": {
            "batch_size": 64,
            "epochs": epochs,
            "lr": 0.0001,
            "weight_decay": 0.01,
            "warmup_steps": 200,
            "lambda_seq": 1.0,
            "lambda_struct": 0.0,
            "lambda_pair": 0.0,
            "grad_clip": 1.0,
            "amp": True,
            "seed": seed,
            "output_dir": output_dir,
            "save_best_by": "val_loss",
            "early_stopping_patience": 3,
            "min_delta": 0.0001,
            "log_every": 50,
            "train_decode_structures": False,
            "val_decode_samples": 0,
            "val_max_batches": 4,
        },
        "decoding": {"use_nussinov": False, "decode_source": "token"},
        "ablation": {
            "use_pair_head": False,
            "use_pair_loss": False,
            "use_nussinov": False,
            "use_pair_aware_masking": False,
            "use_motif_span_masking": False,
            "use_motif_condition": False,
            "use_family_condition": False,
        },
    }


def make_finetune(pretrain_dir: str, output_dir: str, seed: int, epochs: int = 20) -> dict:
    return {
        "data": {
            "train_jsonl": "dataset/archive/train.jsonl",
            "val_jsonl": "dataset/archive/val.jsonl",
            "test_jsonl": "dataset/archive/test.jsonl",
            "max_length": 512,
            "num_workers": 8,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 4,
            "length_grouping": True,
            "length_bucket_size": 50,
        },
        "tasks": {
            "seq2struct": 0.35,
            "invfold": 0.25,
            "inpaint": 0.25,
            "motif_control": 0.15,
        },
        "model": {
            "hidden_size": 512,
            "num_layers": 8,
            "num_heads": 8,
            "dropout": 0.1,
            "max_position_embeddings": 2048,
            "distbias": True,
            "pairrefine": True,
        },
        "training": {
            "batch_size": 24,
            "epochs": epochs,
            "lr": 0.0001,
            "weight_decay": 0.01,
            "warmup_steps": 500,
            "lambda_pair": 5.0,
            "lambda_struct": 1.0,
            "lambda_seq": 1.0,
            "pair_negative_ratio": 5,
            "pair_positive_weight": 10.0,
            "structure_positive_weight": 1.0,
            "grad_clip": 1.0,
            "amp": True,
            "seed": seed,
            "output_dir": output_dir,
            "init_from_pretrain": f"{pretrain_dir}/best.pt",
            "load_encoder_only": True,
            "save_best_by": "val_loss",
            "early_stopping_patience": 10,
            "min_delta": 0.0001,
            "log_every": 20,
            "train_decode_structures": False,
            "val_decode_samples": 16,
            "val_max_batches": 4,
        },
        "decoding": {
            "num_steps": 32,
            "min_loop_length": 3,
            "use_nussinov": True,
            "allow_wobble": True,
            "decode_source": "pair",
            "pair_threshold": 0.25,
            "nussinov_gamma": 2.0,
            "topk_pairs": 20,
        },
        "ablation": {
            "use_pair_head": True,
            "use_pair_loss": True,
            "use_nussinov": True,
            "use_pair_aware_masking": False,
            "use_motif_span_masking": False,
            "use_motif_condition": False,
            "use_family_condition": False,
        },
        "baselines": {
            "run_all_dot": True,
            "run_random_pair": True,
            "run_rnafold": False,
            "rnafold_bin": "RNAfold",
        },
        "debug": {"check_pair_batch": False},
    }


def write_yaml(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RNA-OmniPrefold experiment configs")
    parser.add_argument("--name", required=True, help="Experiment name, e.g. rfam50k")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs_pretrain", type=int, default=5)
    parser.add_argument("--epochs_finetune", type=int, default=20)
    parser.add_argument("--output_prefix", default="outputs")
    args = parser.parse_args()

    pretrain_name = f"seq_pretrain_{args.name}_seed{args.seed}"
    pretrain_config = make_pretrain(
        args.train_jsonl,
        args.val_jsonl,
        args.seed,
        args.epochs_pretrain,
        f"{args.output_prefix}/{pretrain_name}",
    )
    pretrain_path = Path(f"config/{pretrain_name}.yaml")
    write_yaml(pretrain_path, pretrain_config)
    print(f"Wrote {pretrain_path}")

    finetune_name = f"candidate_from_seq_pretrain_{args.name}_seed{args.seed}"
    finetune_config = make_finetune(
        f"{args.output_prefix}/{pretrain_name}",
        f"{args.output_prefix}/{finetune_name}",
        args.seed,
        args.epochs_finetune,
    )
    finetune_path = Path(f"config/{finetune_name}.yaml")
    write_yaml(finetune_path, finetune_config)
    print(f"Wrote {finetune_path}")


if __name__ == "__main__":
    main()
