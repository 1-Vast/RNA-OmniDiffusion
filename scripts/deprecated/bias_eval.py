"""Pair-specific bias table evaluation script.

Runs 1000-step training with bias table enabled on seed 42.
Reports: F1, precision, recall, valid_rate, overpair_rate,
         pair_count_bias, rank_accuracy.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.training import load_config, train_model


def _summarize(history: list[dict]) -> dict:
    if not history:
        return {}
    best = max(history, key=lambda r: float(r.get("val_pair_f1", 0.0)))
    avg_pred = best.get("val_avg_pred_pair_count", 0.0)
    avg_true = best.get("val_avg_true_pair_count", 0.0)
    return {
        "val_pair_f1": float(best.get("val_pair_f1", 0.0)),
        "val_pair_precision": float(best.get("val_pair_precision", 0.0)),
        "val_pair_recall": float(best.get("val_pair_recall", 0.0)),
        "val_valid_structure_rate": float(best.get("val_valid_structure_rate", 0.0)),
        "val_all_dot_ratio": float(best.get("val_all_dot_ratio", 0.0)),
        "overpair_rate": max(0.0, avg_pred / max(1e-8, avg_true) - 1.0),
        "pair_count_bias": avg_pred - avg_true,
        "rank_accuracy": float(best.get("rankAcc", 0.0)) if best.get("rankAcc") is not None else -1.0,
    }


def main() -> None:
    config_path = "config/bias_table.yaml"
    print(f"Loading config: {config_path}")
    config = load_config(config_path)

    # Override output dir and seed for eval
    config["training"]["output_dir"] = "outputs/bias_eval"
    config["training"]["seed"] = 42
    config["training"]["train_decode_structures"] = True
    config["training"]["val_decode_samples"] = 64

    print(f"Running bias table training (max_steps=1000, seed=42)...")
    result = train_model(config, max_steps=1000, device_name="auto")

    metrics = _summarize(result["history"])
    print()
    print("=== Bias Table Eval Results ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print("===============================")


if __name__ == "__main__":
    main()
