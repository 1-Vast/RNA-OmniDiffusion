from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import load_config, run_smoke, train_model


def smoke(args: argparse.Namespace) -> None:
    run_smoke(args)


def overfit(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    result = train_model(config, max_steps=args.steps, train_subset=args.samples, device_name=args.device)
    first = result["history"][0]["train_loss"] if result["history"] else 0.0
    last = result["history"][-1]["train_loss"] if result["history"] else 0.0
    print(f"initial_loss={first:.4f}")
    print(f"final_loss={last:.4f}")


def real(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    train_model(config, max_steps=args.steps, train_subset=args.train, device_name=args.device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe compact RNA-OmniPrefold behavior.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("smoke")
    s.set_defaults(func=smoke)
    o = sub.add_parser("overfit")
    o.add_argument("--config", default="config/candidate.yaml")
    o.add_argument("--samples", type=int, default=16)
    o.add_argument("--steps", type=int, default=200)
    o.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    o.set_defaults(func=overfit)
    r = sub.add_parser("real")
    r.add_argument("--config", default="config/candidate.yaml")
    r.add_argument("--train", type=int, default=128)
    r.add_argument("--val", type=int, default=32)
    r.add_argument("--steps", type=int, default=100)
    r.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    r.set_defaults(func=real)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

