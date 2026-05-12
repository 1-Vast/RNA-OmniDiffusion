"""RNA-OmniPrefold experiment management helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PLAN = """# RNA-OmniPrefold Experiment Plan

## Core Comparisons

| Experiment | Config |
|---|---|
| Supervised baseline | config/candidate.yaml |
| Sequence pretraining | config/seq_pretrain.yaml |
| Fine-tune from sequence pretraining | config/candidate_from_seq_pretrain.yaml |

## Required Controls

- strict Nussinov decoding
- no token-only final metric
- no pseudo-structure labels
- no semantic-token conditioning
"""


def run_plan(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(PLAN, encoding="utf-8")
    print(f"wrote {out}")


def run_manifest(args: argparse.Namespace) -> None:
    manifest = {
        "mainline": "RNA-OmniPrefold",
        "configs": [
            "config/candidate.yaml",
            "config/seq_pretrain.yaml",
            "config/candidate_from_seq_pretrain.yaml",
        ],
        "default_decode": "strict_nussinov",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RNA-OmniPrefold experiment manager")
    sub = parser.add_subparsers(dest="cmd", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--out", default="outputs/experiments/plan.md")
    plan.set_defaults(func=run_plan)
    manifest = sub.add_parser("manifest")
    manifest.add_argument("--out", default="outputs/experiments/manifest.json")
    manifest.set_defaults(func=run_manifest)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
