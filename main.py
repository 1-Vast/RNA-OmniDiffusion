"""RNA-OmniPrefold CLI -- research entry window.

Relational Masked Diffusion for Constraint-Guided RNA Folding:
  Core        Transformer student encoder over RNA tokens.
  Token heads Sequence / structure / general prediction heads.
  Pair head   Pair-relation logits over sequence positions.
  Pair refine Lightweight 2D refinement over the pair-relation field.
  Decoding    Strict Nussinov projection for valid non-crossing structures.

Modes: train | eval | infer | smoke | params | overview.
Torch is lazy-imported per subcommand so overview / params stay lightweight.
"""

from __future__ import annotations

import argparse
import json
from typing import Any


MODE_REGISTRY: dict[str, dict[str, str]] = {
    "overview": {
        "purpose": "Print the RNA-OmniPrefold framework map.",
        "input": "none.",
        "output": "human-readable framework description.",
        "typical": "python main.py overview",
    },
    "models": {
        "purpose": "Alias for overview.",
        "input": "none.",
        "output": "same as overview.",
        "typical": "python main.py models",
    },
    "train": {
        "purpose": "Train RNA-OmniPrefold from a YAML config.",
        "input": "config YAML + dataset JSONL.",
        "output": "trainlog.jsonl, best.pt, last.pt.",
        "typical": "python main.py train --config config/candidate.yaml --device cuda",
    },
    "eval": {
        "purpose": "Evaluate a checkpoint on the validation split.",
        "input": "config YAML + checkpoint.",
        "output": "validation metrics (loss, pair F1, valid rate, etc.).",
        "typical": "python main.py eval --config config/candidate.yaml --ckpt outputs/candidate/best.pt --device cuda",
    },
    "infer": {
        "purpose": "Run single-sample seq2struct or inverse folding inference.",
        "input": "sequence or dot-bracket structure + checkpoint.",
        "output": "predicted structure or sequence.",
        "typical": "python main.py infer --config config/candidate.yaml --ckpt outputs/candidate/best.pt --task seq2struct --seq GCAUAGC",
    },
    "smoke": {
        "purpose": "Run a tiny CPU/GPU sanity test.",
        "input": "synthetic tiny dataset generated locally.",
        "output": "smoke_ok + basic structure prediction check.",
        "typical": "python main.py smoke",
    },
    "params": {
        "purpose": "Inspect tunable model, training, and decoding parameters.",
        "input": "YAML config (optional).",
        "output": "grouped parameter table.",
        "typical": "python main.py params --config config/candidate.yaml",
    },
}


PARAM_REGISTRY: dict[str, list[tuple[str, str, str]]] = {
    "training": [
        ("output_dir", "Training output directory", "change per experiment"),
        ("epochs", "Number of training epochs", "increase for full training"),
        ("batch_size", "Batch size", "reduce if CUDA OOM"),
        ("lr", "Learning rate", "tune carefully, e.g. 5e-5 to 2e-4"),
        ("seed", "Random seed", "fixed for reproducibility"),
        ("lambda_pair", "Base-pair loss weight", "increase if pair recall is low"),
        ("lambda_struct", "Structure CE loss weight", "usually 1.0"),
        ("lambda_seq", "Sequence CE loss weight", "usually 1.0"),
        ("pair_positive_weight", "Positive-pair BCE weight", "increase if positives are underfit"),
        ("pair_negative_ratio", "Sampled negative-pair ratio", "increase for stronger negative contrast"),
        ("lambdaConflict", "Conflict loss weight", "0.0 for candidate"),
        ("conflictMargin", "Conflict margin", "1.0 default"),
        ("amp", "Mixed precision training (CUDA only)", "use true on CUDA"),
        ("grad_clip", "Gradient clipping norm", "keep stable"),
    ],
    "model": [
        ("hidden_size", "Transformer hidden size", "architecture-level; avoid changing candidate"),
        ("num_layers", "Transformer depth", "architecture-level; avoid changing candidate"),
        ("dropout", "Dropout rate", "increase if overfitting"),
        ("pairhead", "Pair prediction head type", "mlp / bilinear / pairmlp"),
        ("pairhidden", "Pair head hidden size", "default matches hidden_size"),
        ("pairdrop", "Pair head dropout", "default matches dropout"),
        ("distbias", "Distance bias for pair logits", "usually true for candidate"),
        ("pairrefine", "2D pair-logit refinement", "candidate uses true"),
        ("pairrefinechannels", "Refine conv channels", "small values only"),
        ("pairrefineblocks", "Number of refine blocks", "small values only"),
    ],
    "decoding": [
        ("decode_source", "Decoding source", "pair / token / hybrid; candidate uses pair"),
        ("use_nussinov", "Strict non-crossing DP decoding", "true for valid structures"),
        ("pair_threshold", "Pair probability threshold", "lower for recall, higher for precision"),
        ("nussinov_gamma", "Nussinov score scale", "higher makes pair scores sharper"),
        ("method", "Decode method override", "leave unset for default"),
        ("min_loop_length", "Minimum pair index distance", "hairpin loop constraint"),
    ],
    "ablation": [
        ("use_pair_head", "Enable pair head", "keep true for candidate"),
        ("use_pair_loss", "Enable pair loss", "keep true for candidate"),
        ("use_nussinov", "Enable Nussinov decode", "keep true for candidate"),
        ("use_pair_aware_masking", "Pair-aware masking", "candidate currently false"),
        ("use_motif_span_masking", "Motif-span masking", "candidate currently false"),
        ("use_motif_condition", "Use motif condition tokens", "candidate currently false"),
        ("use_family_condition", "Use family condition tokens", "candidate currently false"),
    ],
}


OVERVIEW_TEXT = """\
RNA-OmniPrefold
Relation-aware Masked Denoising for Constraint-Guided RNA Folding

Pipeline:
1. sequence-only masked nucleotide denoising
2. supervised pair-relation field prediction
3. lightweight 2D relation refinement
4. strict Nussinov constraint projection
5. planned offline preference optimization

[Core]
  RNAOmniDiffusion
    Student Transformer encoder with task, segment, time, and position embeddings.

[Input Variables]
  x_i  nucleotide token
  y_i  structure token
  r_ij pair-relation variable
  S_ij pair-relation logit
  R    pair-relation field

[Training Stages]
  Stage 1  Sequence-only masked nucleotide denoising.
  Stage 2  Supervised pair-relation prediction with pair-logit relation head.
  Stage 2  Optional token auxiliary objective when enabled by config.
  Stage 3  Strict Nussinov projection with canonical pair and loop constraints.

[Planned Preference Boundary]
  Offline preference critics may compare candidate structures during training data preparation.
  They do not predict dot-bracket truth, generate pair labels, or run inside model forward.

[Recommended Workflow]
  1. python main.py smoke
  2. python main.py params --config config/candidate.yaml
  3. python main.py train --config config/candidate.yaml --device cuda
  4. python scripts/eval.py bench --config config/candidate.yaml --ckpt outputs/candidate/best.pt --split test --device cuda --decode nussinov"""


def run_params(args: argparse.Namespace) -> None:
    """Print tunable parameters with current values, descriptions, and tuning hints."""
    from models.training import load_config

    config = load_config(args.config) if args.config else {}
    section_names = [args.section] if args.section != "all" else ["training", "model", "decoding", "ablation"]

    if args.json:
        result: dict[str, dict[str, Any]] = {}
        for section in section_names:
            registry = PARAM_REGISTRY.get(section, [])
            section_config = config.get(section, {}) if isinstance(config.get(section), dict) else {}
            result[section] = {}
            for name, desc, hint in registry:
                result[section][name] = {
                    "value": section_config.get(name, "<not set>"),
                    "description": desc,
                    "tuning_hint": hint,
                }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    print("Tunable Parameters")
    print(f"Config: {args.config}")
    print()

    for section in section_names:
        registry = PARAM_REGISTRY.get(section, [])
        if not registry:
            continue
        section_config = config.get(section, {}) if isinstance(config.get(section), dict) else {}
        rows: list[tuple[str, str, str, str]] = []
        for name, desc, hint in registry:
            val = section_config.get(name, "<not set>")
            val_str = str(val) if not isinstance(val, float) else f"{val:.6g}"
            rows.append((name, val_str, desc, hint))

        name_w = max(max(len(r[0]) for r in rows), 4)
        val_w = max(max(len(r[1]) for r in rows), 5)
        desc_w = max(max(len(r[2]) for r in rows), 11)

        print(f"[{section}]")
        print(f"  {'name':<{name_w}}  {'value':<{val_w}}  {'description':<{desc_w}}  tuning hint")
        print(f"  {'-' * name_w}  {'-' * val_w}  {'-' * desc_w}  -----------")
        for name, val_str, desc, hint in rows:
            print(f"  {name:<{name_w}}  {val_str:<{val_w}}  {desc:<{desc_w}}  {hint}")
        print()

    print("[quick commands]")
    print(f"  train : python main.py train --config {args.config} --device cuda")
    print("  smoke : python main.py smoke")


def run_overview(args: argparse.Namespace) -> None:
    print(OVERVIEW_TEXT)
    print()
    print("[Modes]")
    mode_w = max(len(name) for name in MODE_REGISTRY)
    for name, info in MODE_REGISTRY.items():
        if name == "models":
            continue
        print(f"  {name:<{mode_w}}  {info['purpose']}")
        print(f"  {' ' * mode_w}  > {info['typical']}")
    print()


def _run_train(args: argparse.Namespace) -> None:
    from models.training import run_train
    return run_train(args)


def _run_eval(args: argparse.Namespace) -> None:
    from models.training import run_eval
    return run_eval(args)


def _run_smoke(args: argparse.Namespace) -> None:
    from models.training import run_smoke
    return run_smoke(args)


def _run_infer(args: argparse.Namespace) -> None:
    from models.training import run_infer
    return run_infer(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RNA-OmniPrefold -- relation-aware masked denoising for constraint-guided RNA folding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train from a YAML config")
    train.add_argument("--config", default="config/candidate.yaml", help="YAML training config")
    train.add_argument("--resume", help="Optional checkpoint to resume from")
    train.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Training device")
    train.add_argument("--max_steps", type=int, help="Optional short-run step cap for sanity checks")
    train.add_argument("--train_subset", type=int, help="Optional number of training samples for sanity checks")
    train.set_defaults(func=_run_train)

    eval_parser = subparsers.add_parser("eval", help="Evaluate validation split from a checkpoint")
    eval_parser.add_argument("--config", default="config/candidate.yaml", help="YAML evaluation config")
    eval_parser.add_argument("--ckpt", required=True, help="Checkpoint path")
    eval_parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Evaluation device")
    eval_parser.set_defaults(func=_run_eval)

    infer = subparsers.add_parser("infer", help="Run single-sample inference")
    infer.add_argument("--config", default="config/candidate.yaml", help="YAML inference config")
    infer.add_argument("--ckpt", required=True, help="Checkpoint path")
    infer.add_argument("--task", required=True, choices=["seq2struct", "invfold"], help="Inference task")
    infer.add_argument("--seq", help="RNA sequence for seq2struct")
    infer.add_argument("--struct", help="Dot-bracket structure for invfold")
    infer.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device")
    infer.set_defaults(func=_run_infer)

    smoke = subparsers.add_parser("smoke", help="Run tiny CPU smoke test")
    smoke.set_defaults(func=_run_smoke)

    params = subparsers.add_parser("params", help="Show adjustable config parameters")
    params.add_argument("--config", default="config/candidate.yaml", help="Config to inspect")
    params.add_argument("--section", default="all", choices=["all", *PARAM_REGISTRY.keys()], help="Parameter section")
    params.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    params.set_defaults(func=run_params)

    overview = subparsers.add_parser("overview", help="Show framework overview")
    overview.set_defaults(func=run_overview)

    models = subparsers.add_parser("models", help="Show framework overview (alias for overview)")
    models.set_defaults(func=run_overview)

    return parser


_AUDIT_REEXPORTS: frozenset[str] = frozenset({
    "load_config",
    "load_checkpoint",
    "build_model",
    "loss_from_batch",
    "move_batch_to_device",
    "resolve_device",
})


def __getattr__(name: str):
    if name in _AUDIT_REEXPORTS:
        from models import training as _training
        return getattr(_training, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
