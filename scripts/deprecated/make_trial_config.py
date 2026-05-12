#!/usr/bin/env python
"""Temporary trial config generator for RNA-OmniPrefold.

Copies a base YAML config and applies overrides via --set key=value,
writing a self-contained config.yaml, changes.json (diff), and README.md
to the specified output directory without modifying the base config.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml


# Allowed override keys (section.key format)
ALLOWED_KEYS = frozenset({
    "tasks.seq2struct",
    "tasks.invfold",
    "tasks.inpaint",
    "tasks.motif_control",
    "training.lr",
    "training.batch_size",
    "training.lambda_pair",
    "training.pair_positive_weight",
    "training.pair_negative_ratio",
    "training.epochs",
    "decoding.pair_threshold",
    "decoding.nussinov_gamma",
    "decoding.decode_source",
    "ablation.use_pair_aware_masking",
    "ablation.use_motif_span_masking",
})

# Blacklisted key prefixes
BLACKLIST_PREFIXES = (
    "model.",
    "tokenizer.",
    "data.",
    "release.",
)

# Blacklisted substrings (paths, labels, checkpoints)
BLACKLIST_SUBSTRINGS = (
    "path",
    "label",
    "checkpoint",
)

# Task keys used for ratio-sum validation
TASK_KEYS = ("seq2struct", "invfold", "inpaint", "motif_control")


def parse_set_arg(raw: str) -> tuple[str, str, str]:
    """Parse 'section.key=value' into (section, key, value_str)."""
    if "=" not in raw:
        raise SystemExit(f"Invalid --set format (expected key=value): {raw}")
    path, value = raw.split("=", 1)
    if "." not in path:
        raise SystemExit(
            f"Invalid --set key format (expected section.key=value): {raw}"
        )
    section, key = path.split(".", 1)
    if not section or not key:
        raise SystemExit(
            f"Invalid --set key format (expected section.key=value): {raw}"
        )
    return section, key, value


def validate_key(full_key: str) -> None:
    """Validate an override key against the whitelist and blacklist."""
    if full_key in ALLOWED_KEYS:
        return

    for prefix in BLACKLIST_PREFIXES:
        if full_key.startswith(prefix):
            raise SystemExit(
                f"Key '{full_key}' is blacklisted (prefix '{prefix}'). "
                f"Only whitelisted keys are allowed."
            )

    lower = full_key.lower()
    for substr in BLACKLIST_SUBSTRINGS:
        if substr in lower:
            raise SystemExit(
                f"Key '{full_key}' appears to reference paths/labels/checkpoints "
                f"and is not allowed."
            )

    matches = sorted(
        k for k in ALLOWED_KEYS
        if any(part in k for part in full_key.split("."))
    )
    hint = ""
    if matches:
        hint = f"\n  Did you mean one of: {', '.join(matches)}"
    raise SystemExit(
        f"Key '{full_key}' is not in the whitelist.{hint}\n"
        f"  Allowed keys: {', '.join(sorted(ALLOWED_KEYS))}"
    )


def coerce_value(raw: str) -> int | float | bool | None | str:
    """Coerce a string value to the most appropriate Python type."""
    lowered = raw.strip()

    if lowered.lower() in ("true", "yes", "on"):
        return True
    if lowered.lower() in ("false", "no", "off"):
        return False
    if lowered.lower() in ("null", "none"):
        return None

    try:
        if "." in lowered or "e" in lowered.lower():
            return float(lowered)
        return int(lowered)
    except ValueError:
        pass

    return raw


def apply_overrides(
    config: dict,
    overrides: list[tuple[str, str, str]],
    normalize_tasks: bool,
) -> tuple[dict, dict]:
    """Apply validated overrides to a deep copy of config.

    Returns (modified_config, changes_record).
    """
    config = copy.deepcopy(config)
    changes: dict = {}

    for section, key, raw_value in overrides:
        full_key = f"{section}.{key}"
        validate_key(full_key)

        value = coerce_value(raw_value)
        old = config.get(section, {}).get(key, "__NOT_SET__")
        config.setdefault(section, {})[key] = value
        changes[full_key] = {"old": old, "new": value}

    # Check task ratio sum if any task keys were modified
    task_keys_modified = any(
        f"tasks.{tk}" in changes for tk in TASK_KEYS
    )
    if task_keys_modified:
        tasks = config.get("tasks", {})
        total = sum(
            float(tasks.get(tk, 0.0)) for tk in TASK_KEYS
        )
        if abs(total - 1.0) > 1e-9:
            if normalize_tasks:
                for tk in TASK_KEYS:
                    if tk in tasks:
                        tasks[tk] = float(tasks[tk]) / total
                changes["__task_ratios_normalized__"] = {
                    "original_sum": total,
                    "action": "normalized to 1.0",
                }
            else:
                changes["__task_ratios_warning__"] = {
                    "sum": total,
                    "warning": (
                        "Task ratios do not sum to 1.0. "
                        "Use --normalize-task-ratios to auto-normalize."
                    ),
                }

    return config, changes


def write_outputs(
    out_dir: Path,
    config: dict,
    changes: dict,
    base_path: str,
) -> None:
    """Write config.yaml, changes.json, and README.md to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = out_dir / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    changes_path = out_dir / "changes.json"
    changes_path.write_text(
        json.dumps(changes, indent=2) + "\n", encoding="utf-8"
    )

    readme_path = out_dir / "README.md"
    lines = [
        "# Trial Config",
        "",
        f"Base: `{base_path}`",
        f"Output: `{config_path.resolve()}`",
        "",
        "## Overrides",
        "",
    ]
    override_items = {
        k: v for k, v in changes.items() if not k.startswith("__")
    }
    if override_items:
        lines.append("| Key | Old | New |")
        lines.append("|---|---|---|")
        for key in sorted(override_items):
            info = override_items[key]
            old_val = info["old"]
            new_val = info["new"]
            if old_val == "__NOT_SET__":
                old_val = "(not set)"
            lines.append(f"| {key} | {old_val} | {new_val} |")
    else:
        lines.append("(no overrides applied)")

    for meta_key in sorted(k for k in changes if k.startswith("__")):
        meta = changes[meta_key]
        lines.append("")
        lines.append(f"- {meta_key}: {json.dumps(meta)}")

    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(out_dir: Path, changes: dict) -> None:
    """Print a human-readable summary to stdout."""
    print(f"Trial config generated: {out_dir.resolve()}")
    override_count = sum(1 for k in changes if not k.startswith("__"))
    print(f"  Overrides applied: {override_count}")

    for key in sorted(changes):
        if key.startswith("__"):
            continue
        info = changes[key]
        old = info["old"]
        new = info["new"]
        if old == "__NOT_SET__":
            print(f"    {key} = {new}  (added)")
        else:
            print(f"    {key} = {old} -> {new}")

    warning = changes.get("__task_ratios_warning__")
    if warning:
        print(
            f"  Warning: task ratios sum to {warning['sum']:.4f} "
            f"(expected 1.0)"
        )

    normalized = changes.get("__task_ratios_normalized__")
    if normalized:
        print(
            f"  Task ratios auto-normalized "
            f"(was {normalized['original_sum']:.4f})"
        )

    print("  Files written:")
    print(f"    {out_dir / 'config.yaml'}")
    print(f"    {out_dir / 'changes.json'}")
    print(f"    {out_dir / 'README.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a trial config by copying a base config and "
            "applying overrides."
        ),
    )
    parser.add_argument(
        "--base",
        default="config/candidate.yaml",
        help="Path to the base YAML config (default: config/candidate.yaml)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory path (e.g., outputs/trials/test_config)",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        metavar="KEY=VALUE",
        help="Override values in section.key=value format (repeatable)",
    )
    parser.add_argument(
        "--normalize-task-ratios",
        action="store_true",
        default=False,
        help="Auto-normalize task ratios to sum to 1.0 if they do not",
    )

    args = parser.parse_args()

    base_path = Path(args.base)
    if not base_path.exists():
        raise SystemExit(f"Base config not found: {base_path}")

    base_config = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
    if not isinstance(base_config, dict):
        raise SystemExit(
            f"Base config is not a valid YAML mapping: {base_path}"
        )

    parsed: list[tuple[str, str, str]] = []
    for raw in args.overrides:
        section, key, value = parse_set_arg(raw)
        parsed.append((section, key, value))

    config, changes = apply_overrides(
        base_config, parsed, args.normalize_task_ratios
    )

    out_dir = Path(args.out)
    write_outputs(out_dir, config, changes, str(base_path.resolve()))

    print_summary(out_dir, changes)


if __name__ == "__main__":
    main()
