# scripts/routeQ_planner.py — LLM-Guided Experiment Planner (Route Q).
"""Generates allowlisted experiment proposals in hand, random, or LLM modes.

Output: YAML config files + JSON proposal report."""

from __future__ import annotations

import argparse, json, os, random, sys, yaml
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASE_CONFIG = Path(ROOT, "config", "mainline_lr0010.yaml")
SEARCHSPACE_DEFAULT = Path(ROOT, "config", "routeQ_searchspace.yaml")


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_searchspace(path: Path) -> dict:
    return load_yaml(path).get("search_space", {})


# ── Hand proposals ────────────────────────────────────────────────────
def generate_hand_proposals(searchspace: dict, error_report: dict) -> list[dict]:
    """Pre-defined hand-crafted proposals targeting common error modes."""
    proposals = [
        {
            "name": "warmup50_lr0012",
            "hypothesis": "Higher learning rate may improve convergence at 300 steps",
            "target_error": ["low_recall"],
            "config_changes": {"training": {"lr": 0.0012, "warmup_steps": 50}},
            "expected_effect": "Increase pair recall",
        },
        {
            "name": "pairrefine_deeper_2x16",
            "hypothesis": "Deeper PairRefine may capture longer-range interactions",
            "target_error": ["long_range", "multi_stem"],
            "config_changes": {"model": {"pairrefine_blocks": 2}, "training": {}},
            "expected_effect": "Improve long-range pair F1",
        },
        {
            "name": "pairrefine_wider_1x32",
            "hypothesis": "Wider PairRefine channels may improve pair representation",
            "target_error": ["over_pairing", "dense_pairs"],
            "config_changes": {"model": {"pairrefine_channels": 32}, "training": {}},
            "expected_effect": "Improve pair precision for dense structures",
        },
        {
            "name": "residual_conv_small",
            "hypothesis": "Small 2D conv residual on pair logits adds capacity",
            "target_error": ["fragmented_stem", "sparse_pairs"],
            "config_changes": {
                "training": {},
                "pair_residual": {"enabled": True, "type": "conv2d", "kernel_size": 3, "residual_scale": 0.1, "init": "zero"},
            },
            "expected_effect": "Improve weak-pattern F1 through residual capacity",
        },
        {
            "name": "decode_gamma3",
            "hypothesis": "Sharper Nussinov may improve precision on over-pairing cases",
            "target_error": ["over_pairing"],
            "config_changes": {"training": {}, "decode": {"nussinov_gamma": 3.0}},
            "expected_effect": "Reduce over-pairing via stricter decode",
        },
        {
            "name": "lambda7_pos15",
            "hypothesis": "Stronger pair loss + higher positive weight for recall",
            "target_error": ["under_pairing", "long_range"],
            "config_changes": {"training": {"lambda_pair": 7.0, "pair_positive_weight": 15.0}},
            "expected_effect": "Increase pair recall",
        },
    ]
    return [p for p in proposals if _validate_proposal(p, searchspace)]


# ── Random proposals ──────────────────────────────────────────────────
def generate_random_proposals(searchspace: dict, num: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    proposals = []
    ss = searchspace

    # Flatten search space choices
    training_choices = {
        "lr": list(ss.get("training", {}).get("lr", [0.001])),
        "warmup_steps": list(ss.get("training", {}).get("warmup_steps", [50])),
        "weight_decay": list(ss.get("training", {}).get("weight_decay", [0.01])),
    }
    model_choices = {
        "pairrefine_blocks": list(ss.get("model", {}).get("pairrefine_blocks", [1])),
        "pairrefine_channels": list(ss.get("model", {}).get("pairrefine_channels", [16])),
        "pairrefine_dropout": list(ss.get("model", {}).get("pairrefine_dropout", [0.0])),
    }
    res_choices = {
        "enabled": list(ss.get("pair_residual", {}).get("enabled", [False])),
        "kernel_size": list(ss.get("pair_residual", {}).get("kernel_size", [3])),
        "residual_scale": list(ss.get("pair_residual", {}).get("residual_scale", [0.1])),
        "init": list(ss.get("pair_residual", {}).get("init", ["zero"])),
    }
    decode_choices = {
        "nussinov_gamma": list(ss.get("decode", {}).get("nussinov_gamma", [2.0])),
        "pair_threshold": list(ss.get("decode", {}).get("pair_threshold", [0.25])),
    }
    loss_choices = {
        "lambda_pair": list(ss.get("loss", {}).get("lambda_pair", [5.0])),
        "pair_positive_weight": list(ss.get("loss", {}).get("pair_positive_weight", [10.0])),
    }

    targets = ["under_pairing", "over_pairing", "low_recall", "low_precision",
               "long_range", "sparse_pairs", "dense_pairs", "fragmented_stem"]

    for i in range(num):
        # Randomly pick 1-2 change categories
        categories = rng.sample(["training", "model", "pair_residual", "decode", "loss"],
                                  k=rng.choice([1, 2]))
        changes = {"training": {}}
        for cat in categories:
            if cat == "training":
                changes["training"] = {
                    "lr": rng.choice(training_choices["lr"]),
                    "warmup_steps": rng.choice(training_choices["warmup_steps"]),
                }
            elif cat == "model":
                changes["model"] = {
                    "pairrefine_blocks": rng.choice(model_choices["pairrefine_blocks"]),
                    "pairrefine_channels": rng.choice(model_choices["pairrefine_channels"]),
                }
            elif cat == "pair_residual":
                changes["pair_residual"] = {
                    "enabled": rng.choice(res_choices["enabled"]),
                    "type": "conv2d",
                    "kernel_size": rng.choice(res_choices["kernel_size"]),
                    "residual_scale": rng.choice(res_choices["residual_scale"]),
                    "init": rng.choice(res_choices["init"]),
                }
            elif cat == "decode":
                changes["decode"] = {
                    "nussinov_gamma": rng.choice(decode_choices["nussinov_gamma"]),
                }
            elif cat == "loss":
                changes["training"].update({
                    "lambda_pair": rng.choice(loss_choices["lambda_pair"]),
                    "pair_positive_weight": rng.choice(loss_choices["pair_positive_weight"]),
                })

        proposal = {
            "name": f"random_{i+1:02d}",
            "hypothesis": f"Randomly sampled config variant #{i+1}",
            "target_error": rng.sample(targets, k=rng.choice([1, 2])),
            "config_changes": changes,
            "expected_effect": "exploratory",
        }
        proposals.append(proposal)

    return [p for p in proposals if _validate_proposal(p, searchspace)]


# ── LLM proposals ─────────────────────────────────────────────────────
def _load_dotenv():
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists(): return
    for line in env_path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or "=" not in line: continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and k not in os.environ: os.environ[k] = v


def generate_llm_proposals(searchspace: dict, error_report: dict, num: int) -> list[dict]:
    _load_dotenv()
    base_url = os.environ.get("LLM_BASE_URL", "")
    token = os.environ.get("LLM_TOKEN", "")
    model_id = os.environ.get("LLM_MODEL", "")
    if not base_url or not token or not model_id:
        print("LLM credentials not found; skipping LLM proposals.")
        return []

    # Compact error taxonomy — only key numbers
    overall = error_report.get("overall", {})
    dist = error_report.get("pair_distance", {})
    pat = error_report.get("structure_pattern", {})

    # Compact search space as flat key-value ranges
    ss_compact = {
        "training.lr": searchspace.get("training", {}).get("lr", []),
        "training.warmup_steps": searchspace.get("training", {}).get("warmup_steps", []),
        "model.pairrefine_blocks": searchspace.get("model", {}).get("pairrefine_blocks", []),
        "model.pairrefine_channels": searchspace.get("model", {}).get("pairrefine_channels", []),
        "pair_residual.enabled": [False, True],
        "pair_residual.kernel_size": searchspace.get("pair_residual", {}).get("kernel_size", []),
        "decode.nussinov_gamma": searchspace.get("decode", {}).get("nussinov_gamma", []),
        "training.lambda_pair": searchspace.get("loss", {}).get("lambda_pair", []),
    }

    prompt = json.dumps({
        "task": f"Propose exactly {num} experiments to improve RNA folding model F1. The model is MS-MPRM + PairRefine with strict Nussinov decode. Current val F1 is {overall.get('f1','?')}. Precision={overall.get('precision','?')}, Recall={overall.get('recall','?')}.",
        "error_summary": {
            "pair_distance": {k: {"f1": v["f1"], "support": v["support"]} for k, v in dist.items()},
            "structure_pattern": {k: {"count": v["count"], "f1": v.get("mean_f1")} for k, v in pat.items()},
        },
        "allowed_fields": ss_compact,
        "output_format": '{"proposals": [{"name": "...", "hypothesis": "...", "target_error": ["..."], "config_changes": {"training": {"lr": 0.001}, "model": {"pairrefine_channels": 32}}, "expected_effect": "..."}]}',
        "rules": ["Use only allowed_fields values", "Prefer 1-2 changes per proposal", f"Return {num} proposals in strict JSON"],
    })

    system_msg = "You are an experiment planner for RNA folding models. Return strict JSON only. No markdown, no explanation outside the JSON."

    import urllib.request
    payload_bytes = json.dumps({
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3, "max_tokens": 1500,
    }).encode()

    try:
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            data=payload_bytes,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            content = json.loads(resp.read())["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rstrip("```").strip()
        raw = json.loads(content)
        proposals = raw.get("proposals", [])
        print(f"LLM returned {len(proposals)} proposals")
    except Exception as e:
        print(f"LLM call failed: {e}")
        return []

    validated = []
    for p in proposals[:num]:
        if not isinstance(p, dict): continue
        if not p.get("name") or not p.get("config_changes"): continue
        if _validate_proposal(p, searchspace):
            validated.append(p)
        else:
            print(f"  Skipped invalid proposal: {p.get('name', '?')}")

    return validated


# ── Validation ────────────────────────────────────────────────────────
def _validate_proposal(proposal: dict, searchspace: dict) -> bool:
    """Check all config_changes fields exist in searchspace with allowed values."""
    changes = proposal.get("config_changes", {})
    ss = searchspace

    field_map = {
        ("training", "lr"): ss.get("training", {}).get("lr", []),
        ("training", "warmup_steps"): ss.get("training", {}).get("warmup_steps", []),
        ("training", "weight_decay"): ss.get("training", {}).get("weight_decay", []),
        ("training", "lambda_pair"): ss.get("loss", {}).get("lambda_pair", []),
        ("training", "pair_positive_weight"): ss.get("loss", {}).get("pair_positive_weight", []),
        ("model", "pairrefine_blocks"): ss.get("model", {}).get("pairrefine_blocks", []),
        ("model", "pairrefine_channels"): ss.get("model", {}).get("pairrefine_channels", []),
        ("model", "pairrefine_dropout"): ss.get("model", {}).get("pairrefine_dropout", []),
        ("decode", "nussinov_gamma"): ss.get("decode", {}).get("nussinov_gamma", []),
        ("decode", "pair_threshold"): ss.get("decode", {}).get("pair_threshold", []),
    }

    for section, sdict in changes.items():
        if not isinstance(sdict, dict): return False
        for key, value in sdict.items():
            # pair_residual is a special nested section
            if section == "pair_residual":
                allowed = ss.get("pair_residual", {}).get(key, [])
                if not allowed: return False
                if value not in allowed: return False
                continue
            # Regular field check
            allowed = field_map.get((section, key))
            if allowed is None: return False
            if value not in allowed: return False

    # Check no forbidden fields
    forbidden = ss.get("forbidden", [])
    for section, sdict in changes.items():
        for key in sdict:
            if key in forbidden: return False

    return True


# ── Config generation ─────────────────────────────────────────────────
def apply_changes_to_config(base_config: dict, changes: dict) -> dict:
    """Apply config_changes to a deep copy of base_config."""
    config = deepcopy(base_config)
    for section, sdict in changes.items():
        if section == "pair_residual":
            config.setdefault("pair_residual", {}).update(sdict)
        elif section == "decode":
            config.setdefault("decoding", {}).update(sdict)
        elif section == "model":
            config.setdefault("model", {}).update(sdict)
        elif section == "training":
            config.setdefault("training", {}).update(sdict)
    return config


def write_proposal_configs(proposals: list[dict], base_config: dict, out_dir: Path, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, prop in enumerate(proposals):
        cfg = apply_changes_to_config(base_config, prop["config_changes"])
        # Update output_dir
        name = prop["name"].replace(" ", "_")
        cfg["training"]["output_dir"] = f"outputs/routeQ_{prefix}_{name}"
        path = out_dir / f"{prefix}_{name}.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        written.append(str(path))
    return written


# ── Main ───────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["hand", "random", "llm"])
    p.add_argument("--searchspace", default=str(SEARCHSPACE_DEFAULT))
    p.add_argument("--error_report")
    p.add_argument("--out_dir", default="config/routeQ_hand")
    p.add_argument("--num", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report", default="outputs/reports/routeQ_plan.md")
    p.add_argument("--cache_dir")
    args = p.parse_args()

    ss = load_searchspace(Path(args.searchspace))
    base_config = load_yaml(BASE_CONFIG)

    error_report = {}
    if args.error_report and Path(args.error_report).exists():
        error_report = json.loads(Path(args.error_report).read_text())

    if args.mode == "hand":
        proposals = generate_hand_proposals(ss, error_report)
    elif args.mode == "random":
        proposals = generate_random_proposals(ss, args.num, args.seed)
    elif args.mode == "llm":
        proposals = generate_llm_proposals(ss, error_report, args.num)

    print(f"Generated {len(proposals)} proposals (mode={args.mode})")

    # Write proposal report
    report_lines = [f"# Route Q {args.mode} proposals", f"Count: {len(proposals)}", ""]
    for prop in proposals:
        report_lines.append(f"## {prop['name']}")
        report_lines.append(f"- Hypothesis: {prop['hypothesis']}")
        report_lines.append(f"- Target: {prop.get('target_error', [])}")
        report_lines.append(f"- Changes: {json.dumps(prop['config_changes'], indent=2)}")
        report_lines.append(f"- Expected: {prop['expected_effect']}")
        report_lines.append("")

    report_path = Path(args.report); report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines))

    # Write YAML configs
    out_dir = Path(args.out_dir)
    written = write_proposal_configs(proposals, base_config, out_dir, args.mode)
    for w in written:
        print(f"  Wrote: {w}")

    # Also write proposals JSON
    json_path = out_dir / f"proposals_{args.mode}.json"
    json_path.write_text(json.dumps(proposals, indent=2))


if __name__ == "__main__":
    main()
