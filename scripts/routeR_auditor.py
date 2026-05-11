# scripts/routeR_auditor.py — Route R: LLM-guided implementation audit hypothesis generator.
"""Generates testable audit hypotheses in hand, random, or LLM modes.

Allowed category taxonomy: data_alignment, split_leakage, decode_mismatch,
loss_mask_error, lr_schedule_mismatch, metric_bug, checkpoint_selection_bug,
eval_config_mismatch, label_conversion_bug, canonical_constraint_mismatch.
"""

from __future__ import annotations

import argparse, json, os, random, sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CATEGORIES = [
    "data_alignment", "split_leakage", "decode_mismatch", "loss_mask_error",
    "lr_schedule_mismatch", "metric_bug", "checkpoint_selection_bug",
    "eval_config_mismatch", "label_conversion_bug", "canonical_constraint_mismatch",
]


# ── Hand hypotheses ───────────────────────────────────────────────────
def generate_hand_hypotheses(audit_pack: dict) -> list[dict]:
    return [
        {
            "name": "min_loop_consistency_check",
            "category": "decode_mismatch",
            "why_plausible": "training pair extraction min_loop may differ from decode min_loop",
            "check": "compare min_loop_length in config with min_loop used in dataset pair parsing",
            "expected_failure_signature": "nonzero count of gold pairs with |i-j| < decode min_loop but >= train min_loop",
            "fix_if_confirmed": "align min_loop in dataset parser and decode config",
        },
        {
            "name": "warmup_effective_lr_mismatch",
            "category": "lr_schedule_mismatch",
            "why_plausible": "warmup=50 may cause effective lr to be lower than nominal lr=0.001 at early steps",
            "check": "verify effective lr = lr * min(1, step/warmup) matches expected schedule",
            "expected_failure_signature": "effective lr at step 50 != 0.001",
            "fix_if_confirmed": "adjust warmup or lr to ensure target lr is reached",
        },
        {
            "name": "pair_label_padding_alignment",
            "category": "data_alignment",
            "why_plausible": "pair_labels matrix may include padding positions beyond sequence length",
            "check": "check pair_labels[i, j] for i>=L or j>=L; verify pair_mask excludes these",
            "expected_failure_signature": "nonzero pair_labels or pair_mask in padded region",
            "fix_if_confirmed": "ensure pair_labels and pair_mask zero out beyond sequence length",
        },
        {
            "name": "val_pair_f1_best_checkpoint_verification",
            "category": "checkpoint_selection_bug",
            "why_plausible": "best.pt may not correspond to highest val_pair_f1 epoch",
            "check": "parse trainlog, find epoch with max val_pair_f1, verify best.pt was saved at that epoch",
            "expected_failure_signature": "best.pt epoch != max val_pair_f1 epoch in trainlog",
            "fix_if_confirmed": "fix save_best_by logic or min_delta threshold",
        },
        {
            "name": "canonical_wobble_constraint_consistency",
            "category": "canonical_constraint_mismatch",
            "why_plausible": "training pair_labels include GU wobble, but decode may filter differently with allow_wobble",
            "check": "count GU pairs in gold labels vs decode predictions; compare allow_wobble settings",
            "expected_failure_signature": "GU pairs in labels but excluded by decode, or vice versa",
            "fix_if_confirmed": "ensure canonical constraint logic same in label parsing and decode",
        },
        {
            "name": "loss_mask_diagonal_exclusion",
            "category": "loss_mask_error",
            "why_plausible": "pair loss mask may include diagonal (i=j) or lower triangle positions",
            "check": "verify pair_mask excludes diagonal and lower triangle for all batch items",
            "expected_failure_signature": "any True values on diagonal or lower triangle of pair_mask",
            "fix_if_confirmed": "add upper-triangle-only mask in pair loss computation",
        },
        {
            "name": "eval_config_vs_train_config_consistency",
            "category": "eval_config_mismatch",
            "why_plausible": "eval bench may use different decode settings than training config",
            "check": "compare decode params (gamma, threshold, min_loop) in train config vs eval config",
            "expected_failure_signature": "decode params differ between train and eval configs",
            "fix_if_confirmed": "use same decode params for training and evaluation",
        },
        {
            "name": "pair_count_bias_in_decode",
            "category": "metric_bug",
            "why_plausible": "predicted pair counts may systematically over-estimate true pair counts",
            "check": "compute mean(pred_pairs - true_pairs) from eval results",
            "expected_failure_signature": "mean pair count bias > 5",
            "fix_if_confirmed": "adjust pair_threshold or nussinov_gamma to reduce over-prediction",
        },
    ]


# ── Random hypotheses ─────────────────────────────────────────────────
def generate_random_hypotheses(audit_pack: dict, num: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    templates = [
        ("{field}_alignment", "data_alignment", "check {field} consistency between dataset and collator", "mismatch between dataset and collator for {field}"),
        ("{field}_leakage", "split_leakage", "check {field} overlap between train and val splits", "duplicate {field} across splits"),
        ("decode_{param}_check", "decode_mismatch", "verify decode param {param} matches training expectation", "{param} mismatch between training and decode"),
        ("loss_{component}_mask", "loss_mask_error", "verify {component} loss mask excludes invalid positions", "invalid positions included in {component} loss mask"),
        ("lr_{aspect}_check", "lr_schedule_mismatch", "verify lr schedule {aspect} matches config", "lr {aspect} deviates from expected schedule"),
        ("metric_{name}_audit", "metric_bug", "audit {name} computation for edge cases", "{name} metric produces incorrect values for edge cases"),
        ("ckpt_{criterion}_check", "checkpoint_selection_bug", "verify checkpoint selection by {criterion}", "best checkpoint by {criterion} not saved correctly"),
    ]

    fields = ["seq", "struct", "pairs", "length", "id"]
    params = ["gamma", "threshold", "min_loop", "allow_wobble"]
    components = ["pair", "token", "struct"]
    aspects = ["warmup", "decay", "clip", "effective"]
    names = ["f1", "precision", "recall", "pair_ratio"]
    criteria = ["val_loss", "val_pair_f1", "epoch"]

    proposals = []
    for i in range(num):
        tmpl = rng.choice(templates)
        name = tmpl[0].format(
            field=rng.choice(fields), param=rng.choice(params),
            component=rng.choice(components), aspect=rng.choice(aspects),
            name=rng.choice(names), criterion=rng.choice(criteria),
        )
        proposals.append({
            "name": name + f"_{i}",
            "category": tmpl[1],
            "why_plausible": "random audit hypothesis",
            "check": tmpl[2].format(
                field=rng.choice(fields), param=rng.choice(params),
                component=rng.choice(components), aspect=rng.choice(aspects),
                name=rng.choice(names), criterion=rng.choice(criteria),
            ),
            "expected_failure_signature": tmpl[3].format(
                field=rng.choice(fields), param=rng.choice(params),
                component=rng.choice(components), aspect=rng.choice(aspects),
                name=rng.choice(names), criterion=rng.choice(criteria),
            ),
            "fix_if_confirmed": "fix the identified inconsistency",
        })

    return proposals


# ── LLM hypotheses ──────────────────────────────────────────────────
def _load_dotenv():
    env_path = ROOT / ".env"
    if not env_path.exists(): return
    for line in env_path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or "=" not in line: continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and k not in os.environ: os.environ[k] = v


def generate_llm_hypotheses(audit_pack: dict, num: int) -> list[dict]:
    _load_dotenv()
    base_url = os.environ.get("LLM_BASE_URL", "")
    token = os.environ.get("LLM_TOKEN", "")
    model_id = os.environ.get("LLM_MODEL", "")
    if not base_url or not token or not model_id:
        print("LLM credentials not found; skipping.")
        return []

    # Compact audit pack
    compact = {
        "mainline": audit_pack.get("mainline_summary", {}),
        "dataflow": audit_pack.get("dataflow", {}),
        "training": audit_pack.get("training", {}),
        "decode": audit_pack.get("decode", {}),
        "model": audit_pack.get("model", {}),
        "failed_routes": audit_pack.get("failed_routes", []),
    }

    prompt = json.dumps({
        "task": f"Propose {num} testable implementation audit hypotheses for an RNA folding model pipeline. You are NOT proposing new models or hyperparameters — you are looking for bugs, mismatches, and inconsistencies.",
        "audit_context": compact,
        "allowed_categories": CATEGORIES,
        "output_format": '{"hypotheses": [{"name": "...", "category": "...", "why_plausible": "...", "check": "...", "expected_failure_signature": "...", "fix_if_confirmed": "..."}]}',
        "rules": [f"Max {num} hypotheses", "Category must be from allowed list", "Check must be executable by a script", "Fix must not introduce new model modules"],
    })

    payload = json.dumps({
        "model": model_id, "temperature": 0.3, "max_tokens": 2000,
        "messages": [
            {"role": "system", "content": "You are a scientific ML implementation auditor. Propose only testable bug/mismatch hypotheses. Return strict JSON."},
            {"role": "user", "content": prompt},
        ],
    }).encode()

    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/v1/chat/completions", data=payload,
                                      headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            content = json.loads(resp.read())["choices"][0]["message"]["content"].strip()
        if content.startswith("```"): content = content.split("\n", 1)[-1].rstrip("```").strip()
        raw = json.loads(content)
        hyps = raw.get("hypotheses", [])
        print(f"LLM returned {len(hyps)} hypotheses")
    except Exception as e:
        print(f"LLM call failed: {e}")
        return []

    valid = []
    for h in hyps[:num]:
        if not isinstance(h, dict): continue
        if not h.get("name") or not h.get("category"): continue
        if h["category"] not in CATEGORIES: continue
        valid.append(h)
    return valid


# ── Main ──────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["hand", "random", "llm"])
    p.add_argument("--audit_pack", default="outputs/reports/routeR_audit_pack.json")
    p.add_argument("--num", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="outputs/reports/routeR_hypotheses.json")
    p.add_argument("--report", default="outputs/reports/routeR_hypotheses.md")
    args = p.parse_args()

    pack = {}
    if Path(args.audit_pack).exists():
        pack = json.loads(Path(args.audit_pack).read_text())

    if args.mode == "hand":
        hyps = generate_hand_hypotheses(pack)
    elif args.mode == "random":
        hyps = generate_random_hypotheses(pack, args.num, args.seed)
    else:
        hyps = generate_llm_hypotheses(pack, args.num)

    print(f"Generated {len(hyps)} hypotheses (mode={args.mode})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(hyps, indent=2))

    lines = [f"# Route R {args.mode} hypotheses", f"Count: {len(hyps)}", ""]
    for h in hyps:
        lines.append(f"## {h['name']}")
        lines.append(f"- Category: {h['category']}")
        lines.append(f"- Why plausible: {h['why_plausible']}")
        lines.append(f"- Check: {h['check']}")
        lines.append(f"- Expected: {h['expected_failure_signature']}")
        lines.append(f"- Fix: {h['fix_if_confirmed']}")
        lines.append("")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines))


if __name__ == "__main__":
    main()
