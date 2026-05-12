# scripts/routeS_reasoner.py — Route S: LLM Causal Reasoning Planner.
"""Generates causal mechanism hypotheses from experiment history in
hand, random, or LLM modes. LLM receives structured history pack."""

from __future__ import annotations

import argparse, json, os, random, sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_dotenv():
    env_path = ROOT / ".env"
    if not env_path.exists(): return
    for line in env_path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or "=" not in line: continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and k not in os.environ: os.environ[k] = v


INTERVENTIONS = [
    "constraint_consistency_fix",
    "decode_calibration",
    "schedule_refinement",
    "pairrefine_capacity_sweep",
    "pair_loss_balance_sweep",
    "data_filtering_or_constraint_alignment",
    "checkpoint_selection_or_early_stopping",
    "evaluation_slice_targeted_fix",
]


# ── Hand ──────────────────────────────────────────────────────────────
def generate_hand_hypotheses(history: dict) -> list[dict]:
    return [
        {
            "name": "min_loop_constraint_alignment",
            "causal_claim": "Aligning min_loop across training loss, labels, and decode removes impossible-pair noise from loss signal.",
            "supporting_evidence": [
                "Route R hand audit found 28 val-set pairs with |i-j| < min_loop=3",
                "strict Nussinov decode drops these pairs",
            ],
            "contradicting_evidence": ["issue severity marked medium/minor; 28 pairs may be too few to matter"],
            "minimal_experiment": {
                "intervention_class": "constraint_consistency_fix",
                "config_or_code_change": "exclude |i-j|<min_loop pairs from pair loss mask",
                "control": "strongest_mainline",
                "budget": "300-step",
            },
            "expected_metric_change": {"overall_f1": "+0.003 to +0.010", "slice": "short_range precision"},
            "failure_interpretation": "28 pairs too few to affect overall F1",
        },
        {
            "name": "checkpoint_by_f1_not_loss",
            "causal_claim": "Saving best.pt by val_pair_f1 instead of val_loss may select a better model.",
            "supporting_evidence": [
                "val_loss and val_pair_f1 can diverge at epoch boundaries",
                "current save_best_by is val_pair_f1 in strongest config",
            ],
            "contradicting_evidence": ["current config already uses save_best_by=val_pair_f1"],
            "minimal_experiment": {
                "intervention_class": "checkpoint_selection_or_early_stopping",
                "config_or_code_change": "compare best.pt at save_best_by=val_loss vs save_best_by=val_pair_f1",
                "control": "strongest_mainline",
                "budget": "300-step",
            },
            "expected_metric_change": {"overall_f1": "0 to +0.003"},
            "failure_interpretation": "val_loss and val_pair_f1 correlate well; no gain",
        },
        {
            "name": "decode_temperature_calibration",
            "causal_claim": "Softening or sharpening Nussinov score scale may improve precision-recall balance.",
            "supporting_evidence": [
                "over_pairing affects 132/282 samples, F1=0.235",
                "adjusting gamma changes effective pair threshold behavior",
            ],
            "contradicting_evidence": [
                "gamma=2.0 is standard; changing it may break valid structure rate",
            ],
            "minimal_experiment": {
                "intervention_class": "decode_calibration",
                "config_or_code_change": "eval-only: test nussinov_gamma in [1.0, 2.0, 3.0]",
                "control": "strongest_mainline (gamma=2.0)",
                "budget": "eval-only",
            },
            "expected_metric_change": {"overall_f1": "+0.003 to +0.008"},
            "failure_interpretation": "gamma=2.0 already optimal",
        },
        {
            "name": "pairrefine_depth_exploration",
            "causal_claim": "Deeper PairRefine (2 blocks) may capture longer-range interactions.",
            "supporting_evidence": [
                "long_range F1=0.28, recall=0.30 — worst distance category",
                "PairRefine operates on full LxL matrix, 2D conv receptive field grows with depth",
            ],
            "contradicting_evidence": [
                "previous deeper pairrefine (32ch 2blk) with no warmup was worse (0.243)",
                "but that was confounded by no-warmup and excessive channels",
            ],
            "minimal_experiment": {
                "intervention_class": "pairrefine_capacity_sweep",
                "config_or_code_change": "pairrefine_blocks=2, channels=16, warmup=50, lr=0.001",
                "control": "strongest_mainline",
                "budget": "300-step",
            },
            "expected_metric_change": {"overall_f1": "+0.003 to +0.010", "slice": "long_range recall"},
            "failure_interpretation": "deeper pairrefine overfits or needs more steps",
        },
        {
            "name": "lambda_pair_sweep",
            "causal_claim": "Higher lambda_pair may improve recall for under-paired samples.",
            "supporting_evidence": [
                "over_pairing affects 132 samples but precision-recall balance may benefit from stronger pair loss",
            ],
            "contradicting_evidence": [
                "previous lambda_pair=7.0 with lower lr was close to mainline (0.3235)",
            ],
            "minimal_experiment": {
                "intervention_class": "pair_loss_balance_sweep",
                "config_or_code_change": "lambda_pair=7.0, keep lr=0.001",
                "control": "strongest_mainline",
                "budget": "300-step",
            },
            "expected_metric_change": {"overall_f1": "-0.005 to +0.005"},
            "failure_interpretation": "pair BCE already well-balanced at lambda=5.0",
        },
        {
            "name": "warmup_reduction",
            "causal_claim": "Reducing warmup to 0 may allow faster convergence at 300 steps.",
            "supporting_evidence": [
                "effective lr at step 50 = 0.001 (correct), but early steps slower",
            ],
            "contradicting_evidence": [
                "previous warmup=0 combined with wider pairrefine was worse",
                "but that may be the pairrefine change, not warmup",
            ],
            "minimal_experiment": {
                "intervention_class": "schedule_refinement",
                "config_or_code_change": "warmup_steps=0, keep lr=0.001",
                "control": "strongest_mainline",
                "budget": "300-step",
            },
            "expected_metric_change": {"overall_f1": "-0.003 to +0.005"},
            "failure_interpretation": "small warmup period stabilizes training; no gain from removing it",
        },
    ]


# ── Random ────────────────────────────────────────────────────────────
def generate_random_hypotheses(history: dict, num: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    hyps = []
    for i in range(num):
        inter_class = rng.choice(INTERVENTIONS)
        hyps.append({
            "name": f"random_causal_{i+1:02d}",
            "causal_claim": f"Random hypothesis #{i+1}: exploring {inter_class}",
            "supporting_evidence": ["exploratory"],
            "contradicting_evidence": ["none known"],
            "minimal_experiment": {
                "intervention_class": inter_class,
                "config_or_code_change": f"random change in {inter_class}",
                "control": "strongest_mainline",
                "budget": "300-step",
            },
            "expected_metric_change": {"overall_f1": "-0.01 to +0.01"},
            "failure_interpretation": "no effect",
        })
    return hyps


# ── LLM ───────────────────────────────────────────────────────────────
def generate_llm_hypotheses(history: dict, num: int) -> list[dict]:
    _load_dotenv()
    base_url = os.environ.get("LLM_BASE_URL", "")
    token = os.environ.get("LLM_TOKEN", "")
    model_id = os.environ.get("LLM_MODEL", "")
    if not base_url or not token or not model_id:
        print("LLM creds not found; skip.")
        return []

    # Compact history
    compact = {
        "mainline": history.get("current_strongest_mainline", {}),
        "positive": history.get("positive_findings", []),
        "negative": [n["route"] for n in history.get("negative_findings", [])],
        "confounds": [c["description"] for c in history.get("confound_corrections", [])],
        "error_highlights": history.get("error_taxonomy_highlights", {}),
    }

    prompt = json.dumps({
        "task": f"Propose {num} causal hypotheses for improving RNA folding model F1 from 0.3321. Each hypothesis must cite supporting and contradicting evidence from the history pack. Define a minimal controlled experiment within an allowed intervention class.",
        "history_pack": compact,
        "allowed_interventions": INTERVENTIONS,
        "output_format": '{"hypotheses": [{"name":"...","causal_claim":"...","supporting_evidence":["..."],"contradicting_evidence":["..."],"minimal_experiment":{"intervention_class":"...","config_or_code_change":"...","control":"strongest_mainline","budget":"300-step"},"expected_metric_change":{"overall_f1":"+0.003"},"failure_interpretation":"..."}]}',
        "rules": [f"Max {num} hypotheses", "Must cite at least one supporting evidence", "Must define minimal controlled experiment"],
    })

    payload = json.dumps({
        "model": model_id, "temperature": 0.3, "max_tokens": 2000,
        "messages": [
            {"role": "system", "content": "You are a scientific reasoning assistant for RNA folding experiments. Propose causal hypotheses from experiment history. Return strict JSON only."},
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
        print(f"LLM fail: {e}")
        return []

    valid = []
    for h in hyps[:num]:
        if not isinstance(h, dict): continue
        if not h.get("name") or not h.get("causal_claim"): continue
        exp = h.get("minimal_experiment", {})
        if exp.get("intervention_class") not in INTERVENTIONS: continue
        valid.append(h)
    return valid


# ── Main ──────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["hand", "random", "llm"])
    p.add_argument("--history", default="outputs/reports/routeS_history_pack.json")
    p.add_argument("--num", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="outputs/reports/routeS_hypotheses.json")
    p.add_argument("--report", default="outputs/reports/routeS_hypotheses.md")
    args = p.parse_args()

    history = {}
    if Path(args.history).exists():
        history = json.loads(Path(args.history).read_text())

    if args.mode == "hand":
        hyps = generate_hand_hypotheses(history)
    elif args.mode == "random":
        hyps = generate_random_hypotheses(history, args.num, args.seed)
    else:
        hyps = generate_llm_hypotheses(history, args.num)

    print(f"Generated {len(hyps)} hypotheses (mode={args.mode})")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(hyps, indent=2))

    lines = [f"# Route S {args.mode} hypotheses", f"Count: {len(hyps)}", ""]
    for h in hyps:
        lines.append(f"## {h['name']}")
        lines.append(f"- Claim: {h['causal_claim']}")
        lines.append(f"- Supporting: {h.get('supporting_evidence',[])}")
        lines.append(f"- Contradicting: {h.get('contradicting_evidence',[])}")
        lines.append(f"- Experiment: {h.get('minimal_experiment',{})}")
        lines.append("")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines))


if __name__ == "__main__":
    main()
