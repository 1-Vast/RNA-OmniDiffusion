"""Preference judge for candidate RNA structures.

Supports three judge modes:
  rule   -- heuristic comparison via score_struct features
  llm    -- call a general-purpose LLM API (mock or env provider)
  mock   -- random preference (control)

Usage
-----
Rule:
  conda run -n DL python scripts/judge.py \\
    --input outputs/pref/cand.jsonl \\
    --out outputs/pref/rulebuf.jsonl \\
    --mode rule

LLM mock:
  conda run -n DL python scripts/judge.py \\
    --input outputs/pref/cand.jsonl \\
    --out outputs/pref/llmbuf.mock.jsonl \\
    --mode llm --provider mock

LLM real:
  conda run -n DL python scripts/judge.py \\
    --input outputs/pref/cand.jsonl \\
    --out outputs/pref/llmbuf.jsonl \\
    --mode llm --provider env --limit 16
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.reward import score_struct


# ---- RAG Knowledge Retrieval ----

def load_kb(path: str | Path) -> str:
    """Load knowledge base text file."""
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def retrieve_evidence(query: str, kb_text: str, top_k: int = 3) -> str:
    """Simple keyword-based retrieval from knowledge base.

    Splits KB into sections, scores by keyword overlap with query.
    Returns top-k sections concatenated.
    """
    sections = kb_text.split("\n\n")
    if len(sections) <= top_k:
        return "\n\n".join(sections)

    query_words = set(query.lower().split())
    scored = []
    for i, sec in enumerate(sections):
        if len(sec.strip()) < 20:
            continue
        sec_words = set(sec.lower().split())
        overlap = len(query_words & sec_words)
        scored.append((overlap, i, sec))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [s[2] for s in scored[:top_k] if s[0] > 0]
    if not top:
        top = [sections[0]]  # fallback: return first section
    return "\n\n".join(top)


def hard_case_filter(candidates: list[dict], threshold: float = 0.15) -> bool:
    """Return True if candidates are a 'hard case' needing LLM.

    Hard case = rule scores are close (ambiguous).
    """
    from utils.reward import score_struct
    if len(candidates) < 2:
        return False
    # Quick heuristic: check if pair counts are similar
    pcs = [c.get("features", {}).get("pair_count", 0) for c in candidates]
    if max(pcs) - min(pcs) <= 2:
        return True
    # Check if canonical ratios are all high
    crs = [c.get("features", {}).get("canonical_ratio", 0) for c in candidates]
    if all(c > 0.9 for c in crs):
        return True
    return False


def _load_dotenv() -> None:
    """Load .env into os.environ if present (stdlib-only, no python-dotenv)."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and val and key not in os.environ:
            os.environ[key] = val


_load_dotenv()


# ---------------------------------------------------------------------------
# Feature weights for rule-based comparison
# ---------------------------------------------------------------------------
RULE_WEIGHTS: Dict[str, float] = {
    "pair_count": 0.3,
    "canonical_ratio": 0.25,
    "stem_continuity": 0.25,
    "pair_density": 0.1,
    "isolated_pairs": -0.1,
    "min_loop_violations": -0.15,
    "all_dot": -0.5,
}


def _rule_score(features: dict) -> float:
    """Weighted sum of normalized structural features."""
    score = 0.0
    for key, weight in RULE_WEIGHTS.items():
        value = features.get(key, 0.0)
        if isinstance(value, bool):
            value = 1.0 if value else 0.0
        score += float(value) * weight
    return score


def rule_judge(candidates: List[dict]) -> dict:
    """Heuristic preference: higher structural score wins."""
    if len(candidates) < 2:
        raise ValueError("Need at least 2 candidates for preference comparison.")

    scored = [(i, _rule_score(c["features"])) for i, c in enumerate(candidates)]
    scored.sort(key=lambda x: x[1], reverse=True)

    preferred = candidates[scored[0][0]]
    rejected = candidates[scored[-1][0]]

    return {
        "preferred": preferred["cid"],
        "rejected": rejected["cid"],
        "confidence": min(0.99, max(0.51, (scored[0][1] - scored[-1][1]) / max(1e-8, abs(scored[0][1])))),
        "reason": "rule_heuristic",
    }


def mock_judge(candidates: List[dict]) -> dict:
    """Random mock judge for control experiments."""
    if len(candidates) < 2:
        raise ValueError("Need at least 2 candidates for preference comparison.")
    rng = random.Random()
    idx_a, idx_b = rng.sample(range(len(candidates)), 2)
    return {
        "preferred": candidates[idx_a]["cid"],
        "rejected": candidates[idx_b]["cid"],
        "confidence": round(rng.uniform(0.5, 0.9), 2),
        "reason": "mock_random",
    }


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------
def _build_llm_prompt(seq: str, candidates: List[dict], evidence: str = "") -> str:
    """Build a minimal comparison prompt for a general-purpose LLM."""
    lines = [
        "You are an RNA structure critic. Compare the following candidate secondary structures",
        "for the given RNA sequence. Choose the better (more plausible) structure and the worse one.",
        "Respond ONLY with a JSON object:",
        '{"preferred": "c<id>", "rejected": "c<id>", "confidence": 0.XX, "reason": "short phrase"}',
    ]
    if evidence:
        lines += ["", "=== RNA Structure Knowledge (for reference) ===", evidence, ""]
    lines += [
        "",
        f"Sequence: {seq}",
        "Candidates:",
    ]
    for c in candidates:
        feats = c.get("features", {})
        lines.append(
            f"  {c['cid']}: struct={c['struct']} "
            f"pairs={feats.get('pair_count',0)} "
            f"canon={feats.get('canonical_ratio',0):.2f} "
            f"stem={feats.get('stem_continuity',0):.2f} "
            f"isolated={feats.get('isolated_pairs',0)}"
        )
    return "\n".join(lines)


def _call_llm_api(prompt: str, provider: str = "env") -> dict:
    """Call the general LLM API via .env credentials.

    Returns a parsed preference dict.  On failure, falls back to rule judge.
    """
    if provider == "mock":
        return {"preferred": "", "rejected": "", "confidence": 0.0, "reason": "mock_no_decision"}

    base_url = os.environ.get("LLM_BASE_URL", "")
    llm_token = os.environ.get("LLM_TOKEN", "") or os.environ.get("LLM_TOKEN", "")
    model = os.environ.get("LLM_MODEL", "")

    if not base_url or not llm_token or not model:
        return {"preferred": "", "rejected": "", "confidence": 0.0, "reason": "env_missing"}

    try:
        # stdlib-only HTTP call (no openai dependency)
        import urllib.request
        import urllib.error

        payload = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a precise RNA structure critic. Output valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 256,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {llm_token}",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return {"preferred": "", "rejected": "", "confidence": 0.0, "reason": f"api_error:{type(exc).__name__}"}

    try:
        content = body["choices"][0]["message"]["content"]
        # Extract JSON from response (handle markdown code blocks)
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:]) if len(lines) > 1 else content
            if content.endswith("```"):
                content = content[:-3]
        result = json.loads(content)
    except (json.JSONDecodeError, KeyError, IndexError):
        return {"preferred": "", "rejected": "", "confidence": 0.0, "reason": "llm_parse_error"}

    return {
        "preferred": str(result.get("preferred", "")),
        "rejected": str(result.get("rejected", "")),
        "confidence": float(result.get("confidence", 0.5)),
        "reason": str(result.get("reason", "llm")),
    }


def llm_judge(seq: str, candidates: List[dict], provider: str = "env", evidence: str = "") -> dict:
    """LLM-based preference comparison.

    On any failure, falls back to rule-based judgement.
    """
    prompt = _build_llm_prompt(seq, candidates, evidence)
    llm_result = _call_llm_api(prompt, provider)

    # Validate LLM output
    cids = {c["cid"] for c in candidates}
    pref = llm_result.get("preferred", "")
    rej = llm_result.get("rejected", "")
    confidence = float(llm_result.get("confidence", 0.0))

    if pref not in cids or rej not in cids or pref == rej or confidence <= 0.0 or confidence > 1.0:
        # Fallback to rule
        rule_result = rule_judge(candidates)
        rule_result["source"] = "rule_fallback"
        rule_result["fallback_reason"] = llm_result.get("reason", "llm_invalid_output")
        return rule_result

    return {
        "preferred": pref,
        "rejected": rej,
        "confidence": min(1.0, max(0.0, confidence)),
        "reason": llm_result.get("reason", "llm"),
        "source": "llm",
    }


# ---------------------------------------------------------------------------
# Buffer output helpers
# ---------------------------------------------------------------------------
def _candidate_by_cid(candidates: List[dict], cid: str) -> dict:
    for c in candidates:
        if c["cid"] == cid:
            return c
    raise KeyError(f"cid {cid!r} not found in candidates")


def build_entry(sample: dict, judge_result: dict, source: str) -> dict:
    """Build a preference-buffer JSONL entry."""
    candidates = sample["candidates"]
    preferred_c = _candidate_by_cid(candidates, judge_result["preferred"])
    rejected_c = _candidate_by_cid(candidates, judge_result["rejected"])

    return {
        "id": sample["id"],
        "preferred_struct": preferred_c["struct"],
        "rejected_struct": rejected_c["struct"],
        "preferred_pairs": preferred_c.get("pairs", []),
        "rejected_pairs": rejected_c.get("pairs", []),
        "confidence": judge_result.get("confidence", 0.5),
        "source": source,
    }


def shuffle_buffer(source_path: str, seed: int = 43) -> list[dict]:
    """Shuffle preferred/rejected assignments across samples.

    Reads a preference buffer, randomly reassigns preferred/rejected pairs
    to different sample IDs, preserving the buffer structure but breaking
    any genuine sample-level correspondence.
    """
    import random as _random
    rng = _random.Random(seed)

    entries = []
    with Path(source_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))

    if len(entries) < 2:
        return entries

    # Extract all (preferred_pairs, rejected_pairs, confidence) tuples
    pair_sets = [(e.get("preferred_pairs", []), e.get("rejected_pairs", []), e.get("confidence", 0.5)) for e in entries]
    ids = [e["id"] for e in entries]
    rng.shuffle(pair_sets)

    shuffled = []
    for i, e in enumerate(entries):
        pref_pairs, rej_pairs, conf = pair_sets[i]
        shuffled.append({
            "id": e["id"],
            "preferred_struct": "",
            "rejected_struct": "",
            "preferred_pairs": pref_pairs,
            "rejected_pairs": rej_pairs,
            "confidence": conf,
            "source": "shuffled",
        })
    return shuffled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preference judge for RNA structure candidates"
    )
    parser.add_argument("--input", required=True, help="Candidate JSONL file")
    parser.add_argument("--out", required=True, help="Output preference-buffer JSONL path")
    parser.add_argument("--mode", required=True, choices=["rule", "llm", "mock", "shuffle"])
    parser.add_argument("--provider", default="env", choices=["env", "mock"])
    parser.add_argument("--limit", type=int, default=0, help="Max samples (0 = all)")
    parser.add_argument("--source_buffer", help="Source buffer for shuffle mode")
    parser.add_argument("--seed", type=int, default=43, help="Seed for mock/shuffle")
    parser.add_argument("--rag", help="Path to knowledge base text file for RAG retrieval")
    parser.add_argument("--hard-only", action="store_true", help="Only send hard cases to LLM")
    parser.add_argument("--hard-threshold", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Error: input file not found: {in_path}", file=sys.stderr)
        raise SystemExit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load knowledge base for RAG
    kb_text = ""
    if args.rag:
        kb_text = load_kb(args.rag)
        print(f"KB loaded: {len(kb_text)} chars from {args.rag}")

    count = 0
    hard_count = 0
    rule_count = 0
    with in_path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            candidates = sample.get("candidates", [])
            if len(candidates) < 2:
                continue

            seq = sample["seq"]

            if args.mode == "rule":
                judge_result = rule_judge(candidates)
                source = "rule"
                rule_count += 1
            elif args.mode == "mock":
                judge_result = mock_judge(candidates)
                source = "mock"
            elif args.mode == "shuffle":
                pass
            else:  # llm mode
                # Hard-case filter: use rule for easy cases
                if args.hard_only and not hard_case_filter(candidates, args.hard_threshold):
                    judge_result = rule_judge(candidates)
                    source = "rule_hard_filter"
                    rule_count += 1
                else:
                    # Retrieve evidence if RAG enabled
                    evidence = ""
                    if kb_text:
                        evidence = retrieve_evidence(seq, kb_text)
                    judge_result = llm_judge(seq, candidates, provider=args.provider, evidence=evidence)
                    source = judge_result.get("source", "llm")
                    if args.hard_only:
                        hard_count += 1

            if args.mode != "shuffle":
                entry = build_entry(sample, judge_result, source)
                dst.write(json.dumps(entry) + "\n")
                count += 1
            if args.limit and count >= args.limit:
                break

        if args.mode == "shuffle":
            src = args.source_buffer
            if not src:
                print("Error: --source_buffer required for shuffle mode", file=sys.stderr)
                raise SystemExit(1)
            shuffled = shuffle_buffer(src, seed=args.seed)
            for entry in shuffled[:args.limit] if args.limit else shuffled:
                dst.write(json.dumps(entry) + "\n")
                count += 1

    print(f"Preference buffer written to {out_path}")
    print(f"  entries: {count}")
    print(f"  mode: {args.mode}  provider: {args.provider}")
    if args.hard_only:
        print(f"  hard_cases(LLM): {hard_count}  rule_filtered: {rule_count}")
    if args.rag:
        print(f"  RAG: {args.rag}")


if __name__ == "__main__":
    main()
