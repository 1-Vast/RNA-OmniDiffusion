"""Typed Data Training Spec -- multi-level lookup with diagnostics.

Supports lookup by: sample_id → family → source → default.
Tracks coverage, match rates, and source distribution for diagnostics.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_SPEC: Dict[str, Any] = {
    "source": "unknown", "family": None, "has_structure": True,
    "stage": "adapt", "task": "pair_relation",
    "mask": {"mode": "random", "ratio": 0.15, "span": 3},
    "sample": {"weight": 1.0, "curriculum": 2},
    "relation": {"use_msmprm": False, "focus": "mixed", "hard_negative": "canonical_nonpair"},
    "confidence": 1.0,
}

VALID_SOURCES = {"rnacentral", "rfam", "bprna", "archiveii", "mixed", "unknown"}
VALID_TASKS = {"seq_denoise", "pair_relation", "mixed"}
VALID_MASK_MODES = {"random", "span"}
VALID_FOCUS = {"global", "stem_local", "hard_negative", "mixed"}
VALID_HARD_NEG = {"random", "canonical_nonpair", "near_diagonal", "medium_long"}


def _validate_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = deepcopy(DEFAULT_SPEC)
    if not isinstance(spec, dict):
        return cleaned
    cleaned["source"] = spec.get("source", "unknown")
    if cleaned["source"] not in VALID_SOURCES:
        cleaned["source"] = "unknown"
    cleaned["family"] = spec.get("family")
    cleaned["has_structure"] = bool(spec.get("has_structure", True))
    cleaned["stage"] = spec.get("stage", "adapt")
    cleaned["task"] = spec.get("task", "pair_relation")
    if cleaned["task"] not in VALID_TASKS:
        cleaned["task"] = "pair_relation"
    mask = spec.get("mask", {})
    if isinstance(mask, dict):
        cleaned["mask"]["mode"] = mask.get("mode", "random")
        if cleaned["mask"]["mode"] not in VALID_MASK_MODES:
            cleaned["mask"]["mode"] = "random"
        cleaned["mask"]["ratio"] = max(0.05, min(0.30, float(mask.get("ratio", 0.15))))
        cleaned["mask"]["span"] = max(1, min(12, int(mask.get("span", 3))))
    samp = spec.get("sample", {})
    if isinstance(samp, dict):
        cleaned["sample"]["weight"] = max(0.5, min(2.0, float(samp.get("weight", 1.0))))
        cleaned["sample"]["curriculum"] = max(1, min(3, int(samp.get("curriculum", 2))))
    rel = spec.get("relation", {})
    if isinstance(rel, dict):
        cleaned["relation"]["use_msmprm"] = bool(rel.get("use_msmprm", False))
        cleaned["relation"]["focus"] = rel.get("focus", "mixed")
        if cleaned["relation"]["focus"] not in VALID_FOCUS:
            cleaned["relation"]["focus"] = "mixed"
        cleaned["relation"]["hard_negative"] = rel.get("hard_negative", "canonical_nonpair")
        if cleaned["relation"]["hard_negative"] not in VALID_HARD_NEG:
            cleaned["relation"]["hard_negative"] = "canonical_nonpair"
    cleaned["confidence"] = max(0.0, min(1.0, float(spec.get("confidence", 0.5))))
    return cleaned


class TypedSpec:
    """Multi-level spec lookup: sample_id → family → source → default.

    Diagnostics track:
      - total hits, misses, hits_by_level
      - source distribution
      - avg mask parameters
    """

    def __init__(self, path: Optional[str | Path] = None) -> None:
        self._by_id: Dict[str, Dict[str, Any]] = {}
        self._by_family: Dict[str, Dict[str, Any]] = {}
        self._by_source: Dict[str, Dict[str, Any]] = {}
        self._source_counts: Dict[str, int] = {}
        # diagnostics
        self.diag_hits: int = 0
        self.diag_misses: int = 0
        self.diag_hit_by_id: int = 0
        self.diag_hit_by_family: int = 0
        self.diag_hit_by_source: int = 0
        self.diag_hit_default: int = 0
        self._loaded_path: str = ""
        self._loaded_entries: int = 0
        if path:
            self.load(path)

    def load(self, path: str | Path) -> int:
        buf_path = Path(path)
        if not buf_path.exists():
            return 0
        count = 0
        with buf_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                spec = _validate_spec(entry.get("spec", {}))
                sid = str(entry.get("id", ""))
                family = str(entry.get("family", "")) if entry.get("family") else ""
                source = str(entry.get("source", spec.get("source", "unknown")))

                # Index by id (always)
                if sid:
                    self._by_id[sid] = spec
                # Index by family
                if family:
                    self._by_family[family] = spec
                # Index by source
                if source and source != "unknown":
                    self._by_source[source] = spec

                self._source_counts[source] = self._source_counts.get(source, 0) + 1
                count += 1

        self._loaded_path = str(path)
        self._loaded_entries = count
        return count

    def get(self, sample_id: str, family: str = "", source: str = "") -> Dict[str, Any]:
        """Lookup spec by sample_id → family → source → default."""
        sid = str(sample_id)
        if sid and sid in self._by_id:
            self.diag_hits += 1
            self.diag_hit_by_id += 1
            return self._by_id[sid]
        fam = str(family)
        if fam and fam in self._by_family:
            self.diag_hits += 1
            self.diag_hit_by_family += 1
            return self._by_family[fam]
        src = str(source)
        if src and src in self._by_source:
            self.diag_hits += 1
            self.diag_hit_by_source += 1
            return self._by_source[src]
        self.diag_misses += 1
        self.diag_hit_default += 1
        return deepcopy(DEFAULT_SPEC)

    def batch_specs(self, sample_ids: List[str], families: List[str] = None, sources: List[str] = None) -> List[Dict[str, Any]]:
        fams = families or [""] * len(sample_ids)
        srcs = sources or [""] * len(sample_ids)
        return [self.get(sid, fam, src) for sid, fam, src in zip(sample_ids, fams, srcs)]

    def reset_diagnostics(self) -> None:
        self.diag_hits = 0
        self.diag_misses = 0
        self.diag_hit_by_id = 0
        self.diag_hit_by_family = 0
        self.diag_hit_by_source = 0
        self.diag_hit_default = 0

    def coverage_report(self) -> Dict[str, Any]:
        total = self.diag_hits + self.diag_misses
        return {
            "total_lookups": total,
            "hits": self.diag_hits,
            "misses": self.diag_misses,
            "coverage": self.diag_hits / max(1, total),
            "hit_by_id": self.diag_hit_by_id,
            "hit_by_family": self.diag_hit_by_family,
            "hit_by_source": self.diag_hit_by_source,
            "hit_default": self.diag_hit_default,
            "loaded_entries": self._loaded_entries,
            "loaded_path": self._loaded_path,
            "source_distribution": dict(self._source_counts),
        }

    @property
    def coverage(self) -> float:
        total = self.diag_hits + self.diag_misses
        return self.diag_hits / max(1, total)

    def __len__(self) -> int:
        return len(self._by_id)

    def __bool__(self) -> bool:
        return len(self._by_id) > 0


# ---- Coach Policy loading ----

def load_policy(path: str | Path) -> dict | None:
    """Load a JSON coach policy file. Returns None if missing or invalid."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def check_match(data_path: str, spec_path: str) -> dict:
    """Check how many data sample IDs match spec IDs."""
    data_ids = set()
    with Path(data_path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            sid = str(obj.get("id", ""))
            if sid: data_ids.add(sid)

    spec_ids = set()
    with Path(spec_path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            sid = str(obj.get("id", ""))
            if sid: spec_ids.add(sid)

    matched = data_ids & spec_ids
    return {
        "data_count": len(data_ids),
        "spec_count": len(spec_ids),
        "matched_count": len(matched),
        "coverage": len(matched) / max(1, len(data_ids)),
        "data_only": len(data_ids - spec_ids),
        "spec_only": len(spec_ids - data_ids),
    }
