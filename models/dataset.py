from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List

from utils.struct import ALLOWED_STRUCT_CHARS, infer_simple_motifs, parse_dot_bracket


class RNAOmniDataset:
    """JSONL dataset for unified RNA sequence, structure, motif, and family modeling."""

    def __init__(
        self,
        jsonl_path: str | Path,
        max_length: int | None = None,
        strict: bool = False,
        allow_unlabeled: bool = False,
    ) -> None:
        self.path = Path(jsonl_path)
        self.max_length = max_length
        self.strict = strict
        self.allow_unlabeled = allow_unlabeled
        if not self.path.exists():
            raise FileNotFoundError(f"RNA JSONL file does not exist: {self.path}")
        self.samples = self._load()
        if not self.samples:
            raise ValueError(f"No valid RNA samples found in {self.path}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]

    def _warn_or_raise(self, message: str) -> None:
        if self.strict:
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning)

    def _load(self) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSON at {self.path}:{line_no}: {exc}") from exc
                try:
                    sample = self._normalize(raw, line_no)
                except ValueError as exc:
                    self._warn_or_raise(str(exc))
                    continue
                samples.append(sample)
        return samples

    def _normalize(self, raw: Dict[str, Any], line_no: int) -> Dict[str, Any]:
        if "seq" not in raw and "sequence" not in raw:
            raise ValueError(f"{self.path}:{line_no} must contain 'seq' or 'sequence'.")
        if "struct" not in raw and not self.allow_unlabeled:
            raise ValueError(f"{self.path}:{line_no} must contain 'struct' unless allow_unlabeled=True.")
        # Support sequence-only JSONL for Stage 1 pretraining (no struct/pairs required)

        seq = str(raw.get("seq", raw.get("sequence"))).upper().replace("T", "U")
        seq = "".join(base if base in {"A", "U", "G", "C", "N"} else "N" for base in seq)
        is_labeled = "struct" in raw and raw.get("struct") is not None
        struct = str(raw["struct"]) if is_labeled else "." * len(seq)
        struct = "".join(char if char in ALLOWED_STRUCT_CHARS else "." for char in struct)

        if len(seq) != len(struct):
            raise ValueError(
                f"{self.path}:{line_no} sequence/structure length mismatch: "
                f"len(seq)={len(seq)} len(struct)={len(struct)}."
            )
        if self.max_length is not None and len(seq) > self.max_length:
            raise ValueError(
                f"{self.path}:{line_no} length {len(seq)} exceeds max_length {self.max_length}."
            )

        try:
            parsed_pairs = parse_dot_bracket(struct)
        except ValueError as exc:
            raise ValueError(f"{self.path}:{line_no} invalid dot-bracket structure: {exc}") from exc

        pairs = self._normalize_pairs(raw.get("pairs"), len(seq), parsed_pairs)
        if raw.get("pairs") is not None and set(pairs) != set(parsed_pairs):
            self._warn_or_raise(
                f"{self.path}:{line_no} provided pairs do not match struct; using pairs parsed from struct."
            )
            pairs = parsed_pairs
        motifs = raw.get("motifs")
        if not motifs:
            motifs = infer_simple_motifs(seq=seq, struct=struct, pairs=pairs)
        else:
            motifs = self._normalize_motifs(motifs, len(seq), line_no)

        struct_tags = raw.get("struct_tags")
        if isinstance(struct_tags, dict):
            struct_tags = {
                "numeric": struct_tags.get("numeric"),
                "categorical": struct_tags.get("categorical"),
            }

        sample = {
            "id": str(raw.get("id", f"{self.path.stem}_{line_no:06d}")),
            "seq": seq,
            "struct": struct,
            "family": str(raw.get("family", "OTHER") or "OTHER"),
            "motifs": motifs,
            "pairs": pairs,
            "length": len(seq),
            "is_labeled": bool(is_labeled),
            "_struct_tags": struct_tags,
        }
        if "_weight" in raw:
            sample["_weight"] = float(raw["_weight"])
        return sample

    def _normalize_pairs(
        self,
        raw_pairs: Iterable[Iterable[int]] | None,
        length: int,
        parsed_pairs: List[tuple[int, int]],
    ) -> List[tuple[int, int]]:
        if raw_pairs is None:
            return parsed_pairs
        pairs: List[tuple[int, int]] = []
        seen = set()
        for raw_pair in raw_pairs:
            pair = list(raw_pair)
            if len(pair) != 2:
                continue
            i, j = int(pair[0]), int(pair[1])
            if i > j:
                i, j = j, i
            if i < 0 or j >= length or i == j:
                continue
            if i in seen or j in seen:
                continue
            seen.add(i)
            seen.add(j)
            pairs.append((i, j))
        return sorted(pairs)

    def _normalize_motifs(self, raw_motifs: Iterable[Dict[str, Any]], length: int, line_no: int) -> List[dict]:
        motifs = []
        for motif in raw_motifs:
            try:
                start = int(motif["start"])
                end = int(motif["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if start > end:
                start, end = end, start
            if start < 0 or end >= length:
                self._warn_or_raise(
                    f"{self.path}:{line_no} motif span {(start, end)} is outside length {length}; skipping."
                )
                continue
            motifs.append(
                {
                    "type": str(motif.get("type", "MULTI_LOOP")).upper(),
                    "start": start,
                    "end": end,
                }
            )
        return motifs or infer_simple_motifs(seq="N" * length, pairs=[])

