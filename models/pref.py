"""Preference-ranking loss for pair-level structure comparison.

Converts a preference buffer (preferred vs rejected candidate pairs)
into a differentiable ranking objective.  LLM results are treated as
weak pairwise preference signals, not strong supervised labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

Pair = Tuple[int, int]


def pref_ranking_loss(
    pair_logits: torch.Tensor,
    preferred_pairs: Sequence[Pair],
    rejected_pairs: Sequence[Pair],
) -> torch.Tensor | None:
    """Compute soft ranking loss for one sample.

    Parameters
    ----------
    pair_logits : (L, L) tensor
        Raw pair logits from the model.
    preferred_pairs : list of (i, j)
        Pairs from the preferred candidate (winner).
    rejected_pairs : list of (i, j)
        Pairs from the rejected candidate (loser).

    Returns
    -------
    loss : scalar tensor or None
        Per-pair BCE loss over difference-set pairs.
        Preferred-only pairs get label=1, rejected-only pairs get label=0.
        Returns None when the pair difference set is empty.
    """
    pref_set = {tuple(p) for p in preferred_pairs}
    rej_set = {tuple(p) for p in rejected_pairs}

    only_good = pref_set - rej_set
    only_bad = rej_set - pref_set

    if not only_good or not only_bad:
        return None

    good_indices = torch.tensor(
        [(i, j) for i, j in only_good if i < pair_logits.size(0) and j < pair_logits.size(1)],
        dtype=torch.long,
        device=pair_logits.device,
    )
    bad_indices = torch.tensor(
        [(i, j) for i, j in only_bad if i < pair_logits.size(0) and j < pair_logits.size(1)],
        dtype=torch.long,
        device=pair_logits.device,
    )

    if good_indices.numel() == 0 or bad_indices.numel() == 0:
        return None

    # Per-pair BCE: each preferred-only pair should have high logit,
    # each rejected-only pair should have low logit.
    good_logits = pair_logits[good_indices[:, 0], good_indices[:, 1]]
    bad_logits = pair_logits[bad_indices[:, 0], bad_indices[:, 1]]
    all_logits = torch.cat([good_logits, bad_logits])
    all_labels = torch.cat([
        torch.ones_like(good_logits),
        torch.zeros_like(bad_logits),
    ])
    return F.binary_cross_entropy_with_logits(all_logits, all_labels)


def load_buffer(path: str | Path) -> List[dict]:
    """Load a JSONL preference buffer.

    Each line is a JSON object with keys:
      id, preferred_pairs, rejected_pairs, confidence, source
    """
    buf_path = Path(path)
    if not buf_path.exists():
        raise FileNotFoundError(f"Preference buffer not found: {buf_path}")
    buffer: List[dict] = []
    with buf_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            buffer.append(json.loads(line))
    return buffer


def build_lookup(buffer: List[dict], min_confidence: float = 0.6) -> Dict[str, dict]:
    """Build a {sample_id: preference_entry} dictionary.

    Filters entries with confidence below *min_confidence*.
    """
    lookup: Dict[str, dict] = {}
    for entry in buffer:
        if float(entry.get("confidence", 0.0)) < min_confidence:
            continue
        sample_id = str(entry.get("id", ""))
        if not sample_id:
            continue
        lookup[sample_id] = entry
    return lookup


def compute_batch_pref_loss(
    pair_logits: torch.Tensor,
    sample_ids: Sequence[str],
    lengths: torch.Tensor,
    pref_lookup: Dict[str, dict],
) -> Tuple[torch.Tensor, int]:
    """Compute mean preference loss over a batch.

    Returns
    -------
    (loss, covered) : (scalar, int)
        *loss* is the mean ranking loss across covered samples.
        *covered* is the count of samples that had a valid preference pair.
    """
    batch_size = pair_logits.size(0)
    losses: List[torch.Tensor] = []
    covered = 0

    for idx in range(batch_size):
        sid = str(sample_ids[idx]) if idx < len(sample_ids) else ""
        entry = pref_lookup.get(sid)
        if entry is None:
            continue
        length = int(lengths[idx].item())
        logits = pair_logits[idx, :length, :length]
        loss = pref_ranking_loss(
            logits,
            entry.get("preferred_pairs", []),
            entry.get("rejected_pairs", []),
        )
        if loss is not None and torch.isfinite(loss):
            losses.append(loss)
            covered += 1

    if not losses:
        return torch.tensor(0.0, device=pair_logits.device), 0

    mean_loss = torch.stack(losses).mean()
    return mean_loss, covered
