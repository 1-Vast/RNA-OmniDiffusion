"""Energy-inspired pair priors for RNA secondary structure decoding.

All functions operate on numpy arrays (not torch tensors) to keep a clean
boundary between model logits and biologically-motivated priors.  Use
``-inf`` for forbidden pairs; the canonical filter inside ``nussinov_decode``
will never examine them during DP.

Performance note: the double loops are O(L²) which is acceptable for
RNA lengths < 500; typical ArchiveII validation sequences average ~115 nt.
"""

from __future__ import annotations

import numpy as np

from utils.struct import canonical_pair


# ---------------------------------------------------------------------------
# individual prior builders
# ---------------------------------------------------------------------------


def canonical_type_prior(seq: str, L: int) -> np.ndarray:
    """Return L×L canonical-pair prior matrix.

    Scores (upper triangle only, i < j):
        GC / CG  → +1.0
        AU / UA  → +0.6
        GU / UG  → +0.3  (wobble)
        other    → -∞
    """
    mat: np.ndarray = np.full((L, L), -np.inf, dtype=np.float32)
    _score = {
        ("G", "C"): 1.0,
        ("C", "G"): 1.0,
        ("A", "U"): 0.6,
        ("U", "A"): 0.6,
        ("G", "U"): 0.3,
        ("U", "G"): 0.3,
    }
    for i in range(L):
        bi = seq[i]
        for j in range(i + 1, L):
            key = (bi, seq[j])
            if key in _score:
                mat[i, j] = _score[key]
    return mat


def loop_penalty_prior(L: int, min_loop: int = 3) -> np.ndarray:
    """Return L×L matrix with -∞ for hairpin-loop-violating close pairs.

    Pairs where ``|i - j| <= min_loop`` are forbidden (minimum hairpin
    loop constraint from Turner energy rules).
    """
    mat: np.ndarray = np.zeros((L, L), dtype=np.float32)
    for i in range(L):
        end = min(L, i + int(min_loop) + 1)
        mat[i, i + 1 : end] = -np.inf
    return mat


def distance_prior(L: int, alpha: float = 0.01) -> np.ndarray:
    """Return L×L matrix with a logarithmic distance penalty.

    ``score = -alpha * log(1 + |i - j|)`` so long-range pairs get a
    slightly lower prior, reflecting the entropic cost of large loops.
    """
    mat: np.ndarray = np.zeros((L, L), dtype=np.float32)
    for i in range(L):
        for j in range(i + 1, L):
            mat[i, j] = -alpha * np.log(1.0 + (j - i))
    return mat


def stem_continuity_prior(
    pair_logits: "np.ndarray | object",
    seq: str,
    L: int,
    bonus_weight: float = 0.1,
) -> np.ndarray:
    """Small bonus for pairs whose anti-diagonal neighbours score high.

    For each canonical pair *(i, j)*, inspect the two anti-diagonal
    neighbours *(i-1, j+1)* and *(i+1, j-1)*.  If either neighbour has a
    high probability under the model logits, *(i, j)* receives a bonus,
    encouraging contiguous stems.

    This is an eval-only helper – it reads model **logits**, not ground
    truth pairs.
    """
    # Accept torch Tensors transparently
    if hasattr(pair_logits, "detach"):
        logits: np.ndarray = pair_logits.detach().float().cpu().numpy()
    else:
        logits = np.asarray(pair_logits, dtype=np.float32)
    logits = logits[:L, :L]

    bonus: np.ndarray = np.zeros((L, L), dtype=np.float32)

    def _safe_sigmoid(x: float) -> float:
        z = float(np.clip(x, -50.0, 50.0))
        return float(1.0 / (1.0 + np.exp(-z)))

    for i in range(L):
        for j in range(i + 1, L):
            # Only bonus canonical pairs
            if not canonical_pair(seq[i], seq[j], allow_wobble=True):
                continue
            neighbour_score = 0.0
            if i > 0 and j + 1 < L:
                neighbour_score += _safe_sigmoid(logits[i - 1, j + 1])
            if i + 1 < L and j > 0:
                neighbour_score += _safe_sigmoid(logits[i + 1, j - 1])
            if neighbour_score > 0:
                bonus[i, j] = bonus_weight * neighbour_score
    return bonus


# ---------------------------------------------------------------------------
# combined builder
# ---------------------------------------------------------------------------


def build_pair_prior_matrix(
    seq: str,
    canonical_weight: float = 1.0,
    loop_penalty: bool = True,
    distance_alpha: float = 0.0,
    stem_bonus: bool = False,
) -> np.ndarray:
    """Combine energy-inspired priors into a single L×L float matrix.

    Parameters
    ----------
    seq:
        RNA sequence string (uppercase, T→U already done by caller).
    canonical_weight:
        Scale for the canonical-type prior.  0 disables it.
    loop_penalty:
        Apply the hairpin-loop minimum-distance filter.
    distance_alpha:
        Logarithmic distance penalty coefficient (0 = disabled).
    stem_bonus:
        Reserved; stem continuity needs model logits and is applied
        separately during evaluation.  This flag is silently ignored.

    Returns
    -------
    np.ndarray  shape (L, L), dtype float32
    """
    L: int = len(seq)
    if L == 0:
        return np.zeros((0, 0), dtype=np.float32)

    prior: np.ndarray = np.zeros((L, L), dtype=np.float32)

    if canonical_weight > 0:
        prior = prior + canonical_weight * canonical_type_prior(seq, L)

    if loop_penalty:
        prior = prior + loop_penalty_prior(L)

    if distance_alpha > 0:
        prior = prior + distance_prior(L, alpha=distance_alpha)

    # stem_bonus is applied separately via stem_continuity_prior() during
    # evaluation where model logits are available.
    _ = stem_bonus

    return prior
