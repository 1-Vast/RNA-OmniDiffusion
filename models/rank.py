"""Local pair ranking loss for RNA secondary structure training.

For each true positive pair (i,j), samples hard negatives from four pools:
  1. Same row i, different column j
  2. Same column j, different row i
  3. Same distance bucket, different pair
  4. Canonical (GC/AU/GU) base-pair, but false pair

Loss = max(0, margin - logit_pos + logit_neg), averaged over all samples.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn.functional as F


def _dist_bucket(dist: torch.Tensor) -> torch.Tensor:
    """Map pairwise distances to bucket indices [0..4]."""
    buckets = torch.zeros_like(dist, dtype=torch.long)
    buckets = torch.where(dist <= 4,  dist.new_tensor(0), buckets)
    buckets = torch.where((dist >= 5)  & (dist <= 10), dist.new_tensor(1), buckets)
    buckets = torch.where((dist >= 11) & (dist <= 30), dist.new_tensor(2), buckets)
    buckets = torch.where((dist >= 31) & (dist <= 80), dist.new_tensor(3), buckets)
    buckets = torch.where(dist >= 81, dist.new_tensor(4), buckets)
    return buckets


def _canonical_mask(seq: str, device: torch.device) -> torch.Tensor:
    """Return (L, L) boolean mask where (i,j) forms a canonical pair."""
    base_to_id = {"A": 0, "U": 1, "G": 2, "C": 3, "N": -1}
    canonical_table = torch.zeros(4, 4, dtype=torch.bool, device=device)
    for b1, b2 in [("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")]:
        canonical_table[base_to_id[b1], base_to_id[b2]] = True
    seq = seq.upper().replace("T", "U")
    L = len(seq)
    bids_list = [base_to_id.get(c, -1) for c in seq]
    bids = torch.tensor(bids_list, dtype=torch.long, device=device)
    valid = bids >= 0
    bi = bids.unsqueeze(1).expand(L, L)
    bj = bids.unsqueeze(0).expand(L, L)
    return valid.unsqueeze(1) & valid.unsqueeze(0) & canonical_table[bi.clamp(0, 3), bj.clamp(0, 3)]


def local_rank_loss(
    pair_logits: torch.Tensor,
    pair_labels: torch.Tensor,
    seqs: Sequence[str],
    margin: float = 1.0,
    negatives_per_positive: int = 4,
) -> torch.Tensor:
    """Local pair ranking loss with hard negative mining.

    Args:
        pair_logits:  (B, L, L) logit matrix.
        pair_labels:  (B, L, L) ground-truth (1.0 = pair, 0.0 = no pair).
        seqs:         RNA sequences, length-B list of strings.
        margin:       Hinge margin for max(0, margin - pos + neg).
        negatives_per_positive: target number of negatives per positive.

    Returns:
        Scalar loss tensor (0.0 if no positives found).
    """
    B, L, _ = pair_logits.shape
    device = pair_logits.device
    n_per_type = max(1, negatives_per_positive // 4)

    # Pre-compute shared tensors
    idx = torch.arange(L, device=device)
    dist_grid = (idx[:, None] - idx[None, :]).abs()
    buckets_grid = _dist_bucket(dist_grid)  # (L, L)

    # min_loop = 4 for upper triangle
    base_triu = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=4)

    all_pos: List[torch.Tensor] = []
    all_neg: List[torch.Tensor] = []

    for b in range(B):
        seq = seqs[b].upper().replace("T", "U")
        slen = len(seq)
        if slen < 2:
            continue

        logits = pair_logits[b, :slen, :slen]
        labels = pair_labels[b, :slen, :slen]

        triu_mask = base_triu[:slen, :slen]
        pos_mask = triu_mask & (labels > 0.5)
        pos_indices = torch.nonzero(pos_mask, as_tuple=False)  # (N, 2)

        if pos_indices.numel() == 0:
            continue

        # Negative pool: all valid upper-triangle that are NOT positive
        neg_pool = triu_mask & (labels <= 0.5)

        # Preparations for type 3 & 4
        buckets = buckets_grid[:slen, :slen]
        canonical = _canonical_mask(seq, device)[:slen, :slen]
        row_arange = torch.arange(slen, device=device)

        for pi in range(pos_indices.shape[0]):
            i = int(pos_indices[pi, 0].item())
            j = int(pos_indices[pi, 1].item())
            pos_logit = logits[i, j]

            # Type 1: same row i, different column
            type1_mask = neg_pool & (row_arange == i).unsqueeze(1)
            _sample_neg(logits[i], type1_mask, n_per_type, all_pos, all_neg, pos_logit, device)

            # Type 2: same column j, different row
            type2_mask = neg_pool & (row_arange == j).unsqueeze(0)
            _sample_neg(logits[:, j], type2_mask, n_per_type, all_pos, all_neg, pos_logit, device)

            # Type 3: same distance bucket
            type3_mask = neg_pool & (buckets == buckets[i, j])
            _sample_neg_matrix(logits, type3_mask, n_per_type, all_pos, all_neg, pos_logit, device)

            # Type 4: canonical but false
            type4_mask = neg_pool & canonical
            _sample_neg_matrix(logits, type4_mask, n_per_type, all_pos, all_neg, pos_logit, device)

    if not all_pos:
        return torch.tensor(0.0, device=device)

    pos_t = torch.stack(all_pos)
    neg_t = torch.stack(all_neg)
    loss = F.relu(margin - pos_t + neg_t).mean()
    return loss


def _sample_neg(
    logit_row: torch.Tensor,
    mask: torch.Tensor,
    n: int,
    all_pos: list,
    all_neg: list,
    pos_logit: torch.Tensor,
    device: torch.device,
) -> None:
    """Sample negatives from a mask; pick column positions (type 1: same row)."""
    cand = torch.nonzero(mask, as_tuple=False)  # (K, 2)
    if cand.numel() == 0:
        return
    n_sample = min(n, cand.shape[0])
    perm = torch.randperm(cand.shape[0], device=device)[:n_sample]
    for idx_p in range(n_sample):
        ii, jj = int(cand[perm[idx_p], 0].item()), int(cand[perm[idx_p], 1].item())
        all_pos.append(pos_logit)
        all_neg.append(logit_row[jj])


def _sample_neg_matrix(
    logits: torch.Tensor,
    mask: torch.Tensor,
    n: int,
    all_pos: list,
    all_neg: list,
    pos_logit: torch.Tensor,
    device: torch.device,
) -> None:
    """Sample negatives from a 2D mask (type 3/4: general pairs)."""
    cand = torch.nonzero(mask, as_tuple=False)  # (K, 2)
    if cand.numel() == 0:
        return
    n_sample = min(n, cand.shape[0])
    perm = torch.randperm(cand.shape[0], device=device)[:n_sample]
    for idx_p in range(n_sample):
        ii, jj = int(cand[perm[idx_p], 0].item()), int(cand[perm[idx_p], 1].item())
        all_pos.append(pos_logit)
        all_neg.append(logits[ii, jj])
