from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from models.token import RNAOmniTokenizer
from utils.struct import canonical_pair, parse_dot_bracket, pairs_to_dot_bracket


GREEDY_DECODE_WARNING = (
    "Greedy decoding is approximate and may allow pseudoknot-like crossing pairs "
    "before dot-bracket conversion. Use Nussinov for strict non-crossing decoding."
)


def _forward_model(model: torch.nn.Module, batch: dict) -> dict:
    return model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        segment_ids=batch["segment_ids"],
        task_ids=batch["task_ids"],
        time_steps=batch["time_steps"],
        seq_positions=batch.get("seq_positions"),
    )


def _build_inference_batch(
    tokenizer: RNAOmniTokenizer,
    task_name: str,
    seq: str,
    struct: str,
    family: str | None = None,
    motifs: Sequence[dict] | None = None,
    device: torch.device | str = "cpu",
) -> tuple[dict, List[int], List[int]]:
    tokens: List[str] = []
    segment_ids: List[int] = []
    seq_positions: List[int] = []
    struct_positions: List[int] = []

    def add(token: str, segment_id: int) -> int:
        tokens.append(token)
        segment_ids.append(segment_id)
        return len(tokens) - 1

    add(tokenizer.task_token(task_name), 0)
    if task_name == "motif_control":
        add("<FAMILY>", 3)
        add(tokenizer.family_token(family), 3)
        add("</FAMILY>", 3)
        add("<MOTIF>", 3)
        for motif in motifs or []:
            add(tokenizer.motif_token(motif.get("type")), 3)
        add("</MOTIF>", 3)

    add("<SEQ>", 1)
    for base in seq:
        seq_positions.append(add(base, 1))
    add("</SEQ>", 1)
    add("<STRUCT>", 2)
    for char in struct:
        struct_positions.append(add(char, 2))
    add("</STRUCT>", 2)

    input_ids = torch.tensor([tokenizer.encode(tokens)], dtype=torch.long, device=device)
    batch = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "segment_ids": torch.tensor([segment_ids], dtype=torch.long, device=device),
        "task_ids": torch.tensor([tokenizer.task_to_id[task_name]], dtype=torch.long, device=device),
        "time_steps": torch.ones(1, dtype=torch.float32, device=device),
        "seq_positions": torch.tensor([seq_positions], dtype=torch.long, device=device),
        "lengths": torch.tensor([len(seq)], dtype=torch.long, device=device),
    }
    return batch, seq_positions, struct_positions


@torch.no_grad()
def entropy_iterative_unmask(
    model: torch.nn.Module,
    batch: dict,
    target_positions: Sequence[int],
    allowed_token_ids: Sequence[int],
    mask_id: int,
    num_steps: int = 32,
) -> torch.Tensor:
    """Reveal masked positions with the lowest predictive entropy first."""
    input_ids = batch["input_ids"].clone()
    if not target_positions:
        return input_ids
    target_set = set(int(pos) for pos in target_positions)
    input_ids[0, list(target_set)] = mask_id
    remaining = set(target_set)
    allowed = torch.tensor(list(allowed_token_ids), dtype=torch.long, device=input_ids.device)
    steps = max(1, int(num_steps))

    for step in range(steps):
        if not remaining:
            break
        batch["input_ids"] = input_ids
        batch["time_steps"] = torch.full_like(batch["time_steps"], 1.0 - (step / steps))
        outputs = _forward_model(model, batch)
        logits = outputs["token_logits"][0, list(remaining)]
        restricted = logits.index_select(-1, allowed)
        probs = torch.softmax(restricted, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
        reveal_count = max(1, math.ceil(len(remaining) / (steps - step)))
        ordered = torch.argsort(entropy)[:reveal_count].tolist()
        remaining_list = list(remaining)
        for idx in ordered:
            token_pos = remaining_list[idx]
            token_id = allowed[int(torch.argmax(probs[idx]))].item()
            input_ids[0, token_pos] = token_id
            remaining.remove(token_pos)
    return input_ids


def nussinov_decode(
    seq: str,
    pair_scores: torch.Tensor | Sequence[Sequence[float]],
    min_loop_length: int = 3,
    allow_wobble: bool = True,
    pair_threshold: float = 0.5,
    nussinov_gamma: float = 1.0,
    token_pair_compatibility: torch.Tensor | None = None,
    token_alpha: float = 0.25,
    input_is_logit: bool = False,
    pair_prior: "np.ndarray | None" = None,
    pair_prior_alpha: float = 0.0,
    pruning_mask: "np.ndarray | None" = None,
) -> str:
    """Decode a non-pseudoknotted dot-bracket structure from pair logits/probabilities.

    The DP score is continuous rather than a hard probability cutoff. A small
    top-candidate floor prevents early, poorly calibrated checkpoints from
    always decoding to all-dot structures solely because every probability is
    below the nominal threshold.
    """
    length = len(seq)
    if length == 0:
        return ""
    if isinstance(pair_scores, torch.Tensor):
        raw_scores = pair_scores.detach().float().cpu().numpy()
    else:
        raw_scores = np.asarray(pair_scores, dtype=np.float32)
    raw_scores = raw_scores[:length, :length].astype(np.float32, copy=False)
    if input_is_logit:
        logits = raw_scores
    else:
        probs = np.clip(raw_scores, 1e-6, 1.0 - 1e-6)
        logits = np.log(probs / (1.0 - probs))
    threshold = min(max(float(pair_threshold), 1e-6), 1.0 - 1e-6)
    threshold_logit = math.log(threshold / (1.0 - threshold))
    score_matrix = float(nussinov_gamma) * (logits - threshold_logit)
    if token_pair_compatibility is not None:
        if isinstance(token_pair_compatibility, torch.Tensor):
            prior = token_pair_compatibility.detach().float().cpu().numpy()
        else:
            prior = np.asarray(token_pair_compatibility, dtype=np.float32)
        score_matrix = score_matrix + float(token_alpha) * prior[:length, :length]

    # Apply biological pair-prior (optional inference-time enhancement)
    if pair_prior is not None and pair_prior_alpha > 0:
        if isinstance(pair_prior, torch.Tensor):
            pp = pair_prior.detach().float().cpu().numpy()
        else:
            pp = np.asarray(pair_prior, dtype=np.float32)
        score_matrix = score_matrix + float(pair_prior_alpha) * pp[:length, :length]
    elif pair_prior_alpha > 0 and pair_prior is None:
        # Auto-build pair-prior from sequence.
        from utils.rna_priors import build_pair_prior_matrix
        pp = build_pair_prior_matrix(seq)
        score_matrix = score_matrix + float(pair_prior_alpha) * pp[:length, :length]

    valid_mask = np.zeros((length, length), dtype=bool)
    for i in range(length):
        for j in range(i + max(1, int(min_loop_length)), length):
            if seq[i] != "N" and seq[j] != "N" and not canonical_pair(seq[i], seq[j], allow_wobble):
                continue
            valid_mask[i, j] = True

    if pruning_mask is not None:
        pm = np.asarray(pruning_mask, dtype=bool)
        valid_mask = valid_mask & pm[:length, :length]

    valid_candidates: List[tuple[float, int, int]] = []
    valid_i, valid_j = np.nonzero(valid_mask)
    for i, j in zip(valid_i.tolist(), valid_j.tolist()):
            valid_candidates.append((float(score_matrix[i, j]), i, j))
    valid_candidates.sort(reverse=True)
    topk = max(1, min(len(valid_candidates), length // 2 if length > 1 else 1))
    for rank, (_, i, j) in enumerate(valid_candidates[:topk]):
        if score_matrix[i, j] <= 0:
            score_matrix[i, j] = 0.05 * (topk - rank) / topk

    dp = np.zeros((length, length), dtype=np.float32)
    choice: Dict[Tuple[int, int], tuple] = {}

    for span in range(1, length):
        for i in range(0, length - span):
            j = i + span
            best = float(dp[i + 1, j]) if i + 1 <= j else 0.0
            choice[(i, j)] = ("skip_i",)
            if dp[i, j - 1] > best:
                best = float(dp[i, j - 1])
                choice[(i, j)] = ("skip_j",)
            if valid_mask[i, j]:
                score = float(score_matrix[i, j])
                paired = (float(dp[i + 1, j - 1]) if i + 1 <= j - 1 else 0.0) + score
                if paired > best:
                    best = paired
                    choice[(i, j)] = ("pair", i, j)
            if j - i > 1:
                split_scores = dp[i, i + 1 : j] + dp[i + 2 : j + 1, j]
                if split_scores.size:
                    offset = int(np.argmax(split_scores))
                    split = float(split_scores[offset])
                    if split > best:
                        best = split
                        choice[(i, j)] = ("split", i + 1 + offset)
            dp[i, j] = best

    pairs: List[tuple[int, int]] = []

    def backtrack(i: int, j: int) -> None:
        if i >= j:
            return
        action = choice.get((i, j), ("skip_i",))
        if action[0] == "skip_i":
            backtrack(i + 1, j)
        elif action[0] == "skip_j":
            backtrack(i, j - 1)
        elif action[0] == "pair":
            pairs.append((action[1], action[2]))
            backtrack(i + 1, j - 1)
        else:
            k = action[1]
            backtrack(i, k)
            backtrack(k + 1, j)

    backtrack(0, length - 1)
    return pairs_to_dot_bracket(pairs, length)


def apply_pruning_mask(
    seq: str,
    pair_scores: "torch.Tensor | np.ndarray",
    strategy: str,
    params: dict | None = None,
) -> np.ndarray:
    """Return a boolean (L x L) mask of allowed pairs for a given pruning strategy.

    Strategies
    ----------
    canonical_only
        Only AU, UA, GC, CG, GU, UG pairs (respects allow_wobble in params).
    min_loop_strict
        Ensure |i - j| > min_loop (param: min_loop, default 3).
    topk_per_base
        Each base i keeps the top-k candidates j by score (param: k).
    distance_bucket_top_percentile
        Grouped by |i - j| distance bucket, keep top percentile (param: percentile).
    canonical_plus_topk
        canonical_only AND topk_per_base.
    canonical_plus_distance_prune
        canonical_only AND distance_bucket_top_percentile.
    """
    params = dict(params or {})
    length = len(seq)
    # Convert pair_scores to numpy
    if isinstance(pair_scores, torch.Tensor):
        scores = pair_scores.detach().float().cpu().numpy()
    else:
        scores = np.asarray(pair_scores, dtype=np.float32)
    scores = scores[:length, :length]

    strategy = str(strategy).lower().strip()

    def _canonical_mask() -> np.ndarray:
        allow_wobble = bool(params.get("allow_wobble", True))
        cm = np.zeros((length, length), dtype=bool)
        for i in range(length):
            for j in range(i + 1, length):
                if canonical_pair(seq[i], seq[j], allow_wobble):
                    cm[i, j] = True
        return cm

    def _min_loop_mask() -> np.ndarray:
        ml = int(params.get("min_loop", 3))
        mm = np.zeros((length, length), dtype=bool)
        for i in range(length):
            for j in range(i + ml + 1, length):
                mm[i, j] = True
        return mm

    def _topk_mask() -> np.ndarray:
        k = int(params.get("k", 1))
        tk = np.zeros((length, length), dtype=bool)
        for i in range(length):
            row = scores[i, i + 1:]
            if len(row) == 0:
                continue
            k_actual = min(k, len(row))
            top_indices = np.argpartition(-row, k_actual - 1)[:k_actual]
            for idx in top_indices:
                tk[i, i + 1 + int(idx)] = True
        return tk

    def _distance_bucket_mask() -> np.ndarray:
        percentile = float(params.get("percentile", 10))
        db = np.zeros((length, length), dtype=bool)
        buckets: dict[int, list[tuple[float, int, int]]] = {}
        for i in range(length):
            for j in range(i + 1, length):
                d = j - i
                buckets.setdefault(d, []).append((float(scores[i, j]), i, j))
        for d, candidates in buckets.items():
            if not candidates:
                continue
            vals = [c[0] for c in candidates]
            threshold = float(np.percentile(vals, 100.0 - percentile))
            for score_val, ci, cj in candidates:
                if score_val >= threshold:
                    db[ci, cj] = True
        return db

    if strategy == "canonical_only":
        return _canonical_mask()
    elif strategy == "min_loop_strict":
        return _min_loop_mask()
    elif strategy == "topk_per_base":
        return _topk_mask()
    elif strategy == "distance_bucket_top_percentile":
        return _distance_bucket_mask()
    elif strategy == "canonical_plus_topk":
        return _canonical_mask() & _topk_mask()
    elif strategy == "canonical_plus_distance_prune":
        return _canonical_mask() & _distance_bucket_mask()
    else:
        raise ValueError(
            f"Unknown pruning strategy: {strategy!r}. "
            f"Valid: canonical_only, min_loop_strict, topk_per_base, "
            f"distance_bucket_top_percentile, canonical_plus_topk, canonical_plus_distance_prune"
        )


def prune_and_decode(
    seq: str,
    pair_scores: "torch.Tensor | np.ndarray",
    pruning_strategy: str,
    pruning_params: dict | None = None,
    **nussinov_kwargs,
) -> str:
    """Apply a pruning mask to pair scores, then run Nussinov decoding.

    Parameters
    ----------
    seq : str
        RNA sequence.
    pair_scores : torch.Tensor or np.ndarray
        (L, L) pair logits or probabilities.
    pruning_strategy : str
        One of the supported pruning strategies (see apply_pruning_mask).
    pruning_params : dict or None
        Additional parameters for the pruning strategy.
    **nussinov_kwargs
        Forwarded to nussinov_decode (min_loop_length, allow_wobble, etc.).

    Returns
    -------
    str
        Dot-bracket structure string.
    """
    pruning_mask = apply_pruning_mask(seq, pair_scores, pruning_strategy, pruning_params)
    return nussinov_decode(seq, pair_scores, pruning_mask=pruning_mask, **nussinov_kwargs)


def _canonical_mask_for_batch(
    seqs: Sequence[str],
    max_len: int,
    device: torch.device,
    allow_wobble: bool,
) -> torch.Tensor:
    code = {"A": 0, "U": 1, "G": 2, "C": 3, "N": 4}
    encoded = torch.full((len(seqs), max_len), 4, dtype=torch.long, device=device)
    for batch_idx, seq in enumerate(seqs):
        values = [code.get(base, 4) for base in seq.upper().replace("T", "U")]
        if values:
            encoded[batch_idx, : len(values)] = torch.tensor(values, dtype=torch.long, device=device)
    allowed = torch.zeros((5, 5), dtype=torch.bool, device=device)
    for left, right in [("A", "U"), ("U", "A"), ("G", "C"), ("C", "G")]:
        allowed[code[left], code[right]] = True
    if allow_wobble:
        allowed[code["G"], code["U"]] = True
        allowed[code["U"], code["G"]] = True
    left = encoded.unsqueeze(2).expand(-1, -1, max_len)
    right = encoded.unsqueeze(1).expand(-1, max_len, -1)
    return allowed[left, right]


@torch.no_grad()
def batched_greedy_decode_gpu(
    pair_logits: torch.Tensor,
    seqs: list[str] | None = None,
    min_loop_length: int = 3,
    pair_threshold: float = 0.25,
    allow_wobble: bool = True,
    canonical_only: bool = True,
    max_pairs: int | None = None,
    prevent_crossing: bool = False,
) -> torch.Tensor:
    """Fast greedy base-pair decoding from batched pair logits.

    This is an approximate benchmark path. It avoids CPU Nussinov DP and keeps
    the score filtering on tensors, then runs a small greedy selection loop over
    top candidates per sample.
    """
    if pair_logits.ndim != 3:
        raise ValueError(f"pair_logits must have shape (B, L, L), got {tuple(pair_logits.shape)}")
    device = pair_logits.device
    batch_size, max_len, _ = pair_logits.shape
    lengths = [max_len] * batch_size if seqs is None else [len(seq) for seq in seqs]
    idx = torch.arange(max_len, device=device)
    length_tensor = torch.tensor(lengths, dtype=torch.long, device=device)
    valid_len = idx.unsqueeze(0) < length_tensor.unsqueeze(1)
    valid = valid_len.unsqueeze(1) & valid_len.unsqueeze(2)
    valid = valid & (idx.view(1, 1, max_len) > idx.view(1, max_len, 1))
    if min_loop_length > 0:
        valid = valid & ((idx.view(1, 1, max_len) - idx.view(1, max_len, 1)) >= int(min_loop_length))
    if canonical_only and seqs is not None:
        valid = valid & _canonical_mask_for_batch(seqs, max_len, device, allow_wobble)

    probs = pair_logits.float().sigmoid().masked_fill(~valid, -1.0)
    threshold = float(pair_threshold)
    flat = probs.flatten(1)
    default_max_pairs = max(1, max_len // 2)
    pair_limit = int(max_pairs or default_max_pairs)
    candidate_count = min(flat.size(1), max(pair_limit * 20, pair_limit))
    values, indices = torch.topk(flat, k=candidate_count, dim=1)
    pred = torch.zeros_like(probs, dtype=torch.bool)

    for batch_idx in range(batch_size):
        used: set[int] = set()
        selected: list[tuple[int, int]] = []
        limit = int(max_pairs or max(1, lengths[batch_idx] // 2))
        for value, flat_idx in zip(values[batch_idx].tolist(), indices[batch_idx].tolist()):
            if value < threshold or len(selected) >= limit:
                break
            i = flat_idx // max_len
            j = flat_idx % max_len
            if i in used or j in used:
                continue
            if prevent_crossing and any(i < a < j < b or a < i < b < j for a, b in selected):
                continue
            selected.append((i, j))
            used.add(i)
            used.add(j)
            pred[batch_idx, i, j] = True
    return pred


def greedy_pairs_to_dotbracket(pairs: Sequence[tuple[int, int]], length: int) -> tuple[str, int]:
    kept: list[tuple[int, int]] = []
    skipped = 0
    used: set[int] = set()
    for raw_i, raw_j in sorted((min(i, j), max(i, j)) for i, j in pairs):
        if raw_i in used or raw_j in used:
            skipped += 1
            continue
        if any(raw_i < a < raw_j < b or a < raw_i < b < raw_j for a, b in kept):
            skipped += 1
            continue
        kept.append((raw_i, raw_j))
        used.add(raw_i)
        used.add(raw_j)
    return pairs_to_dot_bracket(kept, length), skipped


def pairs_matrix_to_dotbracket_batch(pair_matrix: torch.Tensor, lengths: list[int]) -> list[str]:
    structs, _ = pairs_matrix_to_dotbracket_batch_with_stats(pair_matrix, lengths)
    return structs


def pairs_matrix_to_dotbracket_batch_with_stats(
    pair_matrix: torch.Tensor,
    lengths: list[int],
) -> tuple[list[str], list[int]]:
    matrix = pair_matrix.detach().bool().cpu()
    structs: list[str] = []
    skipped: list[int] = []
    for batch_idx, length in enumerate(lengths):
        pairs = [
            (int(i), int(j))
            for i, j in torch.nonzero(matrix[batch_idx, :length, :length], as_tuple=False).tolist()
            if i < j
        ]
        struct, count = greedy_pairs_to_dotbracket(pairs, length)
        structs.append(struct)
        skipped.append(count)
    return structs, skipped


@torch.no_grad()
def generate_structure_seq2struct(
    model: torch.nn.Module,
    tokenizer: RNAOmniTokenizer,
    seq: str,
    decoding_config: dict,
    device: torch.device | str = "cpu",
    pair_prior: "np.ndarray | None" = None,
    pair_prior_alpha: float = 0.0,
) -> str:
    seq = seq.upper().replace("T", "U")
    struct = "." * len(seq)
    batch, _, struct_positions = _build_inference_batch(tokenizer, "seq2struct", seq, struct, device=device)
    allowed = [tokenizer.token_id(token) for token in tokenizer.structure_tokens]
    decode_source = str(decoding_config.get("decode_source", "pair")).lower()
    if decode_source == "pair" and decoding_config.get("use_nussinov", True):
        batch["input_ids"][:, struct_positions] = tokenizer.mask_id
        outputs = _forward_model(model, batch)
        if outputs["pair_logits"] is not None:
            pair_logits = outputs["pair_logits"][0, : len(seq), : len(seq)]
            return nussinov_decode(
                seq,
                pair_logits,
                min_loop_length=int(decoding_config.get("min_loop_length", 3)),
                allow_wobble=bool(decoding_config.get("allow_wobble", True)),
                pair_threshold=float(decoding_config.get("pair_threshold", 0.5)),
                nussinov_gamma=float(decoding_config.get("nussinov_gamma", 1.0)),
                input_is_logit=True,
                pair_prior=pair_prior,
                pair_prior_alpha=pair_prior_alpha,
            )
    input_ids = entropy_iterative_unmask(
        model,
        batch,
        struct_positions,
        allowed,
        tokenizer.mask_id,
        num_steps=int(decoding_config.get("num_steps", 32)),
    )
    batch["input_ids"] = input_ids
    outputs = _forward_model(model, batch)
    ids = input_ids[0, struct_positions].tolist()
    token_struct = "".join(tokenizer.decode(ids))
    if decode_source == "token" or not decoding_config.get("use_nussinov", True):
        return token_struct
    if outputs["pair_logits"] is not None:
        pair_logits = outputs["pair_logits"][0, : len(seq), : len(seq)]
        compatibility = None
        if decode_source == "hybrid":
            compatibility = token_pair_compatibility(token_struct)
        return nussinov_decode(
            seq,
            pair_logits,
            min_loop_length=int(decoding_config.get("min_loop_length", 3)),
            allow_wobble=bool(decoding_config.get("allow_wobble", True)),
            pair_threshold=float(decoding_config.get("pair_threshold", 0.5)),
            nussinov_gamma=float(decoding_config.get("nussinov_gamma", 1.0)),
            token_pair_compatibility=compatibility,
            input_is_logit=True,
        )
    return token_struct


def token_pair_compatibility(struct: str) -> torch.Tensor:
    matrix = torch.zeros((len(struct), len(struct)), dtype=torch.float32)
    for i, left in enumerate(struct):
        if left not in "([{":
            continue
        for j in range(i + 1, len(struct)):
            if struct[j] in ")]}":
                matrix[i, j] = 1.0
    return matrix


@torch.no_grad()
def generate_sequence_invfold(
    model: torch.nn.Module,
    tokenizer: RNAOmniTokenizer,
    struct: str,
    decoding_config: dict,
    device: torch.device | str = "cpu",
) -> str:
    seq = "N" * len(struct)
    batch, seq_positions, _ = _build_inference_batch(tokenizer, "invfold", seq, struct, device=device)
    allowed = [tokenizer.token_id(token) for token in tokenizer.sequence_tokens]
    input_ids = entropy_iterative_unmask(
        model,
        batch,
        seq_positions,
        allowed,
        tokenizer.mask_id,
        num_steps=int(decoding_config.get("num_steps", 32)),
    )
    seq_tokens = tokenizer.decode(input_ids[0, seq_positions].tolist())
    seq_list = list(seq_tokens)
    try:
        for i, j in parse_dot_bracket(struct):
            if not canonical_pair(seq_list[i], seq_list[j], bool(decoding_config.get("allow_wobble", True))):
                seq_list[i], seq_list[j] = "G", "C"
    except ValueError:
        pass
    return "".join(seq_list)

