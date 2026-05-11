# utils/slices.py — Error taxonomy slicers for RNA secondary structure.
from __future__ import annotations

COMPLEMENT = {"A": "U", "U": "AG", "G": "CU", "C": "G"}

def _pair_distances(pairs: list[tuple]) -> list[int]:
    return [abs(int(j) - int(i)) for i, j in pairs if int(j) > int(i)]

def _is_canonical(seq: str, pairs: list[tuple]) -> list[bool]:
    result = []
    for i, j in pairs:
        ci, cj = int(i), int(j)
        if ci >= len(seq) or cj >= len(seq):
            result.append(False)
        else:
            result.append(seq[ci] + seq[cj] in {"AU", "UA", "GC", "CG", "GU", "UG"})
    return result

def short_range_pairs(pairs: list[tuple]) -> bool:
    dists = _pair_distances(pairs)
    return len(dists) > 0 and all(d < 16 for d in dists)

def medium_range_pairs(pairs: list[tuple]) -> bool:
    dists = _pair_distances(pairs)
    return len(dists) > 0 and all(16 <= d < 48 for d in dists)

def long_range_pairs(pairs: list[tuple]) -> bool:
    dists = _pair_distances(pairs)
    return len(dists) > 0 and any(d >= 48 for d in dists)

def all_canonical(seq: str, pairs: list[tuple]) -> bool:
    if not pairs: return False
    return all(_is_canonical(seq, pairs))

def high_gc(seq: str) -> bool:
    gc = sum(1 for c in seq if c in "GC")
    return gc / max(1, len(seq)) >= 0.55

def low_gc(seq: str) -> bool:
    gc = sum(1 for c in seq if c in "GC")
    return gc / max(1, len(seq)) <= 0.40

def sparse_pairs(sample: dict) -> bool:
    return len(sample.get("pairs", [])) < sample["length"] * 0.15

def dense_pairs(sample: dict) -> bool:
    return len(sample.get("pairs", [])) >= sample["length"] * 0.30

def single_stem(sample: dict) -> bool:
    struct = sample.get("struct", "")
    stems = 0
    in_stem = False
    for c in struct:
        if c in "(<[{}>)>]":
            if not in_stem:
                stems += 1
                in_stem = True
        else:
            in_stem = False
    return stems <= 2

def multi_stem(sample: dict) -> bool:
    struct = sample.get("struct", "")
    stems = 0
    in_stem = False
    for c in struct:
        if c in "(<[{}>)>]":
            if not in_stem:
                stems += 1
                in_stem = True
        else:
            in_stem = False
    return stems >= 4

def under_pairing(sample: dict, pred_struct: str) -> bool:
    pred_count = sum(1 for c in pred_struct if c in "(){}<>[]")
    true_count = sum(1 for c in sample.get("struct", "") if c in "(){}<>[]")
    return pred_count < true_count * 0.5 if true_count > 0 else False

def over_pairing(sample: dict, pred_struct: str) -> bool:
    pred_count = sum(1 for c in pred_struct if c in "(){}<>[]")
    true_count = sum(1 for c in sample.get("struct", "") if c in "(){}<>[]")
    return pred_count > true_count * 1.5 and true_count > 0

def fragmented_stem(sample: dict) -> bool:
    struct = sample.get("struct", "")
    stems = []
    current = 0
    for c in struct:
        if c in "(){}<>[]":
            current += 1
        else:
            if current > 0:
                stems.append(current)
                current = 0
    if current > 0:
        stems.append(current)
    n_stems = len(stems)
    return n_stems > 0 and sum(stems) / n_stems <= 3.0

def boundary_shift(sample: dict, pred_struct: str) -> bool:
    from utils.struct import parse_dot_bracket
    try:
        true_pairs = set(tuple(sorted(p)) for p in sample.get("pairs", []))
        pred_pairs = set(tuple(sorted(p)) for p in parse_dot_bracket(pred_struct))
    except ValueError:
        return False
    shift_count = 0
    for pi, pj in pred_pairs:
        for ti, tj in true_pairs:
            if abs(pi - ti) == 1 and abs(pj - tj) == 1 and (pi, pj) != (ti, tj):
                shift_count += 1
                break
    return shift_count >= max(1, len(true_pairs) * 0.2)

SLICE_REGISTRY = {
    "short_range_pairs": ("Pred sample with only short-range true pairs", short_range_pairs),
    "medium_range_pairs": ("Pred sample with only medium-range true pairs", medium_range_pairs),
    "long_range_pairs": ("Pred sample with any long-range true pairs", long_range_pairs),
    "canonical_pairs": ("Pred sample where all true pairs are canonical", all_canonical),
    "high_gc_sequences": ("Sequences with GC >= 55%", lambda s: high_gc(s["seq"])),
    "low_gc_sequences": ("Sequences with GC <= 40%", lambda s: low_gc(s["seq"])),
    "sparse_pair_sequences": ("Few true pairs (< 0.15 * L)", sparse_pairs),
    "dense_pair_sequences": ("Many true pairs (>= 0.30 * L)", dense_pairs),
    "single_stem_sequences": ("<= 2 stems", single_stem),
    "multi_stem_sequences": (">= 4 stems", multi_stem),
    "under_pairing_cases": ("Predicted pairs < 50% of true pairs", under_pairing),
    "over_pairing_cases": ("Predicted pairs > 150% of true pairs", over_pairing),
    "fragmented_stem_cases": ("Average stem length <= 3", fragmented_stem),
    "boundary_shift_cases": (">= 20% pair boundaries shifted by 1", boundary_shift),
}

def dominant_error(metrics: dict) -> str:
    """Identify dominant error type from per-sample metrics."""
    f1 = metrics.get("pair_f1", 0)
    precision = metrics.get("pair_precision", 0)
    recall = metrics.get("pair_recall", 0)
    if recall < precision * 0.6:
        return "under_pairing"
    if precision < recall * 0.6:
        return "over_pairing"
    if f1 < 0.2:
        return "low_accuracy"
    return "balanced"
