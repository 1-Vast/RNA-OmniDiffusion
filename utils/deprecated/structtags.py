"""Structural tag extraction from dot-bracket structures.

Extracts numeric and categorical structural features from RNA secondary
structure annotation. No LLM dependency. Used as No-LLM baseline for
structural semantic auxiliary supervision.
"""

from typing import Dict, Any

def extract_numeric_tags(seq, struct, pairs):
    """Extract continuous numeric structural features from dot-bracket."""
    L = max(1, len(seq))
    pc = len(pairs)
    ps = set(pairs)
    # stem count: consecutive pairs count as 1 stem
    stems = 0
    visited = set()
    for i, j in pairs:
        if (i, j) in visited: continue
        stems += 1
        ci, cj = i, j
        while (ci, cj) in ps:
            visited.add((ci, cj))
            ci += 1; cj -= 1
    # stem lengths
    stem_lens = []
    visited.clear()
    for i, j in pairs:
        if (i, j) in visited: continue
        slen = 0
        ci, cj = i, j
        while (ci, cj) in ps:
            visited.add((ci, cj))
            slen += 1
            ci += 1; cj -= 1
        stem_lens.append(slen)
    max_sl = max(stem_lens) if stem_lens else 0
    mean_sl = sum(stem_lens)/len(stem_lens) if stem_lens else 0
    isolated = sum(1 for i, j in pairs if (i+1,j-1) not in ps and (i-1,j+1) not in ps)
    long_range = sum(1 for i, j in pairs if j - i >= 64)
    canon = 0; gc = 0; au = 0; gu = 0
    if seq:
        for i, j in pairs:
            a, b = seq[i], seq[j]
            if (a,b) in {("A","U"),("U","A"),("G","C"),("C","G"),("G","U"),("U","G")}: canon += 1
            if (a,b) in {("G","C"),("C","G")}: gc += 1
            elif (a,b) in {("A","U"),("U","A")}: au += 1
            elif (a,b) in {("G","U"),("U","G")}: gu += 1
    return {
        "pair_density": pc / L,
        "pair_count_norm": pc / max(1, L),
        "stem_count_norm": stems / max(1, L * 0.25),
        "max_stem_len_norm": max_sl / max(1, L * 0.5),
        "mean_stem_len_norm": mean_sl / max(1, L * 0.5),
        "isolated_pair_ratio": isolated / max(1, pc),
        "long_range_ratio": long_range / max(1, pc),
        "canonical_ratio": canon / max(1, pc),
        "gc_pair_ratio": gc / max(1, pc),
        "au_pair_ratio": au / max(1, pc),
        "gu_pair_ratio": gu / max(1, pc),
    }


def extract_categorical_tags(pairs, seq, struct):
    """Extract discrete categorical structural features."""
    pc = len(pairs)
    ps = set(pairs)
    L = max(1, len(seq))
    density = pc / L

    # structure_class
    if pc == 0:
        sclass = "sparse"
    elif len(ps) <= 5:
        sclass = "single_stem_loop"
    elif sum(1 for i,j in pairs if j-i >= 64) >= 3:
        sclass = "long_range_rich"
    else:
        sclass = "multi_stem"

    # pairing_density
    if density < 0.2: pd = "low"
    elif density > 0.5: pd = "high"
    else: pd = "medium"

    # stem_complexity
    stems = 0
    visited = set()
    for i, j in pairs:
        if (i, j) in visited: continue
        stems += 1
        ci, cj = i, j
        while (ci, cj) in ps:
            visited.add((ci, cj))
            ci += 1; cj -= 1
    if stems == 0: sc = "none"
    elif stems <= 2: sc = "low"
    elif stems <= 4: sc = "medium"
    else: sc = "high"

    # isolated_pair_level
    iso = sum(1 for i, j in pairs if (i+1,j-1) not in ps and (i-1,j+1) not in ps)
    iso_r = iso / max(1, pc)
    if iso_r < 0.1: ip = "low"
    elif iso_r < 0.3: ip = "medium"
    else: ip = "high"

    return {
        "structure_class": sclass,
        "pairing_density": pd,
        "stem_complexity": sc,
        "isolated_pair_level": ip,
    }


CATEGORICAL_ENUMS = {
    "structure_class": ["sparse", "single_stem_loop", "multi_stem", "long_range_rich", "unknown"],
    "pairing_density": ["low", "medium", "high"],
    "stem_complexity": ["none", "low", "medium", "high"],
    "isolated_pair_level": ["low", "medium", "high"],
}

NUMERIC_DIM = 11  # number of numeric features above
CATEGORICAL_DIMS = {k: len(v) for k, v in CATEGORICAL_ENUMS.items()}


def extract_tags(seq, struct, pairs):
    return {
        "numeric": extract_numeric_tags(seq, struct, pairs),
        "categorical": extract_categorical_tags(pairs, seq, struct),
    }


def categorical_to_ids(cat):
    """Convert categorical dict to list of class indices."""
    ids = []
    for key, enum in CATEGORICAL_ENUMS.items():
        val = cat.get(key, "unknown")
        if val not in enum: val = "unknown"
        ids.append(enum.index(val))
    return ids
