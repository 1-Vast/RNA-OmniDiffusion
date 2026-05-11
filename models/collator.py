from __future__ import annotations

import random
from typing import Dict, List, Sequence

import torch

from models.token import RNAOmniTokenizer
from utils.structtags import categorical_to_ids
from models.mask import (
    motif_span_mask_positions,
    pair_aware_mask_positions,
    random_span_positions,
    random_token_mask,
)


class RNAOmniCollator:
    """Build task-conditioned masked discrete diffusion batches."""

    task_names = ["seq2struct", "invfold", "inpaint", "motif_control", "seq_denoise"]

    def __init__(
        self,
        tokenizer: RNAOmniTokenizer,
        task_ratios: Dict[str, float],
        pair_negative_ratio: int = 3,
        seed: int | None = None,
        ablation: Dict[str, object] | None = None,
        typed_spec=None,
        relation_mask: Dict[str, Any] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.task_ratios = task_ratios
        self.pair_negative_ratio = pair_negative_ratio
        self.rng = random.Random(seed)
        self.ablation = ablation or {}
        self.typed_spec = typed_spec
        self.relmask = relation_mask or {}
        self.use_pair_aware_masking = bool(self.ablation.get("use_pair_aware_masking", True))
        self.use_motif_span_masking = bool(self.ablation.get("use_motif_span_masking", True))
        self.use_motif_condition = bool(self.ablation.get("use_motif_condition", True))
        self.use_family_condition = bool(self.ablation.get("use_family_condition", True))
        ratios = dict(task_ratios)
        if "seq_denoise" in ratios and "denoise" not in ratios:
            ratios["denoise"] = ratios["seq_denoise"]
        self.task_ratios = ratios
        weights = [float(ratios.get(task, 0.0)) for task in self.task_names]
        if sum(weights) <= 0:
            raise ValueError("At least one task sampling ratio must be positive.")
        self.task_weights = weights

    def __call__(self, samples: Sequence[dict]) -> dict:
        examples = []
        for sample in samples:
            task_name = self.rng.choices(self.task_names, weights=self.task_weights, k=1)[0]
            time_step = self.rng.random()
            mask_ratio = max(0.15, min(0.95, time_step))
            spec = None
            if self.typed_spec is not None:
                spec = self.typed_spec.get(sample.get("id", ""))
            examples.append(self._build_example(sample, task_name, time_step, mask_ratio, spec))

        max_tokens = max(len(example["input_ids"]) for example in examples)
        max_len = max(example["length"] for example in examples)

        input_ids = torch.full((len(examples), max_tokens), self.tokenizer.pad_id, dtype=torch.long)
        labels = torch.full((len(examples), max_tokens), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(examples), max_tokens), dtype=torch.long)
        segment_ids = torch.zeros((len(examples), max_tokens), dtype=torch.long)
        seq_positions = torch.full((len(examples), max_len), -1, dtype=torch.long)
        struct_positions = torch.full((len(examples), max_len), -1, dtype=torch.long)
        pair_labels = torch.zeros((len(examples), max_len, max_len), dtype=torch.float32)
        pair_mask = torch.zeros((len(examples), max_len, max_len), dtype=torch.bool)
        pair_positive_counts = torch.zeros(len(examples), dtype=torch.long)
        pair_negative_counts = torch.zeros(len(examples), dtype=torch.long)
        is_labeled = torch.tensor([bool(example.get("is_labeled", True)) for example in examples], dtype=torch.bool)

        for batch_idx, example in enumerate(examples):
            token_count = len(example["input_ids"])
            length = example["length"]
            input_ids[batch_idx, :token_count] = torch.tensor(example["input_ids"], dtype=torch.long)
            labels[batch_idx, :token_count] = torch.tensor(example["labels"], dtype=torch.long)
            attention_mask[batch_idx, :token_count] = 1
            segment_ids[batch_idx, :token_count] = torch.tensor(example["segment_ids"], dtype=torch.long)
            seq_positions[batch_idx, :length] = torch.tensor(example["seq_positions"], dtype=torch.long)
            if example["struct_positions"]:
                struct_positions[batch_idx, :length] = torch.tensor(example["struct_positions"], dtype=torch.long)
            positive_count, negative_count = self._fill_pair_tensors(
                pair_labels[batch_idx],
                pair_mask[batch_idx],
                example["pairs"],
                length,
            )
            pair_positive_counts[batch_idx] = positive_count
            pair_negative_counts[batch_idx] = negative_count

        task_names = [example["task_name"] for example in examples]
        task_ids = torch.tensor([self.tokenizer.task_to_id["denoise" if name == "seq_denoise" else name] for name in task_names], dtype=torch.long)

        # ---- structural importance: per-position _imp -> pair-level [B, L, L] ----
        _pair_imp = torch.zeros((len(examples), max_len, max_len), dtype=torch.float32)
        _has_imp = False
        for batch_idx, example in enumerate(examples):
            imp_arr = example.get("_imp")
            if imp_arr is not None and len(imp_arr) > 0:
                _has_imp = True
                imp_t = torch.tensor(imp_arr, dtype=torch.float32)
                L = min(len(imp_t), max_len)
                _pair_imp[batch_idx, :L, :L] = torch.outer(imp_t[:L], imp_t[:L])

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "segment_ids": segment_ids,
            "task_ids": task_ids,
            "task_names": task_names,
            "time_steps": torch.tensor([example["time_step"] for example in examples], dtype=torch.float32),
            "pair_labels": pair_labels,
            "pair_mask": pair_mask,
            "pair_positive_counts": pair_positive_counts,
            "pair_negative_counts": pair_negative_counts,
            "seq_positions": seq_positions,
            "struct_positions": struct_positions,
            "lengths": torch.tensor([example["length"] for example in examples], dtype=torch.long),
            "raw_seq": [example["raw_seq"] for example in examples],
            "raw_struct": [example["raw_struct"] for example in examples],
            "raw_pairs": [example["pairs"] for example in examples],
            "sample_ids": [example["sample_id"] for example in examples],
            "is_labeled": is_labeled,
            "_weight": torch.tensor([example.get("_weight", 1.0) for example in examples], dtype=torch.float32),
            "mask_stats": _accumulate_mask_stats(examples),
        }
        if _has_imp:
            batch["pair_importance"] = _pair_imp

        struct_tags_list = [example.get("_struct_tags") for example in examples]
        if all(tags is not None and isinstance(tags, dict) for tags in struct_tags_list):
            B = len(examples)
            struct_numeric = torch.zeros(B, 11, dtype=torch.float32)
            struct_categorical = torch.zeros(B, 4, dtype=torch.long)
            for i, tags in enumerate(struct_tags_list):
                if tags.get("numeric"):
                    struct_numeric[i] = torch.tensor([tags["numeric"][k] for k in sorted(tags["numeric"].keys())], dtype=torch.float32)
                if tags.get("categorical"):
                    struct_categorical[i] = torch.tensor(categorical_to_ids(tags["categorical"]), dtype=torch.long)
            batch["struct_numeric"] = struct_numeric
            batch["struct_categorical"] = struct_categorical

        return batch

    def _build_example(self, sample: dict, task_name: str, time_step: float, mask_ratio: float, spec: dict | None = None) -> dict:
        tokens: List[str] = []
        segment_ids: List[int] = []
        seq_positions: List[int] = []
        struct_positions: List[int] = []

        # Apply typed spec overrides if present
        if spec:
            sm = spec.get("mask", {})
            mask_ratio = float(sm.get("ratio", mask_ratio))
            mask_ratio = max(0.05, min(0.95, mask_ratio))
            mask_mode = sm.get("mode", "random")
            mask_span = int(sm.get("span", 3))
        else:
            mask_mode = "random"
            mask_span = 3

        def add(token: str, segment_id: int) -> int:
            tokens.append(token)
            segment_ids.append(segment_id)
            return len(tokens) - 1

        token_task_name = "denoise" if task_name == "seq_denoise" else task_name
        add(self.tokenizer.task_token(token_task_name), 0)

        if task_name in {"seq2struct", "invfold", "motif_control"} and not sample.get("is_labeled", True):
            task_name = "seq_denoise"

        if task_name == "motif_control" and (self.use_family_condition or self.use_motif_condition):
            family = sample.get("family") or ""
            if self.use_family_condition:
                add("<FAMILY>", 3)
                add(self.tokenizer.family_token(family), 3)
                add("</FAMILY>", 3)
            if self.use_motif_condition:
                add("<MOTIF>", 3)
                for motif in sample.get("motifs", []):
                    add(self.tokenizer.motif_token(motif.get("type")), 3)
                add("</MOTIF>", 3)

        add("<SEQ>", 1)
        for base in sample["seq"]:
            seq_positions.append(add(base, 1))
        add("</SEQ>", 1)

        if sample.get("is_labeled", True):
            add("<STRUCT>", 2)
            for char in sample["struct"]:
                struct_positions.append(add(char, 2))
            add("</STRUCT>", 2)

        clean_ids = self.tokenizer.encode(tokens)
        input_ids = list(clean_ids)
        labels = [-100] * len(clean_ids)
        masked_token_positions = self._select_masked_token_positions(
            task_name,
            seq_positions,
            struct_positions,
            sample,
            mask_ratio,
            mask_mode=mask_mode,
            mask_span=mask_span,
        )
        for token_pos in masked_token_positions:
            input_ids[token_pos] = self.tokenizer.mask_id
            labels[token_pos] = clean_ids[token_pos]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "segment_ids": segment_ids,
            "seq_positions": seq_positions,
            "struct_positions": struct_positions,
            "task_name": task_name,
            "time_step": time_step,
            "pairs": sample.get("pairs", []),
            "length": sample["length"],
            "raw_seq": sample["seq"],
            "raw_struct": sample["struct"] if sample.get("is_labeled", True) else "",
            "sample_id": sample.get("id", ""),
            "is_labeled": sample.get("is_labeled", True),
            "_weight": sample.get("_weight", 1.0),
            "_struct_tags": sample.get("_struct_tags"),
            "_imp": sample.get("_imp"),
        }
    def _select_masked_token_positions(
        self,
        task_name: str,
        seq_positions: Sequence[int],
        struct_positions: Sequence[int],
        sample: dict,
        mask_ratio: float,
        mask_mode: str = "random",
        mask_span: int = 3,
    ) -> List[int]:
        if task_name == "seq2struct":
            return random_token_mask(struct_positions, mask_ratio, self.rng)
        if task_name == "invfold":
            return random_token_mask(seq_positions, mask_ratio, self.rng)
        if task_name == "motif_control":
            return list(seq_positions) + list(struct_positions)
        if task_name == "seq_denoise":
            if mask_mode == "span":
                return list(random_span_positions(len(sample["seq"]), mask_ratio, self.rng))
            return random_token_mask(seq_positions, mask_ratio, self.rng)

        length = sample["length"]
        if not self.use_pair_aware_masking and not self.use_motif_span_masking:
            nucleotide_positions = set(random_token_mask(list(range(length)), mask_ratio, self.rng))
        elif self.use_motif_span_masking and sample.get("motifs") and self.rng.random() < 0.6:
            nucleotide_positions = motif_span_mask_positions(sample["motifs"], length, self.rng)
        else:
            nucleotide_positions = random_span_positions(length, mask_ratio, self.rng)
        if self.use_pair_aware_masking and self.relmask.get("enabled", False):
            nucleotide_positions = _relation_mask_sample(
                nucleotide_positions, sample, self.relmask, self.rng
            )
            self._last_mask_stats = getattr(_relation_mask_sample, '_last_stats', None)
        elif self.use_pair_aware_masking:
            nucleotide_positions = pair_aware_mask_positions(nucleotide_positions, sample.get("pairs", []))
        token_positions = []
        for nuc_idx in sorted(pos for pos in nucleotide_positions if 0 <= pos < length):
            token_positions.append(seq_positions[nuc_idx])
            token_positions.append(struct_positions[nuc_idx])
        return token_positions or [seq_positions[0], struct_positions[0]]

    def _fill_pair_tensors(
        self,
        labels: torch.Tensor,
        mask: torch.Tensor,
        pairs: Sequence[Sequence[int]],
        length: int,
    ) -> tuple[int, int]:
        positive = set()
        for raw_i, raw_j in pairs:
            i, j = int(raw_i), int(raw_j)
            if i > j:
                i, j = j, i
            if 0 <= i < j < length:
                labels[i, j] = 1.0
                positive.add((i, j))
        for i, j in positive:
            mask[i, j] = True

        # Negative pairs are sampled in compute_omni_loss on the target device.
        # Keeping this collator positive-only avoids Python O(L^2) candidate
        # enumeration, which otherwise starves the GPU on real ArchiveII batches.
        return len(positive), 0


# ---- Relation-mask fine-grained sampler ----

CANONICAL = {("A","U"),("U","A"),("G","C"),("C","G"),("G","U"),("U","G")}

def _relation_mask_sample(
    positions: set,
    sample: dict,
    relmask: dict,
    rng: random.Random,
) -> set:
    """Fine-grained relation-mask sampling with ratio control.

    Creates candidate pools from hard_negative, stem_span, global, and random
    sources, then samples proportionally and tracks mask distribution stats.
    """
    seq = sample.get("seq", "")
    pairs = [(int(p[0]), int(p[1])) for p in sample.get("pairs", [])]
    L = len(seq)
    total_ratio = float(relmask.get("total_ratio", 0.25))
    hard_r = float(relmask.get("hard_negative_ratio", 0.25))
    stem_r = float(relmask.get("stem_span_ratio", 0.35))
    global_r = float(relmask.get("global_ratio", 0.40))
    stem_len = int(relmask.get("stem_span_len", 4))
    long_thresh = int(relmask.get("long_range_threshold", 64))

    # Normalize ratios
    total = max(0.001, hard_r + stem_r + global_r)
    hard_r /= total
    stem_r /= total
    global_r /= total

    target = max(1, int(round(total_ratio * L)))

    # ---- Build candidate pools ----
    pair_set = {tuple(sorted(p)) for p in pairs}

    # hard_negative: canonical pairs NOT in true pairs
    hard_pool = set()
    for i in range(L):
        for d in range(4, min(L - i, 20)):  # limited search for speed
            j = i + d
            if (i, j) in pair_set:
                continue
            if (seq[i], seq[j]) in CANONICAL:
                hard_pool.add(i)
                hard_pool.add(j)

    # stem_span: positions near true pair stems
    stem_pool = set()
    for i, j in pairs:
        for offset in range(-stem_len, stem_len + 1):
            if 0 <= i + offset < L:
                stem_pool.add(i + offset)
            if 0 <= j + offset < L:
                stem_pool.add(j + offset)

    # global: long-range pair endpoints
    global_pool = set()
    for i, j in pairs:
        if j - i >= long_thresh:
            global_pool.add(i)
            global_pool.add(j)

    # Sample from pools
    def sample_from(pool, n):
        available = list(pool - positions)  # don't duplicate base positions
        if not available:
            return set()
        return set(rng.sample(available, min(n, len(available))))

    n_hard = int(round(target * hard_r))
    n_stem = int(round(target * stem_r))
    n_global = int(round(target * global_r))

    hard_selected = sample_from(hard_pool, n_hard)
    stem_selected = sample_from(stem_pool, n_stem)
    global_selected = sample_from(global_pool, n_global)

    result = positions.copy()
    result |= hard_selected | stem_selected | global_selected

    # Store stats for this sample
    _relation_mask_sample._last_stats = {
        "hard_count": len(hard_selected),
        "stem_count": len(stem_selected),
        "global_count": len(global_selected),
        "rand_count": max(0, target - len(hard_selected) - len(stem_selected) - len(global_selected)),
        "total_count": len(result) - len(positions),
        "total_ratio": len(result) / max(1, L),
        "candidate_hard": len(hard_pool),
        "candidate_stem": len(stem_pool),
        "candidate_global": len(global_pool),
    }
    return result


def _accumulate_mask_stats(examples: list[dict]) -> dict:
    """Accumulate relation-mask stats across a batch from _last_stats per sample."""
    # Use the last _relation_mask_sample._last_stats as aggregate
    last = getattr(_relation_mask_sample, '_last_stats', None)
    if last is None:
        return {"relmask_active": False, "sample_count": len(examples)}
    return {"relmask_active": True, "sample_count": len(examples), **last}

