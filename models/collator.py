from __future__ import annotations

import random
from typing import Dict, List, Sequence

import torch

from models.token import RNAOmniTokenizer
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
    ) -> None:
        self.tokenizer = tokenizer
        self.task_ratios = task_ratios
        self.pair_negative_ratio = pair_negative_ratio
        self.rng = random.Random(seed)
        self.ablation = ablation or {}
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
            examples.append(self._build_example(sample, task_name, time_step, mask_ratio))

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
        }
        return batch

    def _build_example(self, sample: dict, task_name: str, time_step: float, mask_ratio: float) -> dict:
        tokens: List[str] = []
        segment_ids: List[int] = []
        seq_positions: List[int] = []
        struct_positions: List[int] = []

        def add(token: str, segment_id: int) -> int:
            tokens.append(token)
            segment_ids.append(segment_id)
            return len(tokens) - 1

        if task_name in {"seq2struct", "invfold", "motif_control"} and not sample.get("is_labeled", True):
            task_name = "seq_denoise"

        token_task_name = "denoise" if task_name == "seq_denoise" else task_name
        add(self.tokenizer.task_token(token_task_name), 0)

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
        }

    def _select_masked_token_positions(
        self,
        task_name: str,
        seq_positions: Sequence[int],
        struct_positions: Sequence[int],
        sample: dict,
        mask_ratio: float,
    ) -> List[int]:
        if task_name == "seq2struct":
            return random_token_mask(struct_positions, mask_ratio, self.rng)
        if task_name == "invfold":
            return random_token_mask(seq_positions, mask_ratio, self.rng)
        if task_name == "motif_control":
            return list(seq_positions) + list(struct_positions)
        if task_name == "seq_denoise":
            return random_token_mask(seq_positions, mask_ratio, self.rng)

        length = sample["length"]
        if not self.use_pair_aware_masking and not self.use_motif_span_masking:
            nucleotide_positions = set(random_token_mask(list(range(length)), mask_ratio, self.rng))
        elif self.use_motif_span_masking and sample.get("motifs") and self.rng.random() < 0.6:
            nucleotide_positions = motif_span_mask_positions(sample["motifs"], length, self.rng)
        else:
            nucleotide_positions = random_span_positions(length, mask_ratio, self.rng)
        if self.use_pair_aware_masking:
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
