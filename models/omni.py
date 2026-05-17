from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairRefineBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits + self.net(logits.unsqueeze(1)).squeeze(1)


class RNAOmniDiffusion(nn.Module):
    """RNA-OmniPrefold relation-aware masked denoising model for RNA folding."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_position_embeddings: int = 2048,
        num_segments: int = 4,
        num_tasks: int = 5,
        use_pair_head: bool = True,
        pairhead: str = "mlp",
        pairhidden: int | None = None,
        pairdrop: float = 0.1,
        distbias: bool = False,
        distbuckets: int = 32,
        distmax: int = 512,
        invalidlogit: float = -20.0,
        pairrefine: bool = False,
        pairrefinechannels: int = 16,
        pairrefineblocks: int = 1,
        pairrefinedrop: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.use_pair_head = use_pair_head
        self.pairhead = pairhead.lower()
        self.invalidlogit = float(invalidlogit)
        self.distbias = bool(distbias)
        self.distbuckets = int(distbuckets)
        self.distmax = int(distmax)
        self.pairrefine = bool(pairrefine)
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        self.segment_embedding = nn.Embedding(num_segments, hidden_size)
        self.task_embedding = nn.Embedding(num_tasks, hidden_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        try:
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
                enable_nested_tensor=False,
            )
        except TypeError:
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.sequence_head = nn.Linear(hidden_size, vocab_size)
        self.structure_head = nn.Linear(hidden_size, vocab_size)
        self.general_head = nn.Linear(hidden_size, vocab_size)
        if self.use_pair_head:
            pairhidden = int(pairhidden or hidden_size)
            if self.pairhead not in {"bilinear", "mlp", "pairmlp"}:
                raise ValueError(f"Unknown pairhead={pairhead}; expected bilinear, mlp, or pairmlp.")
            if self.pairhead == "bilinear":
                self.pair_left = nn.Linear(hidden_size, hidden_size, bias=False)
                self.pair_right = nn.Linear(hidden_size, hidden_size, bias=False)
                self.pair_scale = hidden_size ** 0.5
            elif self.pairhead == "mlp":
                self.pair_left = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.SiLU(),
                    nn.Linear(hidden_size, hidden_size),
                )
                self.pair_right = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.SiLU(),
                    nn.Linear(hidden_size, hidden_size),
                )
                self.pair_scale = hidden_size ** 0.5
            else:
                self.pair_mlp = nn.Sequential(
                    nn.LayerNorm(hidden_size * 4),
                    nn.Linear(hidden_size * 4, pairhidden),
                    nn.GELU(),
                    nn.Dropout(pairdrop),
                    nn.Linear(pairhidden, 1),
                )
            self.pair_bias = nn.Parameter(torch.zeros(1))
            if self.distbias:
                self.distance_bias = nn.Embedding(self.distbuckets, 1)
            if self.pairrefine:
                self.pair_refiner = nn.ModuleList(
                    PairRefineBlock(int(pairrefinechannels), float(pairrefinedrop))
                    for _ in range(max(1, int(pairrefineblocks)))
                )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        segment_ids: torch.Tensor,
        task_ids: torch.Tensor,
        time_steps: torch.Tensor,
        seq_positions: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor | None]:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.position_embedding.num_embeddings:
            raise ValueError(
                f"Input token length {seq_len} exceeds max_position_embeddings "
                f"{self.position_embedding.num_embeddings}."
            )
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        time_emb = self.time_mlp(time_steps.float().view(batch_size, 1)).unsqueeze(1)
        task_emb = self.task_embedding(task_ids).unsqueeze(1)
        hidden = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)
            + self.segment_embedding(segment_ids)
            + task_emb
            + time_emb
        )
        padding_mask = attention_mask == 0
        encoded = self.encoder(hidden, src_key_padding_mask=padding_mask)
        encoded = self.norm(encoded)

        general_logits = self.general_head(encoded)
        sequence_logits = self.sequence_head(encoded)
        structure_logits = self.structure_head(encoded)
        token_logits = general_logits.clone()
        token_logits = torch.where((segment_ids == 1).unsqueeze(-1), sequence_logits, token_logits)
        token_logits = torch.where((segment_ids == 2).unsqueeze(-1), structure_logits, token_logits)

        pair_logits = None
        if self.use_pair_head and seq_positions is not None:
            pair_logits = self._pair_logits(encoded, seq_positions)
        return {
            "hidden_states": encoded,
            "token_logits": token_logits,
            "sequence_logits": sequence_logits,
            "structure_logits": structure_logits,
            "general_logits": general_logits,
            "pair_logits": pair_logits,
        }

    def _pair_logits(self, hidden: torch.Tensor, seq_positions: torch.Tensor) -> torch.Tensor:
        gather_positions = seq_positions.clamp_min(0)
        expanded = gather_positions.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        seq_hidden = hidden.gather(1, expanded)
        valid = (seq_positions >= 0).float().unsqueeze(-1)
        seq_hidden = seq_hidden * valid
        if self.pairhead == "pairmlp":
            length = seq_hidden.size(1)
            hi = seq_hidden.unsqueeze(2).expand(-1, -1, length, -1)
            hj = seq_hidden.unsqueeze(1).expand(-1, length, -1, -1)
            pair_features = torch.cat([hi, hj, hi * hj, (hi - hj).abs()], dim=-1)
            logits = self.pair_mlp(pair_features).squeeze(-1)
        else:
            left = self.pair_left(seq_hidden)
            right = self.pair_right(seq_hidden)
            logits = torch.matmul(left, right.transpose(1, 2)) / self.pair_scale
        logits = logits + self.pair_bias
        logits = 0.5 * (logits + logits.transpose(1, 2))
        if self.distbias:
            length = seq_hidden.size(1)
            idx = torch.arange(length, device=hidden.device)
            dist = (idx[:, None] - idx[None, :]).abs().clamp_max(self.distmax)
            buckets = torch.div(dist * self.distbuckets, self.distmax + 1, rounding_mode="floor").clamp_max(self.distbuckets - 1)
            logits = logits + self.distance_bias(buckets).squeeze(-1)
        if self.pairrefine:
            for block in self.pair_refiner:
                logits = block(logits)
                logits = 0.5 * (logits + logits.transpose(1, 2))
        invalid = valid.squeeze(-1) == 0
        logits = logits.masked_fill(invalid.unsqueeze(1), self.invalidlogit)
        logits = logits.masked_fill(invalid.unsqueeze(2), self.invalidlogit)
        return logits


def _pair_valid_mask(lengths: torch.Tensor, max_len: int, pair_options: dict, device: torch.device) -> torch.Tensor:
    idx = torch.arange(max_len, device=device)
    valid_len = idx.unsqueeze(0) < lengths.to(device).unsqueeze(1)
    mask = valid_len.unsqueeze(1) & valid_len.unsqueeze(2)
    if pair_options.get("pairUpper", True):
        mask = mask & (idx.unsqueeze(0) > idx.unsqueeze(1)).unsqueeze(0)
    if not pair_options.get("pairDiag", False):
        mask = mask & (idx.unsqueeze(0) != idx.unsqueeze(1)).unsqueeze(0)
    loop = int(pair_options.get("pairLoop", 3))
    if loop > 0:
        mask = mask & ((idx.unsqueeze(0) - idx.unsqueeze(1)).abs() >= loop).unsqueeze(0)
    return mask


def _pair_loss_mask(pair_mask: torch.Tensor, lengths: torch.Tensor, pair_options: dict) -> torch.Tensor:
    return pair_mask & _pair_valid_mask(lengths, pair_mask.size(-1), pair_options, pair_mask.device)


def _sample_pair_loss_mask(pair_labels: torch.Tensor, lengths: torch.Tensor, pair_options: dict) -> torch.Tensor:
    valid_mask = _pair_valid_mask(lengths, pair_labels.size(-1), pair_options, pair_labels.device)
    pos_mask = valid_mask & (pair_labels > 0.5)
    neg_mask = valid_mask & (pair_labels <= 0.5)
    pos_count = int(pos_mask.sum().item())
    if pos_count == 0:
        return pos_mask
    ratio = int(pair_options.get("pairRatio", pair_options.get("pair_negative_ratio", 3)))
    neg_total = int(neg_mask.sum().item())
    if neg_total <= 0:
        return pos_mask
    target = min(neg_total, max(1, pos_count * ratio))
    probability = float(target) / float(neg_total)
    sampled_neg = neg_mask & (torch.rand(neg_mask.shape, device=pair_labels.device) < probability)
    return pos_mask | sampled_neg


def _rank_accuracy(pos_logits: torch.Tensor, neg_logits: torch.Tensor) -> torch.Tensor | None:
    if pos_logits.numel() == 0 or neg_logits.numel() == 0:
        return None
    count = min(1024, int(pos_logits.numel()), int(neg_logits.numel()))
    pos_idx = torch.randperm(pos_logits.numel(), device=pos_logits.device)[:count]
    neg_idx = torch.randperm(neg_logits.numel(), device=neg_logits.device)[:count]
    return (pos_logits[pos_idx] > neg_logits[neg_idx]).float().mean()


def _conflict_loss(
    pair_logits: torch.Tensor,
    lengths: torch.Tensor,
    pair_options: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid_upper = _pair_valid_mask(lengths, pair_logits.size(-1), pair_options, pair_logits.device)
    if not valid_upper.any():
        zero = pair_logits.new_zeros(())
        return zero, zero, zero
    probs = pair_logits.float().sigmoid() if pair_options.get("conflictUseProb", True) else F.relu(pair_logits.float())
    upper_probs = torch.where(valid_upper, probs, torch.zeros_like(probs))
    sym_probs = upper_probs + upper_probs.transpose(1, 2)
    idx = torch.arange(pair_logits.size(-1), device=pair_logits.device)
    row_valid = idx.unsqueeze(0) < lengths.to(pair_logits.device).unsqueeze(1)
    row_sum = sym_probs.sum(dim=-1)
    selected = row_sum[row_valid]
    if selected.numel() == 0:
        zero = pair_logits.new_zeros(())
        return zero, zero, zero
    margin = float(pair_options.get("conflictMargin", 1.0))
    loss = F.relu(selected - margin).mean()
    return loss, selected.mean(), selected.max()


def _pair_count_loss(
    pair_logits: torch.Tensor,
    pair_labels: torch.Tensor,
    lengths: torch.Tensor,
    pair_options: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    valid_upper = _pair_valid_mask(lengths, pair_logits.size(-1), pair_options, pair_logits.device)
    if not valid_upper.any():
        zero = pair_logits.new_zeros(())
        return zero, zero, zero, zero
    probs = pair_logits.float().sigmoid()
    pred_counts = (probs * valid_upper.float()).sum(dim=(1, 2))
    true_counts = (pair_labels.float() * valid_upper.float()).sum(dim=(1, 2))
    valid_counts = valid_upper.float().sum(dim=(1, 2)).clamp_min(1.0)
    pred_rate = pred_counts / valid_counts
    true_rate = true_counts / valid_counts
    loss = F.smooth_l1_loss(pred_rate, true_rate)
    return loss, pred_rate.mean(), true_rate.mean(), (pred_rate - true_rate).mean()


def compute_omni_loss(
    outputs: Dict[str, torch.Tensor | None],
    batch: dict,
    lambda_pair: float = 0.5,
    lambda_seq: float = 1.0,
    lambda_struct: float = 1.0,
    token_id_weights: torch.Tensor | None = None,
    pair_pos_weight: torch.Tensor | float | None = None,
    use_pair_loss: bool = True,
    pair_options: dict | None = None,
) -> Dict[str, torch.Tensor]:
    token_logits = outputs["token_logits"]
    labels = batch["labels"].to(token_logits.device)
    flat_logits = token_logits.view(-1, token_logits.size(-1))
    flat_labels = labels.view(-1)
    per_token = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100, reduction="none")
    supervised = flat_labels != -100
    flat_segments = batch["segment_ids"].to(token_logits.device).view(-1)
    segment_weights = torch.zeros_like(per_token)
    segment_weights = torch.where(flat_segments == 1, per_token.new_tensor(float(lambda_seq)), segment_weights)
    segment_weights = torch.where(flat_segments == 2, per_token.new_tensor(float(lambda_struct)), segment_weights)
    if token_id_weights is not None:
        id_weights = token_id_weights.to(token_logits.device).to(per_token.dtype)
        label_weights = torch.ones_like(per_token)
        safe_labels = flat_labels.clamp_min(0)
        label_weights[supervised] = id_weights[safe_labels[supervised]]
        segment_weights = segment_weights * label_weights
    weighted = per_token * segment_weights
    denom = segment_weights[supervised].sum().clamp_min(1.0)
    token_loss = weighted[supervised].sum() / denom

    pair_logits = outputs.get("pair_logits")
    pair_loss = token_loss.new_zeros(())
    conflict_loss = token_loss.new_zeros(())
    pair_count_loss = token_loss.new_zeros(())
    mean_row_sum = token_loss.new_zeros(())
    max_row_sum = token_loss.new_zeros(())
    pred_pair_density = token_loss.new_zeros(())
    true_pair_density = token_loss.new_zeros(())
    soft_pair_density_gap = token_loss.new_zeros(())
    pair_options = pair_options or {}
    pair_stats = {
        "pos": token_loss.new_zeros(()),
        "neg": token_loss.new_zeros(()),
        "weight": token_loss.new_zeros(()),
        "posLogit": token_loss.new_zeros(()),
        "negLogit": token_loss.new_zeros(()),
        "gap": token_loss.new_zeros(()),
        "posProb": token_loss.new_zeros(()),
        "negProb": token_loss.new_zeros(()),
        "rankAcc": None,
    }
    pair_mask = batch["pair_mask"].to(token_logits.device)
    lengths = batch.get("lengths", torch.full((pair_mask.size(0),), pair_mask.size(-1), device=token_logits.device)).to(token_logits.device)
    pair_labels = batch["pair_labels"].to(token_logits.device).float()
    if pair_options.get("sampleNegOnGpu", True):
        loss_mask = _sample_pair_loss_mask(pair_labels, lengths, pair_options)
    else:
        loss_mask = _pair_loss_mask(pair_mask, lengths, pair_options)
    if use_pair_loss and pair_logits is not None and loss_mask.any():
        selected_logits = pair_logits.float()[loss_mask] if pair_options.get("pairFloat", True) else pair_logits[loss_mask]
        selected_labels = pair_labels.float()[loss_mask]
        pos_logits = selected_logits[selected_labels > 0.5]
        neg_logits = selected_logits[selected_labels <= 0.5]
        pair_stats["pos"] = selected_logits.new_tensor(float(pos_logits.numel()))
        pair_stats["neg"] = selected_logits.new_tensor(float(neg_logits.numel()))
        if pos_logits.numel():
            pair_stats["posLogit"] = pos_logits.mean()
            pair_stats["posProb"] = pos_logits.sigmoid().mean()
        if neg_logits.numel():
            pair_stats["negLogit"] = neg_logits.mean()
            pair_stats["negProb"] = neg_logits.sigmoid().mean()
        pair_stats["gap"] = pair_stats["posLogit"] - pair_stats["negLogit"]
        pair_stats["rankAcc"] = _rank_accuracy(pos_logits, neg_logits)
        pos_weight = None
        weight_cfg = str(pair_options.get("pairWeight", "none" if pair_pos_weight is None else pair_pos_weight)).lower()
        if pos_logits.numel() == 0:
            pair_stats["warning"] = "no_positive_pairs"
        elif weight_cfg == "auto":
            weight_value = float(neg_logits.numel()) / max(1.0, float(pos_logits.numel()))
            weight_value = max(1.0, min(50.0, weight_value))
            pos_weight = torch.tensor(weight_value, dtype=torch.float32, device=token_logits.device)
            pair_stats["weight"] = pos_weight
        elif weight_cfg != "none":
            raw_weight = pair_pos_weight if pair_pos_weight is not None else pair_options.get("pairWeight", 1.0)
            pos_weight = torch.as_tensor(float(raw_weight), dtype=torch.float32, device=token_logits.device)
            pair_stats["weight"] = pos_weight
        if pos_logits.numel() > 0:
            pair_loss = F.binary_cross_entropy_with_logits(selected_logits.float(), selected_labels.float(), pos_weight=pos_weight)
    lambda_conflict = float(pair_options.get("lambdaConflict", 0.0))
    if use_pair_loss and pair_logits is not None and lambda_conflict > 0.0:
        conflict_loss, mean_row_sum, max_row_sum = _conflict_loss(pair_logits, lengths, pair_options)
    lambda_pair_count = float(pair_options.get("lambdaPairCount", 0.0))
    if use_pair_loss and pair_logits is not None and lambda_pair_count > 0.0:
        pair_count_loss, pred_pair_density, true_pair_density, soft_pair_density_gap = _pair_count_loss(
            pair_logits,
            pair_labels,
            lengths,
            pair_options,
        )
    total = (
        token_loss
        + float(lambda_pair) * pair_loss
        + lambda_conflict * conflict_loss
        + lambda_pair_count * pair_count_loss
    )
    pair_stats["pos_pair_count"] = pair_stats["pos"]
    pair_stats["neg_pair_count"] = pair_stats["neg"]
    pair_stats["pair_positive_weight_used"] = pair_stats["weight"]
    pair_stats["positive_pair_logit_mean"] = pair_stats["posLogit"]
    pair_stats["negative_pair_logit_mean"] = pair_stats["negLogit"]
    pair_stats["pair_logit_gap"] = pair_stats["gap"]
    pair_stats["positive_pair_prob_mean"] = pair_stats["posProb"]
    pair_stats["negative_pair_prob_mean"] = pair_stats["negProb"]
    pair_stats["pair_ranking_accuracy_sampled"] = pair_stats["rankAcc"]
    result = {
        "loss": total,
        "token_loss": token_loss.detach(),
        "pair_loss": pair_loss.detach(),
        "conflict_loss": conflict_loss.detach(),
        "pair_count_loss": pair_count_loss.detach(),
        "lambdaConflict": token_loss.new_tensor(lambda_conflict),
        "lambdaPairCount": token_loss.new_tensor(lambda_pair_count),
        "mean_row_pair_prob_sum": mean_row_sum.detach(),
        "max_row_pair_prob_sum": max_row_sum.detach(),
        "pred_pair_density": pred_pair_density.detach(),
        "true_pair_density": true_pair_density.detach(),
        "soft_pair_density_gap": soft_pair_density_gap.detach(),
    }
    for key, value in pair_stats.items():
        if torch.is_tensor(value):
            result[key] = value.detach()
        else:
            result[key] = value
    return result
