# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Adapted from the public DSpine implementation (Tencent/DSpine, Apache-2.0)
# and the merged SpecForge DSpine training path (PR #965, corrected by PR #1038).
# Anchor sampling, block masks and the DFlash feature path stay in ASpec.
"""DSpine training objective on the DFlash block backbone."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from aspec.models.dflash import (
    DFlashModel,
    _create_dflash_dense_mask,
    _create_dflash_mask_mod,
)
from aspec.models.ops.flex_attention import compile_friendly_create_block_mask


def refinement_loss(
    *, scores: torch.Tensor, candidates: torch.Tensor,
    target_candidates: torch.Tensor, proposal_labels: torch.Tensor,
    proposal_weights: torch.Tensor, ce_weight: float, mass_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Candidate CE/L1 with the reference normalization and coverage mask."""
    mass = target_candidates.sum(-1)
    covered = candidates == proposal_labels.unsqueeze(-1)
    covered_any = covered.any(-1)
    ref_ce = F.cross_entropy(
        scores.reshape(-1, candidates.shape[-1]),
        covered.long().argmax(-1).reshape(-1),
        reduction="none",
    ).reshape(proposal_labels.shape)
    ref_ce = (ref_ce * covered_any * proposal_weights).sum()
    normalized_target = target_candidates / mass.clamp_min(1e-12).unsqueeze(-1)
    ref_l1 = (
        (torch.softmax(scores, dim=-1) - normalized_target).abs().sum(-1)
        * (mass > mass_threshold)
        * proposal_weights
    ).sum()
    return ce_weight * ref_ce + (1 - ce_weight) * ref_l1, ref_ce, ref_l1


class DSpineModel(DFlashModel):
    """Block-parallel DFlash backbone with adjacent injection and refinement."""

    _materialize_cpu_block_mask = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = getattr(self.draft_model, "config", None)
        if getattr(config, "model_type", None) != "dspine":
            raise ValueError("DSpine training requires a DSpine draft config")
        if not hasattr(self.draft_model, "refine"):
            raise ValueError("DSpine training requires a DSpineDraftModel")
        if config.block_size != self.block_size:
            raise ValueError("DSpine training block_size must match its Draft config")
        if getattr(config, "is_causal", None) is not True:
            raise ValueError("DSpine requires causal draft attention")
        if self.mtp_conditioning is not None:
            raise ValueError("DSpine does not use the native-MTP conditioning contract")
        if self.draft_context_parallel_size != 1:
            raise NotImplementedError(
                "DSpine currently requires draft_context_parallel_size=1"
            )
        if self.loss_objective != "decay":
            raise ValueError("DSpine supports the decay weighting objective only")
        if self.ce_loss_alpha != 1 or self.l1_loss_alpha != 0:
            raise ValueError(
                "DSpine replaces the CE/L1 mix with its own objective; set "
                "ce_loss_alpha=1 and l1_loss_alpha=0"
            )

    def _prepare_draft_prefix(
        self, input_ids, last_hidden_states, attention_mask, draft_prefix,
        *packed_metadata,
    ):
        ctx_doc_ids, _, packed_context_positions, packed_document_lengths = packed_metadata
        if any(
            value is not None
            for value in (ctx_doc_ids, packed_context_positions, packed_document_lengths)
        ):
            raise NotImplementedError(
                "DSpine supports padded, non-packed inputs only; packed documents "
                "or sharded context are rejected before backbone execution"
            )
        return super()._prepare_draft_prefix(
            input_ids, last_hidden_states, attention_mask, draft_prefix,
            *packed_metadata,
        )

    def _block_mask(self, anchors, keep, seq_len, batch):
        draft_len = anchors.shape[1] * self.block_size
        if anchors.device.type == "cuda" and self.attention_backend == "flex_attention":
            mask_mod = _create_dflash_mask_mod(
                anchor_positions=anchors,
                block_keep_mask=keep,
                ctx_len=seq_len,
                block_size=self.block_size,
                is_causal=True,
            )
            return compile_friendly_create_block_mask(
                mask_mod=mask_mod, B=batch, H=None,
                Q_LEN=draft_len, KV_LEN=seq_len + draft_len,
                device=anchors.device,
            ), None
        if self.attention_backend not in {"sdpa", "eager"}:
            raise NotImplementedError(
                "DSpine CPU/reference execution supports sdpa/eager only"
            )
        return None, _create_dflash_dense_mask(
            anchors, keep, seq_len, self.block_size, is_causal=True
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states_list: List[torch.Tensor] | torch.Tensor,
        loss_mask: torch.Tensor,
        lm_head_weight: torch.Tensor,
        last_hidden_states: Optional[torch.Tensor] = None,
        ctx_doc_ids: Optional[torch.Tensor] = None,
        base_position_ids: Optional[torch.Tensor] = None,
        packed_context_positions: Optional[torch.Tensor] = None,
        injected_anchors: Optional[torch.Tensor] = None,
        injected_keep_mask: Optional[torch.Tensor] = None,
        anchor_seed: Optional[int] = None,
        packed_document_lengths: Optional[tuple[int, ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        draft_prefix=None,
        global_step: int = 0,
        total_steps: int | None = None,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        dict, Tuple[torch.Tensor, torch.Tensor],
    ]:
        if last_hidden_states is None:
            raise ValueError(
                "DSpine requires Target last_hidden_states for its teacher distribution"
            )
        self._prepare_draft_prefix(
            input_ids, last_hidden_states, attention_mask, draft_prefix,
            ctx_doc_ids, base_position_ids, packed_context_positions,
            packed_document_lengths,
        )
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        anchors, keep = self._sample_anchor_positions(
            seq_len, loss_mask, device, ctx_doc_ids=ctx_doc_ids,
            injected_anchors=injected_anchors,
            injected_keep_mask=injected_keep_mask,
            anchor_seed=anchor_seed,
            packed_document_lengths=packed_document_lengths,
            attention_mask=attention_mask,
        )
        n_blocks = anchors.shape[1]
        offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)
        label_indices = anchors.unsqueeze(-1) + offsets
        safe_indices = label_indices.clamp(max=seq_len - 1)
        labels = torch.gather(
            input_ids.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_indices
        )
        supervised = (
            (label_indices < seq_len)
            & keep.unsqueeze(-1)
            & (
                torch.gather(
                    loss_mask.unsqueeze(1).expand(-1, n_blocks, -1),
                    2,
                    safe_indices,
                )
                > 0.5
            )
        )
        # The anchor row is context; only proposal positions 1..block_size-1 are supervised.
        supervised = supervised[..., 1:].int().cumprod(dim=-1).bool()
        weights = supervised.float()
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            weights = weights * torch.exp(
                -offsets[..., 1:].float() / self.loss_decay_gamma
            )

        if not bool(self.draft_model.transfer_ready):
            self.draft_model.initialize_transfer_space(lm_head_weight)
        context_feature = self.draft_model.extract_context_feature(hidden_states_list)
        noise = self._create_noise_embed(input_ids, anchors, keep)
        context_pos, draft_pos = self._create_position_ids(
            anchors, seq_len, base_position_ids=base_position_ids
        )
        block_mask, dense_mask = self._block_mask(anchors, keep, seq_len, bsz)
        beta, replacement_probability = self.draft_model.dspine_config.schedule(
            int(global_step), total_steps
        )
        replace = None
        if self.training and replacement_probability > 0:
            replace = (
                torch.rand(anchors.shape, device=device) < replacement_probability
            ) & keep
        output = self.draft_model(
            context_feature=context_feature,
            draft_position_ids=draft_pos,
            context_position_ids=context_pos,
            block_mask=block_mask,
            noise_embedding=noise,
            block_token_ids=labels,
            replace_mask=replace,
            draft_attention_mask=dense_mask,
        )

        proposal_labels = labels[..., 1:].reshape(-1, self.block_size - 1)
        proposal_weights = weights.reshape(-1, self.block_size - 1)
        proposal_supervised = supervised.reshape(-1, self.block_size - 1)
        hidden = output.hidden_states[..., 1:, :].reshape(
            -1, self.block_size - 1, output.hidden_states.shape[-1]
        )
        draft_logits = F.linear(hidden, lm_head_weight)
        ce_per_token = F.cross_entropy(
            draft_logits.reshape(-1, draft_logits.shape[-1]),
            proposal_labels.reshape(-1),
            reduction="none",
        ).reshape(proposal_labels.shape)
        flat_weights = proposal_weights.reshape(-1)
        denominator = flat_weights.sum()
        ce = (ce_per_token * proposal_weights).sum()

        target_indices = (safe_indices[..., 1:] - 1).clamp_min(0)
        teacher_hidden = torch.gather(
            last_hidden_states, 1,
            target_indices.reshape(bsz, -1, 1).expand(
                -1, -1, last_hidden_states.shape[-1]
            ),
        ).reshape(-1, self.block_size - 1, last_hidden_states.shape[-1])
        teacher_logits = F.linear(teacher_hidden, lm_head_weight)
        teacher_probs = torch.softmax(teacher_logits.float(), dim=-1)
        draft_probs = torch.softmax(draft_logits.float(), dim=-1)
        l1 = ((draft_probs - teacher_probs).abs().sum(-1) * proposal_weights).sum()

        config = self.draft_model.dspine_config
        candidates = teacher_logits.topk(config.top_k, dim=-1).indices
        pre_injection = output.pre_injection[..., 1:, :].reshape(
            -1, self.block_size - 1, output.pre_injection.shape[-1]
        )
        predecessors = labels[..., :-1].reshape(-1, self.block_size - 1)
        refined = self.draft_model.refine(pre_injection, predecessors)
        candidate_embeds = F.embedding(candidates, lm_head_weight)
        scores = (refined.unsqueeze(-2) * candidate_embeds).sum(-1).float()
        target_candidates = teacher_probs.gather(-1, candidates)
        refinement, ref_ce, ref_l1 = refinement_loss(
            scores=scores,
            candidates=candidates,
            target_candidates=target_candidates,
            proposal_labels=proposal_labels,
            proposal_weights=proposal_weights,
            ce_weight=config.ce_weight,
            mass_threshold=config.candidate_mass_threshold,
        )

        reference_codes = F.embedding(
            proposal_labels, self.draft_model.transfer_codes
        ).float()
        layer_features = (
            output.layer_features[..., 1:, :]
            .permute(1, 2, 3, 0, 4)
            .reshape(-1, self.block_size - 1, output.layer_features.shape[0],
                     output.layer_features.shape[-1])
        )
        alignment = 1 - F.cosine_similarity(
            layer_features.float(), reference_codes.unsqueeze(2), dim=-1
        )
        alignment = (alignment * proposal_weights.unsqueeze(-1)).sum()
        if denominator > 0:
            alignment_ce = ce.detach() / denominator
        else:
            alignment_ce = ce.detach()
        numerator = (
            config.ce_weight * ce
            + (1 - config.ce_weight) * l1
            + refinement
            + beta * alignment_ce * alignment
        )
        loss = numerator / denominator.clamp_min(1e-6)

        with torch.no_grad():
            binary = proposal_supervised.float()
            count = binary.sum().clamp(min=1.0)
            correct = (
                (draft_logits.argmax(-1) == proposal_labels) & (binary > 0.5)
            )
            accuracy = correct.float().sum() / count
            count_pp = binary.sum(dim=0)
            loss_pp = (ce_per_token * binary).sum(dim=0) / count_pp.clamp(min=1.0)
            acc_pp = correct.float().sum(dim=0) / count_pp.clamp(min=1.0)
        loss_per_position = torch.cat(
            (loss_pp.new_zeros(1), loss_pp)
        )
        acc_per_position = torch.cat(
            (acc_pp.new_zeros(1), acc_pp)
        )
        count_per_position = torch.cat(
            (proposal_weights.new_zeros(1), count_pp)
        )
        components = {
            "dspine/backbone_ce": (ce.detach(), denominator.detach()),
            "dspine/backbone_l1": (l1.detach(), denominator.detach()),
            "dspine/refinement_loss": (refinement.detach(), denominator.detach()),
            "dspine/alignment_loss": (alignment.detach(), denominator.detach()),
            "dspine/replacement_probability": (
                torch.tensor(replacement_probability, device=device),
                torch.tensor(1.0, device=device),
            ),
        }
        return (
            loss, accuracy, loss_per_position, acc_per_position,
            count_per_position, components, (numerator, denominator.detach()),
        )
