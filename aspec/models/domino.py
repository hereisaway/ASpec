# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Adapted from the merged SpecForge Domino training path
# (sgl-project/SpecForge PR #571, specforge/core/domino.py, MIT; see
# licenses/SpecForge-MIT.txt). The objective, predecessor alignment and
# label-shift handling follow that implementation and the public Domino method
# (arXiv:2605.29707). Anchor sampling, block masks, reductions and the DFlash
# training workflow stay in the retained ASpec DFlash path.
"""Domino correction-head training objective on the DFlash block backbone."""

import math

from aspec.models.dflash import DFlashModel


class DominoModel(DFlashModel):
    """Block-parallel DFlash objective mixed with a causal correction head."""

    def __init__(self, *args, lambda_base=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        config = getattr(self.draft_model, "config", None)
        if getattr(config, "projector_type", None) != "domino":
            raise ValueError("Domino training requires a Domino draft config")
        if not hasattr(self.draft_model, "correct_logits"):
            raise ValueError("Domino training requires a DominoDraftModel")
        if config.block_size != self.block_size:
            raise ValueError("Domino training block_size must match its Draft config")
        if self.mtp_conditioning is not None:
            raise ValueError("Domino does not use the native-MTP conditioning contract")
        if self.draft_context_parallel_size != 1:
            raise NotImplementedError("Domino currently requires draft_context_parallel_size=1")
        if self.loss_objective != "decay":
            raise ValueError("Domino supports the decay/uniform weighting objective only")
        if self.ce_loss_alpha != 1 or self.l1_loss_alpha != 0:
            raise ValueError(
                "Domino replaces the CE/L1 mix with its own objective; set "
                "ce_loss_alpha=1 and l1_loss_alpha=0"
            )
        # Shifted labels start at anchor + 1 and supervise position 0; the
        # unshifted layout keeps DFlash's same-position labels and skips the
        # anchor slot. Both match the two documented Domino layouts.
        self.label_start = 1 if self.draft_model.shift_label else 0
        self.prediction_start = 1 - self.label_start
        self.set_lambda_base(lambda_base)

    def _prepare_draft_prefix(
        self,
        input_ids,
        last_hidden_states,
        attention_mask,
        draft_prefix,
        *packed_metadata,
    ):
        ctx_doc_ids, _, packed_context_positions, packed_document_lengths = packed_metadata
        if any(
            value is not None
            for value in (ctx_doc_ids, packed_context_positions, packed_document_lengths)
        ):
            raise NotImplementedError(
                "Domino supports padded, non-packed inputs only; packed documents "
                "or sharded context are rejected before backbone execution"
            )
        return super()._prepare_draft_prefix(
            input_ids,
            last_hidden_states,
            attention_mask,
            draft_prefix,
            *packed_metadata,
        )

    def set_lambda_base(self, lambda_base) -> None:
        """Set the backbone-only loss weight for the current step."""
        if (
            isinstance(lambda_base, bool)
            or not isinstance(lambda_base, (int, float))
            or not math.isfinite(float(lambda_base))
        ):
            raise ValueError("Domino lambda_base must be a finite number in [0, 1]")
        lambda_base = float(lambda_base)
        if not 0.0 <= lambda_base <= 1.0:
            raise ValueError("Domino lambda_base must be a finite number in [0, 1]")
        self.lambda_base = lambda_base

    def _training_loss(
        self,
        *,
        logits,
        draft_hidden,
        target_ids,
        last_hidden_states,
        lm_head_weight,
        safe_label_indices,
        packed_context_positions,
        sequence_length,
        input_ids,
        anchor_positions,
    ):
        base_logits = logits
        corrected_logits = self.draft_model.correct_logits(
            draft_hidden, base_logits, input_ids, anchor_positions
        )
        # The retained CE path supplies the reduction-ready per-token values for
        # both the corrected and backbone-only predictions.
        _, corrected_per_token, _ = super()._training_loss(
            logits=corrected_logits,
            draft_hidden=draft_hidden,
            target_ids=target_ids,
            last_hidden_states=last_hidden_states,
            lm_head_weight=lm_head_weight,
            safe_label_indices=safe_label_indices,
            packed_context_positions=packed_context_positions,
            sequence_length=sequence_length,
            input_ids=input_ids,
            anchor_positions=anchor_positions,
        )
        _, base_per_token, _ = super()._training_loss(
            logits=base_logits,
            draft_hidden=draft_hidden,
            target_ids=target_ids,
            last_hidden_states=last_hidden_states,
            lm_head_weight=lm_head_weight,
            safe_label_indices=safe_label_indices,
            packed_context_positions=packed_context_positions,
            sequence_length=sequence_length,
            input_ids=input_ids,
            anchor_positions=anchor_positions,
        )
        lambda_base = self.lambda_base
        loss_per_token = (
            (1.0 - lambda_base) * corrected_per_token + lambda_base * base_per_token
        )
        # Report the corrected objective's per-position value; accuracy uses the
        # corrected logits as well.
        return loss_per_token, corrected_per_token, corrected_logits
