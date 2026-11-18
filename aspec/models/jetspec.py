# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Objective reimplemented from the public JetSpec paper (Hu et al., 2026,
# arXiv:2606.18394, Section 3.3 and Appendix D): temperature-scaled forward KL
# between the frozen Target and the causal parallel Draft, with optional
# DFlash-style depth weighting and the T_KD^2 gradient scale. Configuration,
# anchor sampling, masks, reductions and recovery stay in the ASpec DFlash path.
"""JetSpec causal-parallel Draft training objective."""

import math

import torch
import torch.nn.functional as F

from aspec.models.dflash import DFlashModel

_MIN_TEMPERATURE = 1e-6


class _ChunkedForwardKL(torch.autograd.Function):
    """Forward KL whose full-vocabulary intermediates are recomputed in backward.

    Saving ``log_q``/``log_p`` per chunk would keep O(tokens * vocabulary) FP32
    state alive until backward. The custom backward instead retains the existing
    Draft logits and teacher hidden states, then recomputes one chunk at a time.
    """

    @staticmethod
    def forward(ctx, draft_logits, hidden, lm_head_weight, temperature, chunk_rows, active):
        rows = draft_logits.shape[0]
        kl = torch.zeros(rows, dtype=torch.float32, device=draft_logits.device)
        for start in range(0, active.numel(), chunk_rows):
            row_ids = active[start : start + chunk_rows]
            target_logits = F.linear(
                hidden.index_select(0, row_ids), lm_head_weight
            ).float()
            log_p = F.log_softmax(target_logits / temperature, dim=-1)
            log_q = F.log_softmax(
                draft_logits.index_select(0, row_ids).float() / temperature, dim=-1
            )
            chunk = F.kl_div(log_q, log_p, reduction="none", log_target=True).sum(-1)
            kl.index_put_((row_ids,), chunk * (temperature * temperature))
        ctx.save_for_backward(draft_logits, hidden, lm_head_weight, active)
        ctx.temperature = temperature
        ctx.chunk_rows = chunk_rows
        return kl

    @staticmethod
    def backward(ctx, grad_kl):
        draft_logits, hidden, lm_head_weight, active = ctx.saved_tensors
        temperature = ctx.temperature
        gradient = torch.zeros_like(draft_logits)
        for start in range(0, active.numel(), ctx.chunk_rows):
            row_ids = active[start : start + ctx.chunk_rows]
            target_logits = F.linear(
                hidden.index_select(0, row_ids), lm_head_weight
            ).float()
            p = F.softmax(target_logits / temperature, dim=-1)
            q = F.softmax(
                draft_logits.index_select(0, row_ids).float() / temperature, dim=-1
            )
            scale = grad_kl.index_select(0, row_ids).float().unsqueeze(-1)
            gradient.index_put_(
                (row_ids,), ((q - p) * temperature * scale).to(gradient.dtype)
            )
        return gradient, None, None, None, None, None


class JetSpecModel(DFlashModel):
    """DFlash block head with causal intra-block attention and forward KL."""

    def __init__(self, *args, distillation_temperature=1.0, kl_chunk_rows=256, **kwargs):
        # Main JetSpec results use uniform per-position weights (gamma = 0).
        kwargs.setdefault("loss_decay_gamma", 0.0)
        super().__init__(*args, **kwargs)
        config = getattr(self.draft_model, "config", None)
        if not bool(getattr(config, "is_causal", False)):
            raise ValueError("JetSpec requires Draft config is_causal=true")
        if getattr(config, "architectures", None) != ["JetSpecDraftModel"]:
            raise ValueError("JetSpecModel requires a JetSpecDraftModel config")
        if self.mtp_conditioning is not None:
            raise ValueError("JetSpec does not use the native-MTP conditioning contract")
        if self.loss_objective != "decay":
            raise ValueError("JetSpec supports the decay/uniform weighting objective only")
        if self.ce_loss_alpha != 0 or self.l1_loss_alpha != 0:
            raise ValueError("JetSpec replaces CE/L1 with forward KL; set both alphas to zero")
        if self.prediction_start != 1:
            raise ValueError("JetSpec supervises the suffix after the anchor position")
        temperature = float(distillation_temperature)
        if (isinstance(distillation_temperature, bool)
                or not isinstance(distillation_temperature, (int, float))
                or not math.isfinite(temperature)
                or temperature < _MIN_TEMPERATURE):
            raise ValueError(
                "JetSpec distillation_temperature must be finite and at least "
                f"{_MIN_TEMPERATURE:g}"
            )
        self.distillation_temperature = temperature
        if (isinstance(kl_chunk_rows, bool) or not isinstance(kl_chunk_rows, int)
                or kl_chunk_rows < 1):
            raise ValueError("JetSpec kl_chunk_rows must be a positive integer")
        self.kl_chunk_rows = int(kl_chunk_rows)
        if config is not None and hasattr(config, "block_size"):
            if int(config.block_size) != self.block_size:
                raise ValueError("JetSpec training block_size must match its Draft config")

    @property
    def uses_target_distribution(self) -> bool:
        return True

    @property
    def uses_target_hidden_states(self) -> bool:
        return True

    def _block_mask_options(self) -> dict:
        return {"is_causal": True}

    def _block_mask_options_for_layer(self, layer_id: int) -> dict:
        return {"is_causal": True}

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
        objective_weight_mask=None,
    ):
        if last_hidden_states is None:
            raise ValueError(
                "JetSpec forward KL requires Target last_hidden_states; set "
                "inference.store_last_hidden_states=true in the run config"
            )
        target_positions = (safe_label_indices - 1).clamp(min=0)
        aligned_hidden = self._align_target_hidden_states(
            last_hidden_states,
            target_positions,
            global_sequence_length=sequence_length,
            packed_context_positions=packed_context_positions,
        )
        vocabulary = logits.size(-1)
        draft_logits = logits.reshape(-1, vocabulary)
        flat_hidden = aligned_hidden.detach().reshape(-1, aligned_hidden.size(-1))
        if flat_hidden.size(0) != draft_logits.size(0):
            raise ValueError("JetSpec teacher rows do not match Draft logits")
        rows = draft_logits.size(0)
        active = (
            torch.arange(rows, device=draft_logits.device)
            if objective_weight_mask is None
            else (objective_weight_mask.reshape(-1) > 0).nonzero(as_tuple=False).flatten()
        )
        # The custom backward recomputes one chunk at a time, so neither the
        # forward nor the saved state keeps every full-vocabulary chunk alive.
        kl = _ChunkedForwardKL.apply(
            draft_logits,
            flat_hidden,
            lm_head_weight.detach(),
            self.distillation_temperature,
            self.kl_chunk_rows,
            active,
        )
        return kl, kl, logits

