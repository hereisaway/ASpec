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


class JetSpecModel(DFlashModel):
    """DFlash block head with causal intra-block attention and forward KL."""

    def __init__(self, *args, distillation_temperature=1.0, kl_chunk_rows=256, **kwargs):
        # Main JetSpec results use uniform per-position weights (gamma = 0).
        kwargs.setdefault("loss_decay_gamma", 0.0)
        super().__init__(*args, **kwargs)
        config = getattr(self.draft_model, "config", None)
        if not bool(getattr(config, "is_causal", False)):
            raise ValueError("JetSpec requires Draft config is_causal=true")
        if self.mtp_conditioning is not None:
            raise ValueError("JetSpec does not use the native-MTP conditioning contract")
        if self.loss_objective != "decay":
            raise ValueError("JetSpec supports the decay/uniform weighting objective only")
        if self.ce_loss_alpha != 0 or self.l1_loss_alpha != 0:
            raise ValueError("JetSpec replaces CE/L1 with forward KL; set both alphas to zero")
        if self.prediction_start != 1:
            raise ValueError("JetSpec supervises the suffix after the anchor position")
        if (isinstance(distillation_temperature, bool)
                or not isinstance(distillation_temperature, (int, float))
                or not math.isfinite(float(distillation_temperature))
                or float(distillation_temperature) <= 0):
            raise ValueError("JetSpec distillation_temperature must be positive and finite")
        self.distillation_temperature = float(distillation_temperature)
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
        # Keep a differentiable path for an entirely masked window.
        if active.numel() == 0:
            active = torch.zeros(1, dtype=torch.long, device=draft_logits.device)

        temperature = self.distillation_temperature
        kl = torch.zeros(rows, dtype=torch.float32, device=draft_logits.device)
        # Bound the full-vocabulary projection and FP32 KL intermediates: each
        # chunk only materializes chunk_rows x vocab tensors.
        for start in range(0, active.numel(), self.kl_chunk_rows):
            row_ids = active[start : start + self.kl_chunk_rows]
            hidden = flat_hidden.index_select(0, row_ids)
            with torch.no_grad():
                target_logits = F.linear(hidden, lm_head_weight.detach()).float()
            log_q = F.log_softmax(
                draft_logits.index_select(0, row_ids).float() / temperature, dim=-1
            )
            log_p = F.log_softmax(target_logits / temperature, dim=-1)
            chunk = F.kl_div(log_q, log_p, reduction="none", log_target=True).sum(-1)
            # The paper's Eq. 6 keeps the standard T^2 gradient scale.
            kl = kl.index_put((row_ids,), chunk * (temperature * temperature))
        return kl, kl, logits

