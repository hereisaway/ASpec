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

    def __init__(self, *args, distillation_temperature=1.0, **kwargs):
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
        with torch.no_grad():
            target_logits = F.linear(
                aligned_hidden.detach(), lm_head_weight.detach()
            ).float().reshape(-1, logits.size(-1))
        draft_logits = logits.float().reshape(-1, logits.size(-1))
        temperature = self.distillation_temperature
        log_q = F.log_softmax(draft_logits / temperature, dim=-1)
        p = F.softmax(target_logits / temperature, dim=-1)
        kl = F.kl_div(log_q, p, reduction="none").sum(-1)
        # The paper's Eq. 6 keeps the standard T^2 gradient scale.
        kl = kl * (temperature * temperature)
        return kl, kl, logits
