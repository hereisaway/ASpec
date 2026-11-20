# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""JetSpec trainer — causal parallel Draft head with target forward KL."""

import math
from argparse import Namespace

from aspec.models.draft.jetspec import JetSpecConfig, JetSpecDraftModel
from aspec.models.jetspec import JetSpecModel
from aspec.training.dflash_trainer import DFlashTrainer

_MIN_TEMPERATURE = 1e-6
_MAX_TEMPERATURE = math.sqrt(3.4028234663852886e38)


class JetSpecTrainer(DFlashTrainer):
    """JetSpec-specific trainer; reuses the DFlash optimizer/checkpoint path."""

    _draft_config_class = JetSpecConfig

    def __init__(self, args: Namespace):
        super().__init__(args)
        self.distillation_temperature = float(
            getattr(args, "jetspec_distillation_temperature", 1.0)
        )
        self.kl_chunk_rows = getattr(args, "jetspec_kl_chunk_rows", 256)
        self.loss_decay_gamma = float(
            getattr(args, "jetspec_loss_decay_gamma", 0.0)
        )
        temperature = self.distillation_temperature
        if (
            isinstance(getattr(args, "jetspec_distillation_temperature", 1.0), bool)
            or not math.isfinite(temperature)
            or not _MIN_TEMPERATURE <= temperature <= _MAX_TEMPERATURE
        ):
            raise ValueError(
                "jetspec_distillation_temperature must be finite and in "
                f"[{_MIN_TEMPERATURE:g}, {_MAX_TEMPERATURE:g}]"
            )
        if (
            isinstance(self.kl_chunk_rows, bool)
            or not isinstance(self.kl_chunk_rows, int)
            or self.kl_chunk_rows < 1
        ):
            raise ValueError("jetspec_kl_chunk_rows must be a positive integer")
        if not math.isfinite(self.loss_decay_gamma) or self.loss_decay_gamma < 0:
            raise ValueError("jetspec_loss_decay_gamma must be finite and non-negative")
        if self.loss_objective != "decay":
            raise ValueError("JetSpec supports the decay/uniform weighting objective only")
        if self.ce_loss_alpha != 0.0 or self.l1_loss_alpha != 0.0:
            raise ValueError(
                "JetSpec replaces CE/L1 with forward KL; set "
                "dflash_ce_loss_alpha=0.0 and dflash_l1_loss_alpha=0.0"
            )

    def _build_draft_model(self, config):
        if config.block_size != self.block_size:
            raise ValueError(
                "training.dflash_block_size must match jetspec_config.block_size "
                f"({self.block_size} != {config.block_size})"
            )
        if config.num_target_layers != self.num_target_layers:
            raise ValueError(
                "training.dflash_num_target_layers must match the number of "
                f"jetspec_config.target_layer_ids ({self.num_target_layers} != "
                f"{config.num_target_layers})"
            )
        return JetSpecDraftModel(config)

    def _build_training_wrapper(self, draft_model):
        wrapper = JetSpecModel(
            draft_model=draft_model,
            block_size=self.block_size,
            num_anchors=self.num_anchors,
            loss_objective=self.loss_objective,
            dpace_alpha=self.dpace_alpha,
            loss_decay_gamma=self.loss_decay_gamma,
            ce_loss_alpha=self.ce_loss_alpha,
            l1_loss_alpha=self.l1_loss_alpha,
            distillation_temperature=self.distillation_temperature,
            kl_chunk_rows=self.kl_chunk_rows,
            deterministic_anchor_sampling=getattr(
                self.args, "deterministic_anchor_sampling", False
            ),
            anchor_seed=getattr(self.args, "seed", 0),
            anchor_sampling_scope=getattr(self.args, "anchor_sampling_scope", "row"),
            attention_backend=getattr(self.args, "attention_backend", "flex_attention"),
            draft_context_parallel_size=getattr(
                self.args, "draft_context_parallel_size", 1
            ),
        )
        self._anchor_slot_offset = wrapper.prediction_start
        return wrapper
