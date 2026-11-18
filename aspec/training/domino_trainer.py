# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Domino trainer — DFlash backbone plus a GRU-conditioned correction head."""

from argparse import Namespace

from aspec.models.domino import DominoModel
from aspec.models.draft.domino import DominoConfig, DominoDraftModel
from aspec.training.dflash_trainer import DFlashTrainer


class DominoTrainer(DFlashTrainer):
    """Domino-specific trainer; reuses the DFlash optimizer/checkpoint path."""

    _draft_config_class = DominoConfig

    def __init__(self, args: Namespace):
        super().__init__(args)
        self.lambda_base = getattr(args, "domino_lambda_base", 0.3)

    def _build_draft_model(self, config):
        if config.block_size != self.block_size:
            raise ValueError(
                "training.dflash_block_size must match domino_config.block_size "
                f"({self.block_size} != {config.block_size})"
            )
        if config.num_target_layers != self.num_target_layers:
            raise ValueError(
                "training.dflash_num_target_layers must match the number of "
                f"domino_config.target_layer_ids ({self.num_target_layers} != "
                f"{config.num_target_layers})"
            )
        return DominoDraftModel(config)

    def _build_training_wrapper(self, draft_model):
        wrapper = DominoModel(
            draft_model=draft_model,
            block_size=self.block_size,
            num_anchors=self.num_anchors,
            loss_objective=self.loss_objective,
            dpace_alpha=self.dpace_alpha,
            loss_decay_gamma=self.loss_decay_gamma,
            ce_loss_alpha=self.ce_loss_alpha,
            l1_loss_alpha=self.l1_loss_alpha,
            lambda_base=self.lambda_base,
            deterministic_anchor_sampling=getattr(
                self.args, "deterministic_anchor_sampling", False
            ),
            anchor_seed=getattr(self.args, "seed", 0),
            anchor_sampling_scope=getattr(self.args, "anchor_sampling_scope", "row"),
            attention_backend=getattr(self.args, "attention_backend", "sdpa"),
            draft_context_parallel_size=getattr(
                self.args, "draft_context_parallel_size", 1
            ),
        )
        # Shifted labels supervise block position 0; unshifted labels skip the
        # anchor slot. Metric aggregators must drop the same prefix.
        self._anchor_slot_offset = wrapper.prediction_start
        return wrapper
