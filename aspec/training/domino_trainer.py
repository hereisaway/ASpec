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
    _extra_loss_component_keys = ["mixed_ce", "backbone_ce", "corrected_ce"]

    def __init__(self, args: Namespace):
        super().__init__(args)
        self.lambda_base = getattr(args, "domino_lambda_base", 0.3)
        self.lambda_base_start = getattr(args, "domino_lambda_base_start", None)
        self.lambda_base_decay_ratio = getattr(
            args, "domino_lambda_base_decay_ratio", None
        )
        self.eval_backbone = bool(getattr(args, "domino_eval_backbone", False))
        self.lambda_total_steps = int(
            getattr(args, "lr_total_steps", None)
            or getattr(args, "num_train_steps", 1)
        )
        if (self.lambda_base_start is None) != (self.lambda_base_decay_ratio is None):
            raise ValueError(
                "domino_lambda_base_start and domino_lambda_base_decay_ratio "
                "must be set together"
            )
        if self.lambda_base_start is not None:
            if not 0.0 <= float(self.lambda_base_start) <= 1.0:
                raise ValueError("domino_lambda_base_start must be in [0, 1]")
            if float(self.lambda_base_decay_ratio) <= 0.0:
                raise ValueError("domino_lambda_base_decay_ratio must be > 0")

    def _schedule_total_steps(self) -> int:
        args = getattr(self, "args", None)
        value = getattr(args, "lr_total_steps", None) if args is not None else None
        if not value:
            value = getattr(args, "num_train_steps", None) if args is not None else None
        if not value:
            value = getattr(self, "lambda_total_steps", 1)
        return max(1, int(value))

    def _lambda_base_for_step(self, step: int) -> float:
        """Linear backbone-weight schedule matching SpecForge's Domino trainer."""
        start = getattr(self, "lambda_base_start", None)
        if start is None:
            return float(getattr(self, "lambda_base", 0.3))
        decay_steps = max(
            1,
            int(
                self._schedule_total_steps()
                * float(getattr(self, "lambda_base_decay_ratio", 1.0))
            ),
        )
        progress = min(max(int(step), 0) / decay_steps, 1.0)
        value = float(start) * (1.0 - progress)
        return max(0.0, min(1.0, value))

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
            lambda_base=self._lambda_base_for_step(0),
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
        # Shifted labels supervise block position 0; unshifted labels skip the
        # anchor slot. Metric aggregators must drop the same prefix.
        self._anchor_slot_offset = wrapper.prediction_start
        return wrapper

    def _aggregate_metrics(self, all_step_metrics, step, *, grad_norm=None):
        metrics = super()._aggregate_metrics(
            all_step_metrics, step, grad_norm=grad_norm
        )
        mixed = metrics.get("train/mixed_ce")
        if mixed is not None:
            metrics["train/avg_loss"] = mixed
        metrics["train/lambda_base"] = self._lambda_base_for_step(
            max(self.global_step - 1, 0)
        )
        return metrics

    def _train_step(self, *args, **kwargs):
        # The schedule is a function of the optimizer step, so every micro-batch
        # in one step uses the same lambda. Exact resume restores global_step.
        self.dflash.set_lambda_base(self._lambda_base_for_step(self.global_step))
        return super()._train_step(*args, **kwargs)

    def eval_from_cache(self):
        # Evaluate with the lambda used by the most recently completed step,
        # not the value the next step would use.
        self.dflash.set_lambda_base(
            self._lambda_base_for_step(max(self.global_step - 1, 0))
        )
        corrected = super().eval_from_cache()
        if not self.eval_backbone or not corrected:
            return corrected
        # Reuse the cached Target features for a backbone-only reporting pass.
        # The mixed loss and the correction head are unchanged by this flag.
        self.dflash.report_backbone = True
        try:
            backbone = super().eval_from_cache()
        finally:
            self.dflash.report_backbone = False
        return self._merge_backbone_eval_metrics(corrected, backbone)

    @staticmethod
    def _merge_backbone_eval_metrics(corrected: dict, backbone: dict) -> dict:
        merged = dict(corrected)
        for key, value in backbone.items():
            if not key.startswith("eval/"):
                continue
            suffix = key[len("eval/"):]
            if (
                suffix in {"avg_loss", "avg_acc"}
                or "ploss_" in suffix
                or suffix.startswith("acc_")
                or "simulated_acc_len" in suffix
            ):
                merged[f"eval/backbone_{suffix}"] = value
        return merged

    def _aggregate_eval_metrics(self, all_step_metrics):
        metrics = super()._aggregate_eval_metrics(all_step_metrics)
        if not getattr(self.dflash, "report_backbone", False):
            mixed = metrics.get("eval/mixed_ce")
            if mixed is not None:
                metrics["eval/avg_loss"] = mixed
        metrics["eval/lambda_base"] = self._lambda_base_for_step(
            max(self.global_step - 1, 0)
        )
        return metrics
