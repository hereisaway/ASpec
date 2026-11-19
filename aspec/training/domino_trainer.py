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

    def _accumulation_steps(self) -> int:
        args = getattr(self, "args", None)
        value = getattr(args, "draft_accumulation_steps", 1) if args is not None else 1
        return max(1, int(value))

    def _lambda_base_for_micro_step(self, micro_step: int) -> float:
        """Linear schedule using SpecForge's 1-based microbatch counter.

        ``total_steps`` is measured in optimizer updates; with accumulation > 1
        the counter advances per microbatch, matching the reference trainer.
        """
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
        progress = min(max(int(micro_step), 1) / decay_steps, 1.0)
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
            lambda_base=self._lambda_base_for_micro_step(1),
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
        last_micro = max(1, int(self.global_step) * self._accumulation_steps())
        metrics["train/lambda_base"] = self._lambda_base_for_micro_step(last_micro)
        return metrics

    def _train_step(self, batch, accumulation_steps, step, batch_idx, num_batches):
        # Match SpecForge: advance the schedule counter before every microbatch.
        micro_step = (
            int(self.global_step) * int(accumulation_steps) + int(batch_idx) + 1
        )
        self.dflash.set_lambda_base(self._lambda_base_for_micro_step(micro_step))
        return super()._train_step(
            batch=batch,
            accumulation_steps=accumulation_steps,
            step=step,
            batch_idx=batch_idx,
            num_batches=num_batches,
        )

    def eval_from_cache(self):
        # Evaluate at the value used by the most recently completed microbatch.
        last_micro = max(1, int(self.global_step) * self._accumulation_steps())
        self.dflash.set_lambda_base(self._lambda_base_for_micro_step(last_micro))
        previous = getattr(self.dflash, "collect_extra_position_metrics", False)
        self.dflash.collect_extra_position_metrics = self.eval_backbone
        try:
            return super().eval_from_cache()
        finally:
            self.dflash.collect_extra_position_metrics = previous

    def _aggregate_eval_metrics(self, all_step_metrics):
        metrics = super()._aggregate_eval_metrics(all_step_metrics)
        mixed = metrics.get("eval/mixed_ce")
        if mixed is not None:
            metrics["eval/avg_loss"] = mixed
        last_micro = max(1, int(self.global_step) * self._accumulation_steps())
        metrics["eval/lambda_base"] = self._lambda_base_for_micro_step(last_micro)
        if not all_step_metrics or "backbone_loss_pp" not in all_step_metrics[0]:
            return metrics

        backbone_loss, backbone_acc, backbone_count = self._reduce_position_metrics(
            all_step_metrics,
            loss_key="backbone_loss_pp",
            acc_key="backbone_acc_pp",
            count_key="backbone_count_pp",
        )
        offset = self._anchor_slot_offset
        pred_loss = backbone_loss[offset:]
        pred_acc = backbone_acc[offset:]
        pred_count = backbone_count[offset:]
        cumulative = 1.0
        simulated_acc_len = 0.0
        for i in range(pred_acc.shape[0]):
            cumulative *= pred_acc[i].item()
            simulated_acc_len += cumulative
        avg_loss, avg_acc = self._compute_scalar_metrics(
            pred_loss, pred_acc, pred_count
        )
        prefix = self._prediction_metric_prefix()
        metrics["eval/backbone_avg_loss"] = avg_loss
        metrics["eval/backbone_avg_acc"] = avg_acc
        metrics[f"eval/backbone_{prefix}simulated_acc_len"] = simulated_acc_len
        for i in range(pred_loss.shape[0]):
            metrics[f"eval/backbone_{prefix}ploss_{i}"] = pred_loss[i].item()
            metrics[f"eval/backbone_{prefix}acc_{i}"] = pred_acc[i].item()
        return metrics
