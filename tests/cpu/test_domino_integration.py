# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Domino integration contracts: registration, Plan, dispatch and resume."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from aspec.config.draft_algorithm import draft_model_semantics, resolve_draft_algorithm
from aspec.config.train_config import load_config
from aspec.models.domino import DominoModel
from aspec.models.draft.auto import AutoDraftModelConfig, AutoEagle3DraftModel
from aspec.models.draft.domino import DominoConfig, DominoDraftModel
from aspec.planning import (
    PlanCompileError,
    _load_draft_descriptor,
    _runtime_values,
    compile_execution_plan,
)
from aspec.training.checkpoint import (
    _logical_training_topology,
    _validate_exact_resume_topology,
)
from aspec.training.domino_trainer import DominoTrainer
from aspec.training.trainer_actor import _trainer_class_for_config

_REPO_ROOT = Path(__file__).parents[2]
_EXAMPLE = _REPO_ROOT / "configs" / "examples" / "domino_qwen3_8b_4k_tcp.yaml"
_DRAFT_JSON = _REPO_ROOT / "aspec" / "config" / "domino_qwen3_8b_draft_config.json"

_ILLEGAL_OVERRIDES = [
    ("training.dflash_packing=true", PlanCompileError),
    ("training.draft_context_parallel_size=2", NotImplementedError),
    ("training.attention_backend=sdpa", PlanCompileError),
    ("training.attention_backend=fa4", PlanCompileError),
    ("training.attention_backend=cudnn_flex_attention", PlanCompileError),
    ("training.compile_model=true", PlanCompileError),
    ("training.dflash_loss_objective=dpace", PlanCompileError),
    ("training.dflash_ce_loss_alpha=0.5", PlanCompileError),
    ("training.dflash_l1_loss_alpha=0.1", PlanCompileError),
    ("training.dflash_block_size=8", PlanCompileError),
    ("training.domino_lambda_base=2.0", PlanCompileError),
]


def _tiny_config(**overrides):
    raw = dict(
        architectures=["DominoDraftModel"], model_type="domino",
        hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, vocab_size=32,
        target_hidden_size=16, target_num_hidden_layers=4, target_layer_ids=[1],
        num_target_layers=1, mask_token_id=31, max_position_embeddings=128,
        block_size=5, emb_dim=6, gru_hidden_dim=8,
    )
    raw.update(overrides)
    return AutoDraftModelConfig.from_dict(raw)


def _fake_qwen_target(tmp_path):
    target = tmp_path / "Qwen3-8B-test"
    target.mkdir()
    (target / "config.json").write_text(json.dumps({
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 4096,
        "vocab_size": 151936,
        "num_hidden_layers": 36,
        "max_position_embeddings": 40960,
    }), encoding="utf-8")
    return target


def _config(tmp_path, *overrides):
    args = [f"model.target_model_path={_fake_qwen_target(tmp_path)}", *overrides]
    return load_config(config_path=str(_EXAMPLE), cli_args=args), args


def test_registration_resolves_domino():
    assert resolve_draft_algorithm({"architectures": ["DominoDraftModel"]}) == "domino"
    shipped = AutoDraftModelConfig.from_file(str(_DRAFT_JSON))
    assert isinstance(shipped, DominoConfig)
    assert shipped.block_size == 16
    assert shipped.shift_label is True
    assert shipped.pure_draft_prefix_len == 1
    assert shipped.emb_dim == 256
    assert shipped.gru_hidden_dim == 1024
    assert shipped.target_layer_ids == [1, 9, 17, 25, 33]

    tiny = _tiny_config()
    model = AutoEagle3DraftModel.from_config(tiny)
    assert isinstance(model, DominoDraftModel)
    assert model.suffix_start == 1
    assert _trainer_class_for_config(tiny) is DominoTrainer
    semantics = draft_model_semantics(tiny)
    assert semantics["projector_type"] == "domino"
    assert semantics["emb_dim"] == 6
    assert semantics["gru_hidden_dim"] == 8
    assert semantics["shift_label"] is True
    assert semantics["pure_draft_prefix_len"] == 1


def test_plan_compiles_with_domino_objective(tmp_path):
    config, args = _config(tmp_path)
    plan = compile_execution_plan(config, config_path=str(_EXAMPLE), cli_args=args).to_dict()
    draft = plan["workload"]["models"]["draft"]
    assert draft["family"] == "domino"
    objective = plan["workload"]["objective"]
    assert objective["loss"] == "mixed_corrected_backbone_ce"
    assert objective["block_size"] == 16
    assert objective["lambda_base"] == 0.3
    assert objective["shift_label"] is True
    assert objective["pure_draft_prefix_len"] == 1
    assert objective["ce_loss_alpha"] == 1.0
    assert objective["l1_loss_alpha"] == 0.0
    assert objective["correction_head"] == {"emb_dim": 256, "gru_hidden_dim": 1024}
    features = plan["workload"]["features"]
    assert features["aux_hidden_state_layers"] == [1, 9, 17, 25, 33]
    assert "last_hidden_states" not in features["tensors"]
    attention = next(
        item for item in plan["capabilities"] if item["name"] == "draft_attention_execution"
    )
    assert attention["requirement"] == "required"
    assert attention["effective_state"] != "omitted"

    runtime = _runtime_values(config, _load_draft_descriptor(config))
    assert runtime["draft_algorithm"] == "domino"
    assert runtime["domino_lambda_base"] == 0.3


def test_trainer_aggregates_real_mixed_component(monkeypatch):
    import aspec.training.dflash_trainer as trainer_module

    monkeypatch.setattr(trainer_module.dist, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer_module.dist, "get_rank", lambda: 1)
    trainer = object.__new__(DominoTrainer)
    trainer._anchor_slot_offset = 0
    trainer.global_step = 0
    trainer.optimizer = SimpleNamespace(get_learning_rate=lambda: 0.0)
    trainer.dflash = SimpleNamespace(mtp_conditioning=None)
    trainer.draft_model = SimpleNamespace(config=SimpleNamespace())
    trainer.loss_decay_gamma = 0.0

    train_metrics = [
        {
            "loss_per_position": torch.tensor([1.0]),
            "acc_per_position": torch.tensor([1.0]),
            "count_per_position": torch.tensor([1.0]),
            "mixed_ce": (torch.tensor(2.0), torch.tensor(1.0)),
        },
        {
            "loss_per_position": torch.tensor([1.0]),
            "acc_per_position": torch.tensor([1.0]),
            "count_per_position": torch.tensor([1.0]),
            "mixed_ce": (torch.tensor(9.0), torch.tensor(3.0)),
        },
    ]
    metrics = trainer._aggregate_metrics(train_metrics, step=1)
    assert metrics["train/mixed_ce"] == pytest.approx(11 / 4)
    assert metrics["train/avg_loss"] == pytest.approx(11 / 4)

    eval_metrics = [
        {
            "loss_pp": torch.tensor([1.0]),
            "acc_pp": torch.tensor([1.0]),
            "count_pp": torch.tensor([1.0]),
            "mixed_ce": (torch.tensor(2.0), torch.tensor(1.0)),
        },
        {
            "loss_pp": torch.tensor([1.0]),
            "acc_pp": torch.tensor([1.0]),
            "count_pp": torch.tensor([1.0]),
            "mixed_ce": (torch.tensor(9.0), torch.tensor(3.0)),
        },
    ]
    eval_output = trainer._aggregate_eval_metrics(eval_metrics)
    assert eval_output["eval/mixed_ce"] == pytest.approx(11 / 4)
    assert eval_output["eval/avg_loss"] == pytest.approx(11 / 4)


def test_plan_rejects_explicit_cudnn_gqa_mode(tmp_path):
    raw = json.loads(_DRAFT_JSON.read_text(encoding="utf-8"))
    raw["cudnn_gqa_mode"] = "native"
    draft_config = tmp_path / "domino_cudnn.json"
    draft_config.write_text(json.dumps(raw), encoding="utf-8")
    target = f"model.target_model_path={_fake_qwen_target(tmp_path)}"
    args = [target, f"model.draft_model_config={draft_config}"]
    with pytest.raises(PlanCompileError, match="cudnn_gqa_mode"):
        config = load_config(config_path=str(_EXAMPLE), cli_args=args)
        compile_execution_plan(config, config_path=str(_EXAMPLE), cli_args=args)


@pytest.mark.parametrize("override, error", _ILLEGAL_OVERRIDES)
def test_plan_rejects_unsupported_domino_combinations(tmp_path, override, error):
    target = f"model.target_model_path={_fake_qwen_target(tmp_path)}"
    args = [target, override]
    with pytest.raises(error):
        config = load_config(config_path=str(_EXAMPLE), cli_args=args)
        compile_execution_plan(config, config_path=str(_EXAMPLE), cli_args=args)


def _trainer_args():
    return SimpleNamespace(
        draft_algorithm="domino", draft_model_config_obj=_tiny_config(), world_size=2,
        dflash_block_size=5, dflash_num_anchors=2, dflash_num_target_layers=1,
        dflash_loss_objective="decay", dflash_dpace_alpha=0.5,
        dflash_loss_decay_gamma=7.0, dflash_ce_loss_alpha=1.0, dflash_l1_loss_alpha=0.0,
        domino_lambda_base=0.3, anchor_sampling_scope="row",
        deterministic_anchor_sampling=False, max_seq_length=128, micro_batch_size=1,
        draft_accumulation_steps=1, max_grad_norm=1.0, seed=42,
        draft_context_parallel_size=1, training_num_nodes=1, training_num_gpus_per_node=2,
        attention_backend="sdpa", fsdp_strategy="REPLICATE", sp_ulysses_size=1,
        sp_ring_size=1,
    )


def test_trainer_builds_domino_wrapper_and_pins_block_size():
    trainer = object.__new__(DominoTrainer)
    trainer.block_size = 5
    trainer.num_anchors = 2
    trainer.num_target_layers = 1
    trainer.loss_objective = "decay"
    trainer.dpace_alpha = 0.5
    trainer.loss_decay_gamma = 7.0
    trainer.ce_loss_alpha = 1.0
    trainer.l1_loss_alpha = 0.0
    trainer.lambda_base = 0.3
    trainer.args = SimpleNamespace(
        deterministic_anchor_sampling=False, seed=42, anchor_sampling_scope="row",
        attention_backend="sdpa", draft_context_parallel_size=1,
    )
    wrapper = trainer._build_training_wrapper(DominoDraftModel(_tiny_config()))
    assert isinstance(wrapper, DominoModel)
    assert wrapper.lambda_base == 0.3
    assert wrapper.block_size == 5
    assert trainer._anchor_slot_offset == 0

    unshifted = object.__new__(DominoTrainer)
    unshifted.__dict__.update(trainer.__dict__)
    unshifted_wrapper = unshifted._build_training_wrapper(
        DominoDraftModel(_tiny_config(shift_label=False, pure_draft_prefix_len=0))
    )
    assert unshifted_wrapper.prediction_start == 1
    assert unshifted._anchor_slot_offset == 1

    trainer.block_size = 4
    with pytest.raises(ValueError):
        trainer._build_draft_model(_tiny_config())


@pytest.mark.parametrize("missing", ["empty", "lambda_base", "shift_label"])
def test_exact_resume_rejects_incomplete_domino_topology(missing):
    args = _trainer_args()
    topology = json.loads(json.dumps(_logical_training_topology(args, 2)))
    if missing == "empty":
        topology = {}
    elif missing == "lambda_base":
        del topology["training_objective"]["lambda_base"]
    else:
        del topology["draft_model_semantics"]["shift_label"]
    with pytest.raises(RuntimeError, match="complete Domino training_topology"):
        _validate_exact_resume_topology(
            args, {"world_size": 2, "training_topology": topology}
        )


def test_exact_resume_requires_topology_for_domino():
    args = _trainer_args()
    with pytest.raises(RuntimeError, match="training_topology for Domino"):
        _validate_exact_resume_topology(args, {"world_size": 2})


def test_checkpoint_topology_pins_domino_objective():
    args = _trainer_args()
    current = _logical_training_topology(args, 2)
    objective = current["training_objective"]
    assert objective["loss"] == "mixed_corrected_backbone_ce"
    assert objective["lambda_base"] == 0.3
    assert objective["block_size"] == 5
    assert current["draft_model_semantics"]["projector_type"] == "domino"
    assert current["draft_model_semantics"]["shift_label"] is True

    metadata = {"world_size": 2, "training_topology": json.loads(json.dumps(current))}
    _validate_exact_resume_topology(args, metadata)
    changed = json.loads(json.dumps(metadata))
    changed["training_topology"]["training_objective"]["lambda_base"] = 0.5
    with pytest.raises(RuntimeError):
        _validate_exact_resume_topology(args, changed)
