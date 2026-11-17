# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Domino causal correction head and mixed backbone/head objective.

Reference: Domino, arXiv:2605.29707, and the merged SpecForge training path
(PR #571). These checks compare the implementation with explicit equations;
they do not reproduce the paper's training data, budget or serving results.
"""

import copy

import pytest
import torch
from torch.nn import functional as F

from aspec.models.dflash import DFlashModel
from aspec.models.domino import DominoModel
from aspec.models.draft.dflash import DFlashConfig, DFlashDraftModel
from aspec.models.draft.domino import DominoConfig, DominoDraftModel


def make_config(**overrides):
    values = dict(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, vocab_size=32,
        num_target_layers=1, target_hidden_size=16, target_num_hidden_layers=4,
        target_layer_ids=[1], mask_token_id=31, max_position_embeddings=128,
        block_size=5, rms_norm_eps=1e-6, emb_dim=6, gru_hidden_dim=8,
    )
    values.update(overrides)
    return DominoConfig(**values)


def make_model(
    *, shift_label=True, pure_draft_prefix_len=None, lambda_base=0.3,
    loss_decay_gamma=0.0, seed=17, **config_overrides
):
    if pure_draft_prefix_len is None:
        pure_draft_prefix_len = 1 if shift_label else 0
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        draft = DominoDraftModel(
            make_config(
                shift_label=shift_label,
                pure_draft_prefix_len=pure_draft_prefix_len,
                **config_overrides,
            )
        )
        draft.embed_tokens.requires_grad_(False)
    return DominoModel(
        draft, block_size=draft.config.block_size, num_anchors=2,
        attention_backend="sdpa", loss_decay_gamma=loss_decay_gamma,
        lambda_base=lambda_base,
    )


def make_inputs(seed=31, length=12, batch=1):
    generator = torch.Generator().manual_seed(seed)
    ids = torch.arange(batch * length).reshape(batch, length) % 32
    return dict(
        input_ids=ids,
        hidden_states_list=torch.randn(batch, length, 16, generator=generator),
        loss_mask=torch.ones_like(ids),
        lm_head_weight=torch.randn(32, 16, generator=generator),
        last_hidden_states=torch.randn(batch, length, 16, generator=generator),
        injected_anchors=torch.tensor([[2, 7]]).expand(batch, -1),
        injected_keep_mask=torch.ones(batch, 2, dtype=torch.bool),
    )


def run_and_capture(model, inputs):
    captured = []
    handle = model.draft_model.register_forward_hook(
        lambda _, __, output: captured.append(output)
    )
    result = model(**inputs)
    handle.remove()
    return result, captured[0]


def reference_objective(model, draft_hidden, inputs):
    """Dense Domino equations with explicit label/anchor/mask alignment."""
    draft_model = model.draft_model
    anchors = inputs["injected_anchors"]
    keep = inputs["injected_keep_mask"]
    ids = inputs["input_ids"]
    block_size = model.block_size
    suffix = draft_model.suffix_start

    offsets = torch.arange(model.label_start, model.label_start + block_size)[None, None]
    label_indices = anchors.unsqueeze(-1) + offsets
    valid = label_indices < ids.shape[1]
    safe_indices = label_indices.clamp(max=ids.shape[1] - 1)
    expanded_ids = ids.unsqueeze(1).expand(-1, anchors.shape[1], -1)
    target_ids = torch.gather(expanded_ids, 2, safe_indices)

    base_logits = F.linear(draft_hidden, inputs["lm_head_weight"])
    hidden4d = draft_hidden.reshape(-1, anchors.shape[1], block_size, 16)
    logits4d = base_logits.reshape(-1, anchors.shape[1], block_size, 32)

    block_offsets = torch.arange(block_size)[None, None]
    block_ids = torch.gather(
        expanded_ids,
        2,
        (anchors.unsqueeze(-1) + block_offsets).clamp(max=ids.shape[1] - 1),
    )
    if draft_model.shift_label:
        gru_inputs = block_ids
        state_slice = slice(suffix, None)
    else:
        gru_inputs = block_ids[..., : block_size - 1]
        state_slice = slice(suffix - 1, None)
    states, _ = draft_model.prefix_gru(
        draft_model.embed_tokens(gru_inputs).reshape(
            -1, gru_inputs.size(-1), draft_model.hidden_size
        )
    )
    states = states.reshape(-1, anchors.shape[1], gru_inputs.size(-1), 8)[
        :, :, state_slice, :
    ]
    delta = draft_model.embed_proj(torch.cat([hidden4d[:, :, suffix:, :], states], dim=-1))
    corrected = torch.cat(
        [logits4d[:, :, :suffix, :], logits4d[:, :, suffix:, :] + delta], dim=2
    )
    corrected_ce = F.cross_entropy(
        corrected.reshape(-1, 32), target_ids.reshape(-1), reduction="none"
    )
    base_ce = F.cross_entropy(
        base_logits.reshape(-1, 32), target_ids.reshape(-1), reduction="none"
    )

    gathered_mask = torch.gather(
        inputs["loss_mask"].unsqueeze(1).expand(-1, anchors.shape[1], -1),
        2, safe_indices,
    )
    positions = torch.arange(block_size)[None, None]
    weights = keep.unsqueeze(-1).float() * valid.float()
    weights = weights * (positions >= model.prediction_start).float()
    weights = weights * gathered_mask
    decayed = weights
    if model.loss_decay_gamma is not None and model.loss_decay_gamma > 0:
        decayed = weights * torch.exp(
            -(positions - model.prediction_start).clamp_min(0).float()
            / model.loss_decay_gamma
        )
    flat_weights = decayed.reshape(-1)
    mixed = (1.0 - model.lambda_base) * corrected_ce + model.lambda_base * base_ce
    loss = (mixed * flat_weights).sum() / flat_weights.sum().clamp_min(1e-6)

    binary = weights.reshape(-1)
    count = binary.reshape(-1, anchors.shape[1], block_size).sum(dim=(0, 1))
    per_position = (
        corrected_ce.reshape(-1, anchors.shape[1], block_size) * weights
    ).sum(dim=(0, 1)) / count.clamp(min=1.0)
    return loss, corrected_ce.reshape(-1, anchors.shape[1], block_size), per_position


def _assert_parameter_gradients(actual, expected):
    actual_parameters = dict(actual.named_parameters())
    expected_parameters = dict(expected.named_parameters())
    assert actual_parameters.keys() == expected_parameters.keys()
    for name, parameter in actual_parameters.items():
        reference = expected_parameters[name]
        assert (parameter.grad is None) == (reference.grad is None), name
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
            torch.testing.assert_close(
                parameter.grad, reference.grad, rtol=2e-5, atol=2e-6, msg=name
            )


def test_config_pins_the_correction_contract():
    config = make_config()
    assert config.model_type == "domino"
    assert config.architectures == ["DominoDraftModel"]
    assert config.projector_type == "domino"
    assert config.shift_label is True
    assert config.pure_draft_prefix_len == 1
    restored = DominoConfig.from_dict(config.to_dict())
    assert restored.architectures == ["DominoDraftModel"]
    assert restored.shift_label is True
    assert restored.pure_draft_prefix_len == 1
    assert restored.emb_dim == 6 and restored.gru_hidden_dim == 8


@pytest.mark.parametrize("override", [
    {"projector_type": "linear"},
    {"emb_dim": 0},
    {"emb_dim": 6.5},
    {"gru_hidden_dim": -1},
    {"shift_label": 1},
    {"pure_draft_prefix_len": 1.5},
    {"shift_label": False, "pure_draft_prefix_len": 4},
    {"architectures": ["DFlashDraftModel"]},
    {"is_causal": True},
])
def test_config_fail_closed(override):
    with pytest.raises(ValueError):
        make_config(**override)


def test_config_rejects_unsupported_backbones():
    with pytest.raises(ValueError):
        make_config(ffn_type="moe", num_experts=4, num_experts_per_tok=2,
                    moe_intermediate_size=32)
    with pytest.raises(ValueError):
        make_config(layer_types=["sliding_attention"], sliding_window=64)


def test_shifted_layout_supervises_the_anchor_slot():
    shifted = make_model(shift_label=True)
    assert shifted.label_start == 1
    assert shifted.prediction_start == 0
    result, _ = run_and_capture(shifted, make_inputs())
    assert int(result[4][0]) > 0

    unshifted = make_model(shift_label=False)
    assert unshifted.label_start == 0
    assert unshifted.prediction_start == 1
    result, _ = run_and_capture(unshifted, make_inputs())
    assert int(result[4][0]) == 0


@pytest.mark.parametrize("shift_label,pure", [(True, 1), (False, 0), (True, 0), (False, 1)])
@pytest.mark.parametrize("loss_decay_gamma", [0.0, 3.0])
@pytest.mark.parametrize("lambda_base", [0.0, 0.3, 1.0])
def test_loss_and_gradients_match_independent_equations(
    shift_label, pure, loss_decay_gamma, lambda_base
):
    actual = make_model(
        shift_label=shift_label, pure_draft_prefix_len=pure,
        lambda_base=lambda_base, loss_decay_gamma=loss_decay_gamma,
    )
    reference = copy.deepcopy(actual)
    inputs, reference_inputs = make_inputs(), make_inputs()

    result, captured = run_and_capture(actual, inputs)
    _, reference_captured = run_and_capture(reference, reference_inputs)
    expected_loss, expected_ce, expected_per_position = reference_objective(
        reference, reference_captured, reference_inputs
    )
    actual_loss, actual_ce, actual_per_position = reference_objective(
        actual, captured, inputs
    )

    torch.testing.assert_close(result[0], expected_loss, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual_ce, expected_ce, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(result[2], expected_per_position, rtol=2e-5, atol=2e-6)
    assert actual_per_position.shape == (actual.block_size,)

    result[0].backward()
    expected_loss.backward()
    _assert_parameter_gradients(actual, reference)


def test_lambda_one_recovers_the_backbone_only_objective():
    model = make_model(lambda_base=1.0, loss_decay_gamma=0.0)
    inputs = make_inputs()
    result, captured = run_and_capture(model, inputs)
    expected, _, _ = reference_objective(model, captured, inputs)
    torch.testing.assert_close(result[0], expected, rtol=2e-5, atol=2e-6)
    result[0].backward()
    for parameter in (
        model.draft_model.prefix_gru.weight_ih_l0,
        model.draft_model.embed_proj[0].weight,
        model.draft_model.embed_proj[2].weight,
    ):
        assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
    assert torch.count_nonzero(model.draft_model.context_proj.weight.grad) > 0


def test_two_row_layout_matches_reference_and_backward():
    actual = make_model(shift_label=True, lambda_base=0.4)
    reference = copy.deepcopy(actual)
    inputs, reference_inputs = make_inputs(batch=2), make_inputs(batch=2)
    result, captured = run_and_capture(actual, inputs)
    _, reference_captured = run_and_capture(reference, reference_inputs)
    expected_loss, expected_ce, _ = reference_objective(
        reference, reference_captured, reference_inputs
    )
    actual_loss, actual_ce, _ = reference_objective(actual, captured, inputs)
    assert result[0].ndim == 0
    torch.testing.assert_close(result[0], expected_loss, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual_ce, expected_ce, rtol=2e-5, atol=2e-6)
    result[0].backward()
    expected_loss.backward()
    _assert_parameter_gradients(actual, reference)


def test_loss_mask_excludes_unsupervised_labels():
    model = make_model(shift_label=True)
    inputs = make_inputs()
    inputs["loss_mask"][0, 6] = 0
    result, captured = run_and_capture(model, inputs)
    expected, _, _ = reference_objective(model, captured, inputs)
    torch.testing.assert_close(result[0], expected, rtol=2e-5, atol=2e-6)
    offsets = torch.arange(model.label_start, model.label_start + model.block_size)
    label_indices = inputs["injected_anchors"].unsqueeze(-1) + offsets
    gathered = torch.gather(
        inputs["loss_mask"].unsqueeze(1).expand(-1, 2, -1),
        2, label_indices.clamp(max=inputs["input_ids"].size(1) - 1),
    )
    valid = (label_indices < inputs["input_ids"].size(1)) & (gathered > 0.5)
    valid = valid & (torch.arange(model.block_size) >= model.prediction_start)
    assert int(result[4].sum()) == int(valid.sum())


def test_empty_supervision_is_neutral_and_finite():
    model = make_model()
    inputs = make_inputs()
    inputs["loss_mask"] = torch.zeros_like(inputs["loss_mask"])
    inputs["injected_keep_mask"] = torch.zeros_like(inputs["injected_keep_mask"])
    result, _ = run_and_capture(model, inputs)
    assert float(result[0].detach()) == 0.0
    assert torch.isfinite(result[0])
    assert int(result[4].sum()) == 0
    result[0].backward()
    for parameter in model.parameters():
        assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0


def test_target_features_are_detached_and_embedding_stays_frozen():
    model = make_model()
    inputs = make_inputs()
    inputs["last_hidden_states"].requires_grad_(True)
    result, _ = run_and_capture(model, inputs)
    result[0].backward()
    assert inputs["last_hidden_states"].grad is None
    assert model.draft_model.embed_tokens.weight.grad is None


@pytest.mark.parametrize("length", [3, 6])
def test_short_sequences_keep_loss_finite(length):
    model = make_model()
    inputs = make_inputs(length=length)
    inputs["injected_anchors"] = torch.tensor([[0, max(length - 3, 1)]])
    result, captured = run_and_capture(model, inputs)
    expected, _, _ = reference_objective(model, captured, inputs)
    torch.testing.assert_close(result[0], expected, rtol=2e-5, atol=2e-6)
    assert torch.isfinite(result[0])
    result[0].backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("lambda_base", [-0.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_lambda_base_fails_closed(lambda_base):
    draft = DominoDraftModel(make_config())
    with pytest.raises(ValueError, match="lambda_base"):
        DominoModel(
            draft, block_size=5, num_anchors=2, attention_backend="sdpa",
            lambda_base=lambda_base,
        )


def test_set_lambda_base_validates_and_changes_the_objective():
    model = make_model(lambda_base=0.0)
    model.set_lambda_base(1.0)
    assert model.lambda_base == 1.0
    with pytest.raises(ValueError, match="lambda_base"):
        model.set_lambda_base(float("nan"))
    with pytest.raises(ValueError, match="lambda_base"):
        model.set_lambda_base(2.0)


def test_non_domino_inputs_fail_closed():
    config = DFlashConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, vocab_size=32,
        num_target_layers=1, target_hidden_size=16, target_num_hidden_layers=4,
        target_layer_ids=[1], mask_token_id=31, max_position_embeddings=128,
        block_size=5,
    )
    draft = DFlashDraftModel(config)
    with pytest.raises(ValueError, match="Domino draft config"):
        DominoModel(draft, block_size=5, num_anchors=2, attention_backend="sdpa")
    with pytest.raises(ValueError, match="block_size"):
        DominoModel(
            DominoDraftModel(make_config()), block_size=4, num_anchors=2,
            attention_backend="sdpa",
        )
    with pytest.raises(ValueError, match="CE/L1"):
        DominoModel(
            DominoDraftModel(make_config()), block_size=5, num_anchors=2,
            attention_backend="sdpa", ce_loss_alpha=0.5,
        )
    with pytest.raises(ValueError, match="CE/L1"):
        DominoModel(
            DominoDraftModel(make_config()), block_size=5, num_anchors=2,
            attention_backend="sdpa", l1_loss_alpha=1.0,
        )


def test_correction_keeps_backbone_only_prefix_and_changes_the_suffix():
    base = make_model(shift_label=True)
    corrected = copy.deepcopy(base)
    inputs = make_inputs()
    inputs["injected_anchors"] = torch.tensor([[2, 7]])
    _, base_hidden = run_and_capture(base, inputs)
    _, corrected_hidden = run_and_capture(corrected, inputs)
    torch.testing.assert_close(base_hidden, corrected_hidden, rtol=0, atol=0)

    base_logits = F.linear(base_hidden, inputs["lm_head_weight"])
    corrected_logits = base.draft_model.correct_logits(
        base_hidden, base_logits, inputs["input_ids"], inputs["injected_anchors"]
    )
    base4d = base_logits.reshape(1, 2, 5, 32)
    corrected4d = corrected_logits.reshape(1, 2, 5, 32)
    suffix = base.draft_model.suffix_start
    torch.testing.assert_close(
        base4d[:, :, :suffix], corrected4d[:, :, :suffix], rtol=0, atol=0
    )
    assert not torch.equal(base4d[:, :, suffix:], corrected4d[:, :, suffix:])


def test_plain_dflash_wrapper_keeps_unshifted_layout():
    config = DFlashConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, vocab_size=32,
        num_target_layers=1, target_hidden_size=16, target_num_hidden_layers=4,
        target_layer_ids=[1], mask_token_id=31, max_position_embeddings=128,
        block_size=5,
    )
    draft = DFlashDraftModel(config)
    model = DFlashModel(draft, block_size=5, num_anchors=2, attention_backend="sdpa")
    assert model.label_start == 0
    result, _ = run_and_capture(model, make_inputs())
    assert int(result[4][0]) == 0
