# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""JetSpec causal-parallel head and forward-KL objective.

Reference: JetSpec, arXiv:2606.18394 (Section 3.3 and Appendix D). These checks
compare the implementation with explicit equations; they do not reproduce the
paper's training data, budget or serving results.
"""

import copy

import pytest
import torch
from torch.nn import functional as F

from aspec.models.dflash import _create_dflash_mask_mod
from aspec.models.draft.jetspec import JetSpecConfig, JetSpecDraftModel
from aspec.models.jetspec import JetSpecModel


def make_config(**overrides):
    values = dict(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, vocab_size=32,
        num_target_layers=1, target_hidden_size=16, target_num_hidden_layers=4,
        target_layer_ids=[1], mask_token_id=31, max_position_embeddings=128,
        block_size=5, rms_norm_eps=1e-6,
    )
    values.update(overrides)
    return JetSpecConfig(**values)


def make_model(*, temperature=1.0, loss_decay_gamma=0.0, kl_chunk_rows=256, seed=17, **config_overrides):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        draft = JetSpecDraftModel(make_config(**config_overrides))
        draft.embed_tokens.requires_grad_(False)
    return JetSpecModel(
        draft, block_size=draft.config.block_size, num_anchors=2,
        attention_backend="sdpa", ce_loss_alpha=0.0, l1_loss_alpha=0.0,
        loss_decay_gamma=loss_decay_gamma,
        distillation_temperature=temperature, kl_chunk_rows=kl_chunk_rows,
    )


def make_inputs(seed=31, length=12, batch=1):
    generator = torch.Generator().manual_seed(seed)
    ids = (torch.arange(batch * length).reshape(batch, length) % 32)
    return dict(
        input_ids=ids,
        hidden_states_list=torch.randn(batch, length, 16, generator=generator),
        loss_mask=torch.ones_like(ids),
        lm_head_weight=torch.randn(32, 16, generator=generator),
        last_hidden_states=torch.randn(batch, length, 16, generator=generator),
        injected_anchors=torch.tensor([[3, 5]]).expand(batch, -1),
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


def reference_kl(model, draft_hidden, inputs):
    """Dense forward-KL equations with explicit label/anchor/mask alignment."""
    anchors = inputs["injected_anchors"]
    keep = inputs["injected_keep_mask"]
    ids = inputs["input_ids"]
    offsets = torch.arange(model.block_size)[None, None]
    label_indices = anchors.unsqueeze(-1) + offsets
    valid = label_indices < ids.shape[1]
    safe_indices = label_indices.clamp(max=ids.shape[1] - 1)

    expanded_ids = ids.unsqueeze(1).expand(-1, anchors.shape[1], -1)
    target_ids = torch.gather(expanded_ids, 2, safe_indices)
    teacher_positions = (safe_indices - 1).clamp(min=0)
    teacher_hidden = torch.gather(
        inputs["last_hidden_states"].detach(), 1,
        teacher_positions.reshape(ids.shape[0], -1, 1).expand(-1, -1, 16),
    )
    target_logits = F.linear(teacher_hidden, inputs["lm_head_weight"].detach()).float()
    draft_logits = F.linear(draft_hidden, inputs["lm_head_weight"]).float()

    temperature = model.distillation_temperature
    log_q = F.log_softmax(draft_logits / temperature, dim=-1)
    p = F.softmax(target_logits / temperature, dim=-1)
    kl = F.kl_div(log_q, p, reduction="none").sum(-1)
    kl = kl * (temperature * temperature)

    gathered_mask = torch.gather(
        inputs["loss_mask"].unsqueeze(1).expand(-1, anchors.shape[1], -1),
        2, safe_indices,
    )
    positions = offsets.squeeze(0)
    weights = keep.unsqueeze(-1).float() * valid.float()
    weights = weights * (positions >= model.prediction_start).float()
    weights = weights * gathered_mask
    if model.loss_decay_gamma is not None and model.loss_decay_gamma > 0:
        weights = weights * torch.exp(
            -(positions - model.prediction_start).clamp_min(0).float()
            / model.loss_decay_gamma
        )
    kl = kl.reshape_as(weights)
    loss = (kl * weights).sum() / weights.sum()
    assert target_ids.shape == kl.shape
    return loss, kl


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


def test_config_pins_the_causal_block_contract():
    config = make_config()
    assert config.model_type == "jetspec"
    assert config.architectures == ["JetSpecDraftModel"]
    assert config.is_causal is True
    assert config.block_size == 5
    assert config.layer_types == ["full_attention"]
    restored = JetSpecConfig.from_dict(config.to_dict())
    assert restored.is_causal is True
    assert restored.block_size == 5
    assert restored.architectures == ["JetSpecDraftModel"]


@pytest.mark.parametrize("override", [
    {"is_causal": False},
    {"is_causal": None},
    {"block_size": 1},
    {"block_size": 4.5},
    {"architectures": ["DFlashDraftModel"]},
])
def test_config_fail_closed(override):
    with pytest.raises(ValueError):
        make_config(**override)


def test_config_rejects_unsupported_head_shapes():
    with pytest.raises(ValueError):
        make_config(ffn_type="moe", num_experts=4, num_experts_per_tok=2,
                    moe_intermediate_size=32)
    with pytest.raises(ValueError):
        make_config(layer_types=["sliding_attention"], sliding_window=64)


def test_blocks_use_causal_intra_block_attention():
    model = make_model()
    assert model._block_mask_options_for_layer(0) == {"is_causal": True}
    mask = _create_dflash_mask_mod(
        torch.tensor([[3, 8]]), torch.ones(1, 2, dtype=torch.bool),
        ctx_len=10, block_size=5, is_causal=True,
    )
    # Query positions are block-relative; KV is [context | block 0 | block 1].
    for query in range(5):
        assert mask(0, 0, query, 2)            # verified prefix before the anchor
        assert not mask(0, 0, query, 3)        # anchor is draft context, not target context
        assert mask(0, 0, query, 10)           # block 0 anchor
        assert mask(0, 0, query, 10 + query)   # own causal prefix
        assert not mask(0, 0, query, 10 + query + 1) if query < 4 else True
        assert not mask(0, 0, query, 15)       # another block is invisible
    assert not mask(0, 0, 0, 11)               # no causal future inside block 0


@pytest.mark.parametrize("temperature", [1.0, 2.0])
@pytest.mark.parametrize("loss_decay_gamma", [0.0, 3.0])
def test_loss_and_gradients_match_independent_equations(temperature, loss_decay_gamma):
    actual = make_model(temperature=temperature, loss_decay_gamma=loss_decay_gamma)
    reference = copy.deepcopy(actual)
    inputs, reference_inputs = make_inputs(), make_inputs()

    result, captured = run_and_capture(actual, inputs)
    expected, reference_captured = run_and_capture(reference, reference_inputs)
    expected_loss, expected_per_position = reference_kl(
        reference, reference_captured, reference_inputs
    )
    actual_loss, actual_per_position = reference_kl(actual, captured, inputs)

    torch.testing.assert_close(result[0], expected[0], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual_loss, result[0], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(expected_loss, expected[0], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual_per_position, expected_per_position,
                               rtol=2e-5, atol=2e-6)
    assert result[2][0] == 0 and result[4][0] == 0  # anchor slot is not supervised

    result[0].backward()
    expected_loss.backward()
    _assert_parameter_gradients(actual, reference)


def test_two_row_forward_kl_matches_reference_and_backward():
    actual = make_model(temperature=1.5)
    reference = copy.deepcopy(actual)
    inputs, reference_inputs = make_inputs(batch=2), make_inputs(batch=2)
    result, _ = run_and_capture(actual, inputs)
    expected, reference_captured = run_and_capture(reference, reference_inputs)
    expected_loss, _ = reference_kl(reference, reference_captured, reference_inputs)
    assert result[0].ndim == 0
    torch.testing.assert_close(result[0], expected[0], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(expected_loss, expected[0], rtol=2e-5, atol=2e-6)
    result[0].backward()
    expected_loss.backward()
    _assert_parameter_gradients(actual, reference)


def test_chunked_kl_matches_dense_objective_and_gradients():
    chunked = make_model(temperature=1.5, kl_chunk_rows=3)
    dense = make_model(temperature=1.5, kl_chunk_rows=256)
    inputs, dense_inputs = make_inputs(batch=2), make_inputs(batch=2)
    result, _ = run_and_capture(chunked, inputs)
    expected, dense_captured = run_and_capture(dense, dense_inputs)
    reference_loss, _ = reference_kl(dense, dense_captured, dense_inputs)
    torch.testing.assert_close(result[0], expected[0], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(result[0], reference_loss, rtol=2e-5, atol=2e-6)
    result[0].backward()
    expected[0].backward()
    _assert_parameter_gradients(chunked, dense)


def test_target_projection_uses_bounded_token_chunks(monkeypatch):
    model = make_model(kl_chunk_rows=3)
    inputs = make_inputs(batch=2)
    rows_per_projection = []
    original_linear = F.linear

    def recording_linear(input, weight, bias=None):
        if not torch.is_grad_enabled():
            rows_per_projection.append(input.shape[0])
        return original_linear(input, weight, bias)

    monkeypatch.setattr(F, "linear", recording_linear)
    result, _ = run_and_capture(model, inputs)
    result[0].backward()
    assert rows_per_projection
    assert max(rows_per_projection) <= 3


def test_chunked_backward_preserves_forward_autocast_state():
    outside = make_model(temperature=1.5)
    inside = make_model(temperature=1.5)
    inputs, inside_inputs = make_inputs(batch=2), make_inputs(batch=2)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result, _ = run_and_capture(outside, inputs)
    result[0].backward()

    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected, _ = run_and_capture(inside, inside_inputs)
        expected[0].backward()

    torch.testing.assert_close(result[0], expected[0], rtol=2e-5, atol=2e-6)
    _assert_parameter_gradients(outside, inside)


def test_chunked_backward_restores_autocast_dtype_for_bf16_teacher():
    model = make_model()
    inputs = make_inputs()
    inputs["hidden_states_list"] = inputs["hidden_states_list"].to(torch.bfloat16)
    inputs["last_hidden_states"] = inputs["last_hidden_states"].to(torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result, _ = run_and_capture(model, inputs)
    result[0].backward()
    assert torch.isfinite(result[0])
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_cpu_causal_path_rejects_packed_documents():
    model = make_model()
    inputs = make_inputs()
    inputs["injected_anchors"] = torch.tensor([[1, 2]])
    inputs["ctx_doc_ids"] = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]])
    with pytest.raises(ValueError, match="packed ctx_doc_ids"):
        model(**inputs)


def test_direct_draft_rejects_layer_without_effective_direct_cp():
    draft = JetSpecDraftModel(make_config(num_hidden_layers=2))
    with pytest.raises(ValueError, match="every Draft layer"):
        draft.forward(
            draft_input_ids=torch.zeros(1, 10, dtype=torch.long),
            context_feature=torch.zeros(1, 12, 16),
            draft_position_ids=torch.zeros(1, 10, dtype=torch.long),
            context_position_ids=torch.zeros(1, 12, dtype=torch.long),
            direct_cp_metadata=[None, None],
        )


def test_direct_draft_forward_requires_explicit_causal_visibility():
    draft = JetSpecDraftModel(make_config())
    with pytest.raises(ValueError, match="causal visibility"):
        draft.forward(
            draft_input_ids=torch.zeros(1, 10, dtype=torch.long),
            context_feature=torch.zeros(1, 12, 16),
            draft_position_ids=torch.zeros(1, 10, dtype=torch.long),
            context_position_ids=torch.zeros(1, 12, dtype=torch.long),
        )


def test_direct_draft_rejects_a_layer_without_an_effective_mask():
    draft = JetSpecDraftModel(make_config(num_hidden_layers=2))
    with pytest.raises(ValueError, match="every Draft layer"):
        draft.forward(
            draft_input_ids=torch.zeros(1, 10, dtype=torch.long),
            context_feature=torch.zeros(1, 12, 16),
            draft_position_ids=torch.zeros(1, 10, dtype=torch.long),
            context_position_ids=torch.zeros(1, 12, dtype=torch.long),
            draft_attention_mask=[torch.ones(1, 10, 22, dtype=torch.bool), None],
        )


def test_draft_forward_keeps_noise_embedding_positional_slot():
    import inspect

    from aspec.models.draft.dflash import DFlashDraftModel

    parameters = list(inspect.signature(DFlashDraftModel.forward).parameters)
    assert parameters.index("noise_embedding") < parameters.index("draft_attention_mask")


def test_cpu_forward_enforces_block_isolation():
    model = make_model()
    inputs = make_inputs()
    inputs["injected_anchors"] = torch.tensor([[2, 7]])
    _, base_hidden = run_and_capture(model, inputs)
    changed = {key: value.clone() if torch.is_tensor(value) else value
               for key, value in inputs.items()}
    # Block 0 (anchor 2) sees neither position 7 nor context positions 4..5.
    changed["input_ids"][0, 7] = (changed["input_ids"][0, 7] + 1) % 32
    changed["hidden_states_list"][0, 4:6] += 5.0
    _, moved_hidden = run_and_capture(model, changed)
    base_blocks = base_hidden.reshape(1, 2, 5, -1)
    moved_blocks = moved_hidden.reshape(1, 2, 5, -1)
    torch.testing.assert_close(base_blocks[:, 0], moved_blocks[:, 0], rtol=0, atol=0)
    assert not torch.equal(base_blocks[:, 1], moved_blocks[:, 1])


def test_non_causal_cpu_fallback_remains_bidirectional():
    from aspec.models.dflash import DFlashModel
    from aspec.models.draft.dflash import DFlashConfig, DFlashDraftModel

    config = DFlashConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, vocab_size=32,
        num_target_layers=1, target_hidden_size=16, target_num_hidden_layers=4,
        target_layer_ids=[1], mask_token_id=31, max_position_embeddings=128,
    )
    draft = DFlashDraftModel(config)
    draft.embed_tokens.requires_grad_(False)
    model = DFlashModel(draft, block_size=5, num_anchors=2, attention_backend="sdpa")
    inputs = make_inputs()
    inputs["injected_anchors"] = torch.tensor([[2, 7]])
    _, base_hidden = run_and_capture(model, inputs)
    changed = {key: value.clone() if torch.is_tensor(value) else value
               for key, value in inputs.items()}
    changed["hidden_states_list"][0, 10:] += 5.0
    _, moved_hidden = run_and_capture(model, changed)
    assert not torch.equal(base_hidden.reshape(1, 2, 5, -1)[:, 0],
                           moved_hidden.reshape(1, 2, 5, -1)[:, 0])


def test_loss_mask_excludes_unsupervised_labels():
    model = make_model()
    inputs = make_inputs()
    inputs["loss_mask"][0, 8] = 0
    result, captured = run_and_capture(model, inputs)
    expected, per_position = reference_kl(model, captured, inputs)
    torch.testing.assert_close(result[0], expected, rtol=2e-5, atol=2e-6)
    assert int(result[4][1:].sum()) == 7
    assert int(result[4][0]) == 0
    assert float(result[2][0].detach()) == 0.0
    assert per_position.shape == (1, 2, model.block_size)


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


def test_target_features_are_detached_and_required():
    model = make_model()
    inputs = make_inputs()
    inputs["last_hidden_states"].requires_grad_(True)
    result, _ = run_and_capture(model, inputs)
    result[0].backward()
    assert inputs["last_hidden_states"].grad is None
    assert model.draft_model.embed_tokens.weight.grad is None

    missing = make_inputs()
    missing["last_hidden_states"] = None
    with pytest.raises(ValueError, match="forward KL requires Target last_hidden_states"):
        model(**missing)


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf"), True, 1e-40])
def test_invalid_temperature_fails_closed(temperature):
    draft = JetSpecDraftModel(make_config())
    with pytest.raises(ValueError, match="temperature"):
        JetSpecModel(draft, block_size=5, num_anchors=2, attention_backend="sdpa",
                     ce_loss_alpha=0.0, l1_loss_alpha=0.0,
                     distillation_temperature=temperature)


@pytest.mark.parametrize("kl_chunk_rows", [0, -1, True, 1.5])
def test_invalid_kl_chunk_rows_fail_closed(kl_chunk_rows):
    draft = JetSpecDraftModel(make_config())
    with pytest.raises(ValueError, match="kl_chunk_rows"):
        JetSpecModel(draft, block_size=5, num_anchors=2, attention_backend="sdpa",
                     ce_loss_alpha=0.0, l1_loss_alpha=0.0,
                     kl_chunk_rows=kl_chunk_rows)


def test_non_kl_components_fail_closed():
    draft = JetSpecDraftModel(make_config())
    with pytest.raises(ValueError, match="CE/L1"):
        JetSpecModel(draft, block_size=5, num_anchors=2, attention_backend="sdpa")
    with pytest.raises(ValueError, match="CE/L1"):
        JetSpecModel(draft, block_size=5, num_anchors=2, attention_backend="sdpa",
                     ce_loss_alpha=0.0, l1_loss_alpha=1.0)
    with pytest.raises(ValueError, match="block_size"):
        JetSpecModel(draft, block_size=4, num_anchors=2, attention_backend="sdpa",
                     ce_loss_alpha=0.0, l1_loss_alpha=0.0)


def test_model_requires_a_causal_draft_config():
    from aspec.models.draft.dflash import DFlashConfig, DFlashDraftModel

    config = DFlashConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, vocab_size=32,
        num_target_layers=1, target_hidden_size=16, target_num_hidden_layers=4,
        target_layer_ids=[1], mask_token_id=31, max_position_embeddings=128,
        is_causal=False,
    )
    draft = DFlashDraftModel(config)
    with pytest.raises(ValueError, match="is_causal"):
        JetSpecModel(draft, block_size=5, num_anchors=2, attention_backend="sdpa",
                     ce_loss_alpha=0.0, l1_loss_alpha=0.0)


def test_model_rejects_a_non_jetspec_causal_config():
    from aspec.models.draft.dflash import DFlashConfig, DFlashDraftModel

    config = DFlashConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, vocab_size=32,
        num_target_layers=1, target_hidden_size=16, target_num_hidden_layers=4,
        target_layer_ids=[1], mask_token_id=31, max_position_embeddings=128,
        is_causal=True,
    )
    draft = DFlashDraftModel(config)
    with pytest.raises(ValueError, match="JetSpecDraftModel"):
        JetSpecModel(draft, block_size=5, num_anchors=2, attention_backend="sdpa",
                     ce_loss_alpha=0.0, l1_loss_alpha=0.0)


def test_short_sequences_keep_loss_finite():
    model = make_model(block_size=5)
    inputs = make_inputs(length=6)
    inputs["injected_anchors"] = torch.tensor([[2, 4]])
    result, captured = run_and_capture(model, inputs)
    expected, _ = reference_kl(model, captured, inputs)
    torch.testing.assert_close(result[0], expected, rtol=2e-5, atol=2e-6)
    assert torch.isfinite(result[0])
    result[0].backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_changing_a_later_block_does_not_change_an_earlier_block_loss():
    model = make_model()
    captured = []
    original_training_loss = model._training_loss

    def recording_training_loss(*args, **kwargs):
        result = original_training_loss(*args, **kwargs)
        captured.append(result[0].detach().clone())
        return result

    model._training_loss = recording_training_loss
    inputs = make_inputs()
    inputs["injected_anchors"] = torch.tensor([[2, 7]])
    _, base_hidden = run_and_capture(model, inputs)
    _, base_per_block = reference_kl(model, base_hidden, inputs)
    changed = {key: value.clone() if torch.is_tensor(value) else value
               for key, value in inputs.items()}
    # Change block 1's anchor token and later teacher states only.
    changed["input_ids"][0, 7] = (changed["input_ids"][0, 7] + 1) % 32
    changed["input_ids"][0, 8:] = (changed["input_ids"][0, 8:] + 1) % 32
    changed["last_hidden_states"][0, 8:] = torch.randn_like(
        changed["last_hidden_states"][0, 8:]
    )
    _, moved_hidden = run_and_capture(model, changed)
    _, moved_per_block = reference_kl(model, moved_hidden, changed)
    torch.testing.assert_close(base_per_block[0, 0], moved_per_block[0, 0],
                               rtol=0, atol=0)
    assert not torch.equal(base_per_block[0, 1], moved_per_block[0, 1])
    base_production = captured[0].reshape(1, 2, model.block_size)
    moved_production = captured[1].reshape(1, 2, model.block_size)
    torch.testing.assert_close(base_production[0, 0], moved_production[0, 0],
                               rtol=0, atol=0)
    assert not torch.equal(base_production[0, 1], moved_production[0, 1])
