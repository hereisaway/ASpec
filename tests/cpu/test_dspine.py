# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""DSpine model/injection/loss contracts on CPU."""

import pytest
import torch

from aspec.models.draft.dspine import (
    AdjacentInjection,
    DSpineConfig,
    DSpineDraftModel,
    DSpineOptions,
)
from aspec.models.dspine import DSpineModel, refinement_loss


def make_config(**overrides):
    raw = dict(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=32,
        num_target_layers=1,
        target_hidden_size=16,
        target_num_hidden_layers=4,
        target_layer_ids=[1],
        mask_token_id=31,
        max_position_embeddings=128,
        block_size=4,
        dspine_config={
            "transfer_rank": 8,
            "message_size": 8,
            "top_k": 4,
            "alignment_warmup_steps": 0,
            "replacement_layers": 1,
            "replacement_probability": 0.0,
        },
    )
    raw.update(overrides)
    return DSpineConfig(**raw)


def make_model(config=None):
    config = config or make_config()
    draft = DSpineDraftModel(config)
    draft.freeze_embedding()
    return DSpineModel(
        draft, block_size=config.block_size, num_anchors=2,
        attention_backend="sdpa",
    )


def make_inputs(seed=0, length=12, batch=1):
    generator = torch.Generator().manual_seed(seed)
    return {
        "input_ids": torch.randint(0, 31, (batch, length), generator=generator),
        "hidden_states_list": torch.randn(batch, length, 16, generator=generator),
        "loss_mask": torch.ones(batch, length),
        "lm_head_weight": torch.randn(32, 16, generator=generator),
        "last_hidden_states": torch.randn(batch, length, 16, generator=generator),
    }


def test_config_fails_closed_on_non_causal_and_invalid_rank():
    with pytest.raises(ValueError, match="causal"):
        make_config(is_causal=False)
    with pytest.raises(ValueError, match="transfer_rank"):
        make_config(dspine_config={"transfer_rank": 32})
    with pytest.raises(ValueError, match="block_size"):
        make_config(block_size=1)


def test_forward_initializes_transfer_space_and_backpropagates():
    torch.manual_seed(3)
    model = make_model()
    inputs = make_inputs()
    loss, accuracy, loss_pp, acc_pp, count_pp, components, terms = model(
        **inputs, global_step=0, total_steps=1,
    )

    assert torch.isfinite(loss)
    assert loss_pp.shape == acc_pp.shape == count_pp.shape == (4,)
    assert loss_pp[0] == 0 and acc_pp[0] == 0
    assert torch.equal(count_pp[0], torch.tensor(0.0))
    assert bool(model.draft_model.transfer_ready)
    assert model.draft_model.transfer_codes.shape == (32, 8)
    assert "dspine/backbone_ce" in components
    assert terms[1] > 0

    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)


def test_transfer_space_codes_are_normalized_and_readouts_use_pca():
    model = make_model()
    head = torch.randn(32, 16)
    model.draft_model.initialize_transfer_space(head)
    norms = torch.linalg.vector_norm(model.draft_model.transfer_codes.float(), dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4)
    for readout in model.draft_model.injection.readouts:
        assert readout.weight.shape == (8, 16)
        assert torch.isfinite(readout.weight).all()


def test_adjacent_injection_does_not_leak_future_tokens_backwards():
    options = DSpineOptions(
        transfer_rank=8, message_size=8, top_k=4,
        alignment_warmup_steps=0, replacement_layers=1,
        replacement_probability=0.0,
    )
    injection = AdjacentInjection(16, 1, options)
    generator = torch.Generator().manual_seed(11)
    hidden = torch.randn(1, 4, 16, generator=generator)
    token_ids = torch.randint(0, 32, (1, 4), generator=generator)
    codes = torch.randn(32, 8, generator=generator)

    first = injection(hidden, 0, token_ids, codes)
    changed = hidden.clone()
    changed[..., -1, :] += 10.0
    second = injection(changed, 0, token_ids, codes)

    torch.testing.assert_close(first[..., :-1, :], second[..., :-1, :], atol=0, rtol=0)
    assert not torch.equal(first[..., -1, :], second[..., -1, :])


def _reference_injection(injection, hidden, layer, token_ids, codes, replace):
    import math

    import torch.nn.functional as F

    features = injection.feature(hidden, layer)
    anchor = math.sqrt(injection.rank) * F.embedding(token_ids[..., :1], codes)
    predecessors = torch.cat((anchor, features[..., 1:-1, :]), dim=-2)
    if replace is not None and layer < injection.replacement_layers:
        correct = math.sqrt(injection.rank) * F.embedding(token_ids[..., :-1], codes)
        predecessors = torch.where(replace[..., None, None], correct, predecessors)
    gate = torch.sigmoid(
        injection.receiver_gate(features[..., 1:, :])
        + injection.predecessor_gate(predecessors)
        + injection.bias
    )
    delta = injection.writes[layer](gate * injection.message(predecessors))
    scale = (
        torch.linalg.vector_norm(hidden[..., 1:, :].float(), dim=-1, keepdim=True)
        / math.sqrt(hidden.shape[-1])
    ).to(hidden.dtype)
    return torch.cat((hidden[..., :1, :], hidden[..., 1:, :] + scale * delta), dim=-2)


def test_adjacent_injection_matches_independent_equation():
    options = DSpineOptions(
        transfer_rank=8, message_size=8, top_k=4,
        alignment_warmup_steps=0, replacement_layers=1,
        replacement_probability=0.0,
    )
    injection = AdjacentInjection(16, 2, options)
    generator = torch.Generator().manual_seed(19)
    hidden = torch.randn(1, 2, 4, 16, generator=generator)
    token_ids = torch.randint(0, 32, (1, 2, 4), generator=generator)
    codes = torch.randn(32, 8, generator=generator)
    replace = torch.tensor([[True, False]])

    actual = injection(hidden, 0, token_ids, codes, replace)
    expected = _reference_injection(injection, hidden, 0, token_ids, codes, replace)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_refine_matches_independent_equation():
    model = make_model()
    model.draft_model.initialize_transfer_space(torch.randn(32, 16))
    generator = torch.Generator().manual_seed(23)
    hidden = torch.randn(1, 2, 3, 16, generator=generator)
    predecessors = torch.randint(0, 32, (1, 2, 3), generator=generator)
    layer = len(model.draft_model.layers) - 1
    actual = model.draft_model.refine(hidden, predecessors)
    import math

    import torch.nn.functional as F

    receiver = model.draft_model.injection.feature(hidden, layer)
    codes = math.sqrt(model.draft_model.injection.rank) * F.embedding(
        predecessors, model.draft_model.transfer_codes
    )
    expected = model.draft_model.final_norm(
        model.draft_model.injection.write(hidden, receiver, codes, layer)
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_refinement_loss_is_zero_for_equal_candidate_distributions():
    scores = torch.tensor([[[2.0, 1.0, 0.0], [2.0, 1.0, 0.0]]])
    target = torch.softmax(scores, dim=-1)
    candidates = torch.tensor([[[4, 5, 6], [4, 5, 6]]])
    labels = torch.tensor([[4, 4]])
    weights = torch.ones_like(labels, dtype=torch.float32)
    refinement, ref_ce, ref_l1 = refinement_loss(
        scores=scores, candidates=candidates, target_candidates=target,
        proposal_labels=labels, proposal_weights=weights,
        ce_weight=0.1, mass_threshold=1e-4,
    )
    assert ref_l1 == 0
    torch.testing.assert_close(refinement, 0.1 * ref_ce)


def test_packed_inputs_are_rejected_before_backbone():
    model = make_model()
    inputs = make_inputs()
    with pytest.raises(NotImplementedError, match="padded, non-packed"):
        model(**inputs, ctx_doc_ids=torch.zeros_like(inputs["input_ids"]))
