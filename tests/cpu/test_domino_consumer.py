# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Domino native proposal mirrors the published sequential correction rollout."""

import torch

from aspec.draft_artifact import DominoConsumer
from aspec.models.draft import AutoDraftModelConfig, AutoEagle3DraftModel


def _raw(*, shift_label, prefix_len):
    return {
        "architectures": ["DominoDraftModel"],
        "model_type": "domino",
        "projector_type": "domino",
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "vocab_size": 32,
        "target_hidden_size": 16,
        "num_target_layers": 2,
        "target_num_hidden_layers": 6,
        "target_layer_ids": [1, 4],
        "max_position_embeddings": 128,
        "mask_token_id": 31,
        "block_size": 5,
        "emb_dim": 6,
        "gru_hidden_dim": 8,
        "shift_label": shift_label,
        "pure_draft_prefix_len": prefix_len,
    }


def _setup(*, shift_label, prefix_len):
    torch.manual_seed(11)
    model = AutoEagle3DraftModel.from_config(
        AutoDraftModelConfig.from_dict(_raw(shift_label=shift_label, prefix_len=prefix_len))
    ).eval()
    ids = torch.tensor([[2, 3, 5, 7]])
    features = torch.randn(1, 4, 32)
    head = torch.randn(32, 16)
    return model, ids, features, head


def _block_hidden(consumer, ids, features):
    anchor = torch.tensor([[ids.shape[1] - 1]])
    padded = torch.nn.functional.pad(ids, (0, consumer.block_size), value=consumer.config.mask_token_id)
    features = torch.nn.functional.pad(features, (0, 0, 0, consumer.block_size))
    hidden = consumer.backbone._draft_backbone(
        padded, features, torch.ones((1, padded.shape[1])),
        injected_anchors=anchor,
        injected_keep_mask=torch.ones_like(anchor, dtype=torch.bool),
    )[0][0]
    return hidden.unsqueeze(0) if consumer.shift_label else hidden[1:].unsqueeze(0)


def _reference(consumer, ids, features, head):
    hidden = _block_hidden(consumer, ids, features)
    base_logits = torch.nn.functional.linear(hidden, head)
    prefix = base_logits[:, : consumer.suffix_start].argmax(-1)
    realized = torch.cat((ids[:, -1:], prefix), dim=1)
    embeds = consumer.model.embed_tokens(realized)
    _, state = consumer.model.prefix_gru(embeds)
    tokens = [prefix]
    for position in range(consumer.suffix_start, consumer.k_draft):
        z = hidden[:, position:position + 1, :]
        bias = consumer.model.embed_proj(torch.cat((z, state.transpose(0, 1)), dim=-1))
        token = (base_logits[:, position:position + 1, :] + bias).argmax(-1)
        tokens.append(token)
        if position + 1 < consumer.k_draft:
            _, state = consumer.model.prefix_gru(consumer.model.embed_tokens(token), state)
    return torch.cat(tokens, dim=1)


def test_domino_consumer_matches_sequential_reference_for_both_layouts():
    for shift_label, prefix_len, expected_len in ((True, 1, 5), (False, 0, 4)):
        model, ids, features, head = _setup(shift_label=shift_label, prefix_len=prefix_len)
        consumer = DominoConsumer(model, block_size=model.config.block_size)
        actual = consumer.propose(ids, features, lm_head_weight=head)
        expected = _reference(consumer, ids, features, head)
        assert actual["algorithm"] == "domino"
        assert actual["token_ids"].shape == (1, expected_len)
        torch.testing.assert_close(actual["token_ids"], expected, atol=0, rtol=0)


def test_domino_prefix_is_backbone_only_and_suffix_uses_correction_head():
    model, ids, features, head = _setup(shift_label=True, prefix_len=1)
    consumer = DominoConsumer(model, block_size=model.config.block_size)
    baseline = consumer.propose(ids, features, lm_head_weight=head)["token_ids"]

    with torch.no_grad():
        model.embed_proj[2].weight.zero_()
        model.embed_proj[2].weight[5].fill_(100.0)
    corrected = consumer.propose(ids, features, lm_head_weight=head)["token_ids"]
    torch.testing.assert_close(corrected[:, :1], baseline[:, :1], atol=0, rtol=0)
    assert torch.equal(corrected[:, 1:], torch.full_like(corrected[:, 1:], 5))


def test_domino_consumer_rejects_other_block_algorithms():
    from aspec.models.draft.dflash import DFlashDraftModel

    raw = _raw(shift_label=True, prefix_len=1)
    raw.update(architectures=["DFlashDraftModel"], model_type="dflash")
    config = AutoDraftModelConfig.from_dict(raw)
    try:
        DominoConsumer(DFlashDraftModel(config), block_size=5)
    except ValueError as exc:
        assert "Domino" in str(exc)
    else:
        raise AssertionError("DominoConsumer accepted a non-Domino asset")
