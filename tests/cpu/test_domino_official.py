# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Map the published Qwen3 Domino config/weight layout into ASpec."""


import torch
from safetensors.torch import save_file

from aspec.models.draft import AutoDraftModelConfig, AutoEagle3DraftModel
from aspec.models.draft.domino import (
    normalize_official_domino_config,
    official_domino_config,
)
from aspec.models.draft.keymap import to_export_keys
from aspec.training.checkpoint import load_initial_draft_weights

_OFFICIAL = {
    "architectures": ["DFlashDraftModel"],
    "auto_map": {"AutoModel": "dflash.DFlashDraftModel"},
    "model_type": "qwen3",
    "dtype": "bfloat16",
    "vocab_size": 151936,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "block_size": 16,
    "num_target_layers": 36,
    "num_hidden_layers": 5,
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "emb_dim": 256,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-06,
    "max_position_embeddings": 40960,
    "rope_theta": 1000000,
    "tie_word_embeddings": False,
    "layer_types": ["full_attention"] * 5,
    "max_window_layers": 5,
    "dflash_config": {
        "projector_type": "domino",
        "mask_token_id": 151669,
        "shift_label": True,
        "target_layer_ids": [1, 9, 17, 25, 33],
        "pure_draft_prefix_len": 1,
        "gru_hidden_dim": 1024,
        "emb_dim": 256,
    },
}


def _raw(**overrides):
    raw = {
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
        "shift_label": True,
        "pure_draft_prefix_len": 1,
    }
    raw.update(overrides)
    return raw


def test_official_qwen3_domino_config_maps_to_flat_aspec_layout():
    config = AutoDraftModelConfig.from_dict(_OFFICIAL)
    assert config.architectures == ["DominoDraftModel"]
    assert config.model_type == "domino"
    assert config.projector_type == "domino"
    assert config.block_size == 16
    assert config.num_target_layers == 5
    assert config.target_num_hidden_layers == 36
    assert config.target_layer_ids == [1, 9, 17, 25, 33]
    assert config.shift_label is True
    assert config.pure_draft_prefix_len == 1
    assert config.emb_dim == 256
    assert config.gru_hidden_dim == 1024


def test_flat_config_roundtrips_through_published_layout():
    flat = _raw()
    published = official_domino_config(flat)
    assert published["architectures"] == ["DFlashDraftModel"]
    assert published["model_type"] == "qwen3"
    assert published["num_target_layers"] == flat["target_num_hidden_layers"]
    assert published["dflash_config"]["target_layer_ids"] == flat["target_layer_ids"]
    restored = normalize_official_domino_config(published)
    assert restored["architectures"] == ["DominoDraftModel"]
    assert restored["model_type"] == "domino"
    assert restored["num_target_layers"] == flat["num_target_layers"]
    assert restored["target_num_hidden_layers"] == flat["target_num_hidden_layers"]
    for key in ("projector_type", "block_size", "emb_dim", "gru_hidden_dim",
                "shift_label", "pure_draft_prefix_len", "target_layer_ids",
                "mask_token_id"):
        assert restored[key] == flat[key]


def test_official_weight_names_load_with_borrowed_embedding(tmp_path):
    raw = _raw()
    source = AutoEagle3DraftModel.from_config(AutoDraftModelConfig.from_dict(raw))
    official = to_export_keys(source.state_dict())
    official.pop("embed_tokens.weight")
    path = tmp_path / "model.safetensors"
    save_file(official, str(path))

    target = AutoEagle3DraftModel.from_config(AutoDraftModelConfig.from_dict(raw))
    load_initial_draft_weights(target, str(path), allow_target_embedding=True)
    loaded = target.state_dict()
    for key, value in source.state_dict().items():
        if key == "embed_tokens.weight":
            continue
        torch.testing.assert_close(loaded[key], value, atol=0, rtol=0)


def test_unknown_nested_projector_is_not_rewritten():
    published = {key: value for key, value in _OFFICIAL.items() if key != "max_window_layers"}
    raw = {**published, "dflash_config": {**_OFFICIAL["dflash_config"],
                                          "projector_type": "dflash"}}
    config = AutoDraftModelConfig.from_dict(raw)
    assert config.architectures == ["DFlashDraftModel"]
    assert config.model_type == "qwen3"


def test_published_layout_includes_default_projector():
    flat = _raw()
    flat.pop("projector_type")
    published = official_domino_config(flat)
    assert published["dflash_config"]["projector_type"] == "domino"
    restored = normalize_official_domino_config(published)
    assert restored["architectures"] == ["DominoDraftModel"]
    assert restored["projector_type"] == "domino"
