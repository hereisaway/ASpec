# Copyright 2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Adapted from the merged SpecForge Domino training path
# (sgl-project/SpecForge PR #571, specforge/core/domino.py and the
# modelling/draft/dflash.py projector addition, MIT; see
# licenses/SpecForge-MIT.txt). The GRU-conditioned logit correction, predecessor
# alignment and label-shift conventions follow that implementation and the
# public Domino method (arXiv:2605.29707). ASpec supplies its own configuration,
# fail-closed draft contract and DFlash training integration.
"""Domino Draft: block-parallel DFlash backbone with a causal correction head."""

import torch
from torch import nn

from aspec.models.draft.dflash import DFlashConfig, DFlashDraftModel


class DominoConfig(DFlashConfig):
    """DFlash block backbone plus a GRU-conditioned correction head."""

    model_type = "domino"

    def __init__(
        self,
        *,
        projector_type="domino",
        emb_dim=256,
        gru_hidden_dim=1024,
        shift_label=True,
        pure_draft_prefix_len=1,
        block_size=16,
        **kwargs,
    ):
        if projector_type != "domino":
            raise ValueError("Domino requires projector_type='domino'")
        for name, value in (("emb_dim", emb_dim), ("gru_hidden_dim", gru_hidden_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Domino requires a positive integer {name}")
        if not isinstance(shift_label, bool):
            raise ValueError("Domino shift_label must be a boolean")
        if isinstance(pure_draft_prefix_len, bool) or not isinstance(
            pure_draft_prefix_len, int
        ):
            raise ValueError("Domino pure_draft_prefix_len must be an integer")
        if pure_draft_prefix_len < 0:
            raise ValueError("Domino pure_draft_prefix_len must be non-negative")
        if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 2:
            raise ValueError("Domino block_size must be an integer >= 2")
        suffix_start = pure_draft_prefix_len if shift_label else 1 + pure_draft_prefix_len
        if not 0 <= suffix_start < block_size:
            raise ValueError(
                "Domino pure_draft_prefix_len leaves no corrected position in the block"
            )
        architectures = kwargs.pop("architectures", ["DominoDraftModel"])
        if architectures != ["DominoDraftModel"]:
            raise ValueError("Domino uses exactly one DominoDraftModel architecture")
        if kwargs.get("is_causal"):
            raise ValueError(
                "Domino keeps the block-parallel DFlash backbone; causal modelling "
                "belongs to the correction head"
            )
        kwargs.update(
            projector_type="domino",
            emb_dim=emb_dim,
            gru_hidden_dim=gru_hidden_dim,
            shift_label=shift_label,
            pure_draft_prefix_len=pure_draft_prefix_len,
        )
        super().__init__(
            architectures=architectures, block_size=block_size, **kwargs
        )
        if self.ffn_type != "dense":
            raise ValueError("Domino currently supports dense Draft FFNs only")
        layer_types = list(getattr(self, "layer_types", []) or [])
        if any(kind != "full_attention" for kind in layer_types):
            raise ValueError("Domino currently supports full intra-block attention only")
        if getattr(self, "mtp_conditioning", None) is not None:
            raise ValueError("Domino does not use the native-MTP conditioning contract")
        for stage in ("csa_training_stage", "qsa_training_stage", "dsa_training_stage"):
            if getattr(self, stage, None) is not None:
                raise ValueError("Domino is not integrated with sparse-attention training")


class DominoDraftModel(DFlashDraftModel):
    """DFlash block backbone with the Domino GRU correction head."""

    config_class = DominoConfig

    def __init__(self, config):
        super().__init__(config)
        self.projector_type = config.projector_type
        self.shift_label = config.shift_label
        self.pure_draft_prefix_len = config.pure_draft_prefix_len
        self.suffix_start = (
            config.pure_draft_prefix_len
            if config.shift_label
            else 1 + config.pure_draft_prefix_len
        )
        self.emb_dim = config.emb_dim
        self.gru_hidden_dim = config.gru_hidden_dim
        # Reference names are retained so future export can map checkpoint keys.
        self.prefix_gru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=config.gru_hidden_dim,
            num_layers=1,
            batch_first=True,
            bias=False,
        )
        self.embed_proj = nn.Sequential(
            nn.Linear(
                config.hidden_size + config.gru_hidden_dim,
                config.emb_dim,
                bias=False,
            ),
            nn.SiLU(),
            nn.Linear(config.emb_dim, config.vocab_size, bias=False),
        )

    def correct_logits(
        self,
        draft_hidden: torch.Tensor,
        base_logits: torch.Tensor,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Add the causal correction delta to the block suffix.

        ``draft_hidden`` and ``base_logits`` are ``[B, N * block_size, ...]``.
        Position 0 (or ``suffix_start`` for a pure draft prefix) keeps the
        backbone logits; every later position receives the GRU-conditioned
        delta. ``shift_label`` selects the reference alignment: the shifted
        layout consumes ``anchor ... anchor + block_size - 1`` tokens, while the
        unshifted layout consumes the first ``block_size - 1`` of them.
        """
        batch, draft_len = base_logits.shape[:2]
        block_size = self.config.block_size
        if draft_len != anchor_positions.shape[1] * block_size:
            raise ValueError("Domino logits do not match the anchor/block layout")
        hidden = draft_hidden.reshape(batch, -1, block_size, self.hidden_size)
        logits = base_logits.reshape(batch, -1, block_size, base_logits.size(-1))

        offsets = torch.arange(block_size, device=input_ids.device).view(1, 1, -1)
        token_indices = (anchor_positions.unsqueeze(-1) + offsets).clamp(
            max=input_ids.size(1) - 1
        )
        block_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.shape[1], -1),
            2,
            token_indices,
        )

        if self.shift_label:
            gru_inputs = block_ids
            state_slice = slice(self.suffix_start, None)
        else:
            gru_inputs = block_ids[..., : block_size - 1]
            state_slice = slice(self.suffix_start - 1, None)

        block_emb = self.embed_tokens(gru_inputs).to(self.prefix_gru.weight_ih_l0.dtype)
        states, _ = self.prefix_gru(
            block_emb.reshape(batch * anchor_positions.shape[1], -1, self.hidden_size)
        )
        states = states.reshape(
            batch, anchor_positions.shape[1], gru_inputs.size(-1), self.gru_hidden_dim
        )[:, :, state_slice, :]
        suffix_hidden = hidden[:, :, self.suffix_start :, :]
        if states.size(2) != suffix_hidden.size(2):
            raise ValueError("Domino correction state and hidden suffix lengths differ")

        delta = self.embed_proj(torch.cat([suffix_hidden, states], dim=-1))
        corrected = torch.cat(
            [
                logits[:, :, : self.suffix_start, :],
                logits[:, :, self.suffix_start :, :] + delta.to(logits.dtype),
            ],
            dim=2,
        )
        return corrected.reshape(batch, draft_len, logits.size(-1))
