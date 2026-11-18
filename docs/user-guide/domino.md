# Domino Draft training

This implements [ASpec #38](https://github.com/inclusionAI/ASpec/issues/38) through
`aspec.models.draft.domino.DominoDraftModel` and
`aspec.models.domino.DominoModel`: a block-parallel DFlash backbone with a
GRU-conditioned causal correction head. The ASpec training CLI now selects this
algorithm through `aspec/config/domino_qwen3_8b_draft_config.json`; the
[example workflow](../../configs/examples/domino_qwen3_8b_4k_tcp.yaml) runs one
Target GPU plus two Draft GPUs at 4K (three GPUs total).

## Training selection

- Draft config: `aspec/config/domino_qwen3_8b_draft_config.json` (Qwen3-8B,
  `block_size=16`, `shift_label=true`, `pure_draft_prefix_len=1`).
- Example: `configs/examples/domino_qwen3_8b_4k_tcp.yaml` (FlexAttention, CP1, MBS1,
  non-packing, REPLICATE, Store/TCP transport).
- The trainer owns a fixed per-run `training.domino_lambda_base`; the curriculum
  schedule is not implemented yet. Exact recovery records `lambda_base`, the label
  layout and the correction-head dimensions.
- The supported path fails closed outside dense full attention, padded inputs,
  `attention_backend=flex_attention`, `draft_context_parallel_size=1` and
  `dflash_packing=false`. The CUDA FlexAttention mask and the CPU dense fallback
  share the same block visibility; the CPU mask is built for contract tests only. MoE, sparse attention, MTP conditioning, packing, CP>1
  and whole-model compile are rejected during Plan compilation or configuration.

## Running

Compile the example before launching GPUs, then train with the same recipe:

```bash
aspec --config configs/examples/domino_qwen3_8b_4k_tcp.yaml --dry-run \
  model.target_model_path=/path/to/Qwen3-8B
RAY_ADDRESS=local aspec --config configs/examples/domino_qwen3_8b_4k_tcp.yaml \
  model.target_model_path=/path/to/Qwen3-8B
```

For exact recovery, point at a complete checkpoint with `training.load_path` and
write to a fresh `output_dir`; keep `training.continual_training=false` (the default):

```bash
aspec --config configs/examples/domino_qwen3_8b_4k_tcp.yaml \
  model.target_model_path=/path/to/Qwen3-8B \
  training.load_path=outputs/domino_qwen3_8b_4k_tcp/checkpoints \
  training.continual_training=false output_dir=outputs/domino_qwen3_8b_4k_tcp_resumed
```

The example is a short functional smoke-test recipe, not evidence that GPU execution
has been validated. Only CPU and Plan checks have been reported; the GPU run remains
pending. It uses the bundled conversations and a short step budget, not convergence,
throughput or decoding acceptance. Supply licensed data and a full learning-rate
horizon for a real run.

## Validation status

Plan compilation, registration, trainer construction, checkpoint topology and the
CPU numerical contracts are covered by `tests/cpu`. The public CI lane is CPU-only;
GPU end-to-end training and exact resume still need a hardware run before claiming
throughput or quality.

## Reference and selected scope

The implementation is **adapted** from the merged
[SpecForge Domino training path](https://github.com/sgl-project/SpecForge/pull/571)
(`specforge/core/domino.py` and the DFlash projector addition, merge
`7de39e32fc845caf59bca9841595145d24e1796f`, MIT; see
`licenses/SpecForge-MIT.txt`) and follows the public
[Domino method](https://arxiv.org/abs/2605.29707) at
[jianuo-huang/Domino @ `44e1e6912ba9d301899739a54bfda2ce8826980a`](https://github.com/jianuo-huang/Domino/tree/44e1e6912ba9d301899739a54bfda2ce8826980a)
for the architecture and label layout. It is not a reproduction of the paper's
training data, budget or acceptance results, and it does not copy the unlicensed
reference repository's source.

The selected first scope is one dense full-attention DFlash backbone such as Qwen3
with a frozen Target LM head and features. Dense FFNs and full intra-block attention
are enforced. Native MTP conditioning, sparse-attention training stages and MoE
FFNs fail closed.

## Correction head

The head owns two parameter groups, named as in the reference for future export:

- `prefix_gru`: a bias-free GRU over the block's real tokens, producing a causal
  predecessor state per suffix position;
- `embed_proj`: `Linear(hidden + gru_hidden) -> SiLU -> Linear(vocab)` producing the
  logit delta.

The block backbone itself stays block-parallel: attention visibility, anchor
sampling, noise construction and Target-feature fusion are unchanged from DFlash.
Only the correction head is causal, and its inputs are the **ground-truth block
tokens**, not Target features.

## Label-shift layouts

`DominoConfig` makes the alignment explicit instead of assuming one checkpoint
layout. `shift_label` and `pure_draft_prefix_len` select the reference conventions:

| Field | `shift_label=true` (official Qwen3) | `shift_label=false` |
| --- | --- | --- |
| Supervised label for block position `k` | `x[anchor + 1 + k]` | `x[anchor + k]` |
| First supervised block position | `0` | `1` (anchor is context only) |
| `suffix_start` | `pure_draft_prefix_len` | `1 + pure_draft_prefix_len` |
| GRU input tokens | `x[anchor .. anchor + block_size - 1]`, states used from `suffix_start` | `x[anchor .. anchor + block_size - 2]`, states used from `suffix_start - 1` |

`pure_draft_prefix_len` must leave at least one corrected position in the block.
The official `Qwen3-8B-Domino-b16` configuration uses `shift_label=true` and
`pure_draft_prefix_len=1`, so block position 0 keeps the backbone logits and
positions 1..`block_size-1` receive the GRU delta. The unshifted layout matches
plain DFlash supervision: position 0 is the anchor and excluded from the loss.

## Objective

`DominoModel` mixes the corrected and backbone-only cross-entropies:

```text
loss = (1 - lambda_base) * CE(corrected logits, labels)
     + lambda_base     * CE(backbone logits, labels)
```

`lambda_base` is an explicit per-step value in `[0, 1]`; use
`model.set_lambda_base(value)` from a trainer-owned curriculum. `lambda_base=1`
removes the correction contribution and keeps the selected label layout (the
correction head receives no useful gradient). It equals the plain DFlash objective
only for the unshifted layout (`shift_label=false`) with otherwise matching
weighting settings; the shifted layout supervises different slots. Anchor sampling,
block masks, label bounds, `loss_mask` and DFlash-style exponential depth decay are
reused; `ce_loss_alpha=1` and `l1_loss_alpha=0` are enforced. Reporting accuracy and per-position loss use the
corrected logits. The trainer reports `mixed_ce` as `train/avg_loss`/`eval/avg_loss`
(the objective actually used for backward) and exposes `mixed_ce`, `backbone_ce` and
`corrected_ce` as separate token-pooled component curves.

## Minimal numerical use

Inside the [isolated Linux environment](installation.md):

```python
import torch
from aspec.models.domino import DominoModel
from aspec.models.draft.domino import DominoConfig, DominoDraftModel

config = DominoConfig(
    hidden_size=16, intermediate_size=32, num_hidden_layers=1,
    num_attention_heads=2, num_key_value_heads=1, head_dim=8, vocab_size=32,
    num_target_layers=1, target_hidden_size=16, target_num_hidden_layers=4,
    target_layer_ids=[1], mask_token_id=31, max_position_embeddings=128,
    block_size=5, emb_dim=6, gru_hidden_dim=8,
    shift_label=True, pure_draft_prefix_len=1,
)
draft = DominoDraftModel(config)
draft.embed_tokens.requires_grad_(False)
model = DominoModel(
    draft, block_size=5, num_anchors=2, attention_backend="sdpa",
    lambda_base=0.3,
)
result = model(
    input_ids=torch.arange(12).reshape(1, 12) % 32,
    hidden_states_list=torch.randn(1, 12, 16),
    loss_mask=torch.ones(1, 12),
    lm_head_weight=torch.randn(32, 16),
    last_hidden_states=torch.randn(1, 12, 16),
    injected_anchors=torch.tensor([[2, 7]]),
    injected_keep_mask=torch.ones(1, 2, dtype=torch.bool),
)
result[0].backward()
```

Run `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q tests/cpu/test_domino.py`
for configuration fail-closed checks, independent forward/gradient equations for both
label layouts, `lambda_base` identity, anchor exclusion, loss-mask and short-sequence
handling, empty supervision, and frozen Target/embedding behavior.

## Remaining work

The `lambda_base` curriculum schedule, a maintained recipe catalog entry, native
export with strict reloading, Target bindings, proposal consumers and held-out
backbone-vs-corrected observations remain pending. CPU numerical and Plan checks do
not establish Qwen3-8B training quality, GPU throughput or decoding acceptance.
