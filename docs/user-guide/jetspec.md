# JetSpec model and loss core

This is the first prerequisite for [ASpec #15](https://github.com/inclusionAI/ASpec/issues/15).
It provides `aspec.models.jetspec.JetSpecModel`,
`aspec.models.draft.jetspec.JetSpecConfig` and
`aspec.models.draft.jetspec.JetSpecDraftModel`. It is a model/loss API; the ASpec
training CLI does not select this algorithm yet.

## Reference and selected method

The implementation targets **JetSpec**, not JetSpec++. The selected reference is
[JetSpec: Breaking the Scaling Ceiling of Speculative Decoding with Parallel Tree
Drafting](https://arxiv.org/abs/2606.18394) (Hu et al., 2026), Section 3.3 and
Appendix D. The upstream [implementation](https://github.com/hao-ai-lab/JetSpec) is
MIT-licensed; this is an independent ASpec implementation of the published
objective, and no JetSpec source file is copied.

JetSpec's Draft head injects fused features from selected frozen Target layers as
contextual K/V state and predicts a whole block in one pass. ASpec already retains
that head as the DFlash block head, including dual-source KV and anchor sampling.
The JetSpec core therefore adapts the retained DFlash path instead of duplicating
it, and adds the two pieces that distinguish JetSpec:

- **causal intra-block attention** (`is_causal=true`), so each Draft position sees
  the verified prefix, its anchor and earlier positions in the same block, but no
  future Draft position and no other block; and
- a **temperature-scaled forward-KL distillation objective** against the frozen
  Target, rather than CE or L1.

The DFlash2 selector, MoE Draft FFNs, sliding-window Draft layers, native-MTP
conditioning and packed/CP execution are outside this core.

FlexAttention is used on CUDA. On the CPU SDPA fallback, the same block-causal
visibility is materialized explicitly so the isolation contract is exercised by
the default tests. Draft configurations that are not causal keep the previous
unmasked fallback behavior.

## Prediction and supervision contract

A block is anchored at an original sequence position `a`. Position `0` keeps the
real token `x[a]` as block context and is excluded from the loss. Position `k >= 1`
predicts `x[a+k]` and distills the Target distribution obtained from Target hidden
state `h[a+k-1]` and the frozen Target LM head. This is the next-token alignment
used by the retained DFlash path.

The loss is

```text
p = softmax(target_logits / T)
q = softmax(draft_logits   / T)
L = T^2 * sum(w * KL(p || q)) / sum(w)
```

where `w` includes block validity, sequence bounds, `loss_mask`, the anchor
exclusion, and optional depth weighting `exp(-(k-1)/gamma)`. The default
`distillation_temperature=1.0` and `loss_decay_gamma=0.0` (uniform weights) match
the main paper configuration; both are configurable. Target logits are produced
under `torch.no_grad()`. The Target hidden states must come from
`inference.store_last_hidden_states=true`, as for the existing L1 distillation path.

`JetSpecModel` fails closed when the Draft config is not a causal
`JetSpecDraftModel`, when CE/L1 alphas are nonzero, when the training block size
disagrees with the Draft config, or when the temperature is not finite and at
least `1e-6`. `JetSpecConfig` fixes the architecture
to `JetSpecDraftModel`, forces `is_causal=true`, and rejects non-dense FFNs,
sliding-window layers and MTP conditioning.

The full-vocabulary Target projection and FP32 KL intermediates are processed in
bounded token chunks (`kl_chunk_rows`, default 256). Only rows with positive
objective weight are projected, and a custom backward recomputes each chunk
instead of retaining every chunk's log-probability tensors until backward. An
entire masked window keeps an exact differentiable zero without projecting a
teacher row.

The training wrapper always supplies explicit visibility. A direct
`JetSpecDraftModel.forward` call must pass `block_mask`, `draft_attention_mask` or
`direct_cp_metadata`; otherwise it fails closed instead of silently running
bidirectional SDPA. The CPU dense-mask path also rejects packed `ctx_doc_ids`,
because the dense helper cannot preserve document isolation.

## Minimal numerical use

Inside the [isolated environment](installation.md):

```python
import torch
from aspec.models.draft.jetspec import JetSpecConfig, JetSpecDraftModel
from aspec.models.jetspec import JetSpecModel

config = JetSpecConfig(
    hidden_size=16, intermediate_size=32, num_hidden_layers=1,
    num_attention_heads=2, num_key_value_heads=1, head_dim=8, vocab_size=32,
    num_target_layers=1, target_hidden_size=16, target_num_hidden_layers=4,
    target_layer_ids=[1], mask_token_id=31, max_position_embeddings=128,
    block_size=5,
)
draft = JetSpecDraftModel(config)
draft.embed_tokens.requires_grad_(False)
model = JetSpecModel(
    draft, block_size=5, num_anchors=2, attention_backend="sdpa",
    ce_loss_alpha=0.0, l1_loss_alpha=0.0, distillation_temperature=1.0,
)

ids = torch.arange(12)[None]
outputs = model(
    input_ids=ids,
    hidden_states_list=torch.randn(1, 12, 16),   # selected Target layers
    loss_mask=torch.ones_like(ids),
    lm_head_weight=torch.randn(32, 16),          # frozen Target head
    last_hidden_states=torch.randn(1, 12, 16),   # frozen Target final state
    injected_anchors=torch.tensor([[3, 5]]),
    injected_keep_mask=torch.ones(1, 2, dtype=torch.bool),
)
outputs[0].backward()
```

The return value follows `DFlashModel.forward`: scalar loss, accuracy,
per-position loss/accuracy/count, extra components and the objective numerator /
denominator. Per-position loss now reports KL rather than CE.

## Boundaries and validation

- This core does not provide a recipe, trainer selection, exact resume, tree
  construction, export, strict reloading or a serving consumer. Those remain for
  follow-up work under the issue's expected outcomes.
- CPU checks cover the configuration contract, causal mask construction, bounded
  forward KL and gradients against independent equations, explicit-mask and packed
  fail-closed paths, anchor/loss-mask exclusion, empty supervision and next-token
  label/teacher alignment on unpacked inputs. They do not certify GPU attention
  execution, convergence, acceptance length or serving performance.
- The default tests are in `tests/cpu/test_jetspec.py`.
