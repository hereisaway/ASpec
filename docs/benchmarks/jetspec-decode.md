# JetSpec bounded decode compatibility

This is a bounded decode integration check, not a throughput benchmark, convergence
result or serving-quality claim. It verifies that an ASpec-exported JetSpec asset is
loaded by official SGLang 0.5.21, that the causal head is active, and that exporting
the published checkpoint through ASpec preserves the decode result.

## Setup

- Host: one H100 PCIe 80GB, `sglang==0.5.21`, torch 2.13.0+cu130.
- Target: `Qwen/Qwen3-8B` (`/public/models/Qwen3-8B`).
- Data: 16 prompts from the held-out validation split.
- Decode: greedy, 32 new tokens, two complete repeats, normal radix cache.
- Draft: JetSpec block size 16, five Target feature layers `[1, 9, 17, 25, 33]`.
- Export: `aspec export` default SGLang layout with top-level `is_causal=true` and
  `dflash_config.causal_head=true`.

SGLang 0.5.21 reads the top-level `is_causal` flag for DFlash full-attention layers;
`dflash_config.causal_head` alone leaves those layers bidirectional.

## Results

| Draft | Acceptance length | Verifications | Median speedup | Greedy equal | Repeats stable |
| --- | ---: | ---: | ---: | --- | --- |
| ASpec 8-step checkpoint | 1.032 | 992 | 0.795x | yes | yes |
| Published `JetSpec/jetspec-qwen3-8b` loaded directly | 2.723 | 376 | 1.823x | yes | yes |
| Published checkpoint exported through ASpec | 2.723 | 376 | 1.823x | yes | yes |

Acceptance length follows `aspec evaluate`: completion tokens divided by real
verifier calls, including Target tokens. Both runs produced 512 completion tokens
across two repeats.

The ASpec-exported published checkpoint matched the directly loaded checkpoint's
acceptance length, verification count and speedup in this bounded setup. That checks
config/weight/layout compatibility; it does not establish convergence or a trained
quality comparison.

The 8-step ASpec checkpoint was trained only as a functional smoke test on synthetic
data. Its lower acceptance and slower speculative decode are expected and are not
evidence against the method.

## Reproduction

```bash
aspec export --input-dir outputs/jetspec/checkpoints/iter_0000009 \
  --output-dir outputs/jetspec-sglang

CUDA_VISIBLE_DEVICES=0 aspec evaluate \
  --target /models/Qwen3-8B --draft outputs/jetspec-sglang \
  --data held-out.jsonl --output outputs/jetspec-decode.json \
  --samples 16 --max-new-tokens 32 --repeats 2
```

The published checkpoint can be exported through ASpec with the same command after
converting its config with `AutoDraftModelConfig.from_dict`.
