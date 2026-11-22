# Python reference harness

FP32 teacher-forced reference for the Inkling 2 text decoder and the Inkling 2 Mini loop decoder. The Rust model is not in this tree yet. Vision and multi-token prediction are out of scope.

Parity target: Hugging Face transformers **5.24.0**, commit `eeee11099135592f1d03c2871fbd3d8132e5a4d7`, eager attention, CPU, one thread. Semantics, including the sliding-window inequality and the loop cache, are in [SEMANTICS.md](SEMANTICS.md).

## Setup

From the repository root:

```bash
python3 -m venv tools/ref/.venv
source tools/ref/.venv/bin/activate
pip install -r tools/ref/requirements.txt
```

`requirements.txt` pins `torch==2.15.0+cpu` (the CPU index) and `transformers==5.24.0`. `generate.py` hashes the installed `modeling_inkling2.py` and `modeling_inkling2_loop.py` against [pin.json](pin.json) and stops if they are not the files from that commit.

Do not install `causal-conv1d` or the Hugging Face `kernels` package in this environment. Those replace the PyTorch convolutions the dump is measured against.

## CI

`.github/workflows/ci.yml` runs `cargo check` and the two micro configs only. The tiny checkpoints are hundreds of megabytes and also download a tokenizer; regenerate those locally. Micro still covers both architectures, the sliding-window boundary, a loop pass, clipping, and the 1e-6 cached-decode check.

## Generate

```bash
python tools/ref/generate.py
```

One config:

```bash
python tools/ref/generate.py --only inkling2/micro
```

Names: `inkling2/micro`, `inkling2/tiny`, `inkling2_loop/micro`, `inkling2_loop/tiny`.

Each run writes `artifacts/ref/<family>/<size>/`:

| file | contents |
| --- | --- |
| `config.json` | text causal LM config. `model_type` is `inkling2_text` or `inkling2_loop_text`. |
| `model.safetensors` | `save_pretrained` output: Thinking Machines names (`model.llm.*`), interleaved `w13_weight`, zero-padded unembed. Do not `chunk(2)` this file. |
| `model.transformers.safetensors` | `state_dict()`. Experts are concatenated `[gate \| up]`, which is what `chunk(2)` expects. |
| `activations.safetensors` | one full-sequence forward. Keys `L{i}.{site}` or, on looped layers, `L{i}.p{pass}.{site}`. Shape `[T, ...]`, fp32. `expert_ids` are int64, sorted ascending, with `expert_weights` permuted to match. |
| `manifest.json` | site descriptions, layer table, per-kind clip counts, softcap deviations, router margin, sha256 of the three safetensors files, token ids. |
| `input_ids.json` | the token sequence. Tiny ids come from `tiny-random/inkling-2` at revision `0a57c72e4e5b68aafe1d983dbb62980344a5fcd7`. |

`artifacts/` is gitignored. The two micro checkpoints are about 6–8 MiB each and the activations another 3–8 MiB; the tiny checkpoints are a few hundred MiB. Regeneration is the command above; weights and tokens are seeded.

One thread and MKLDNN disabled. A full-sequence GEMM and a single-token GEMM otherwise disagree by more than `1e-6`.

## Measured

transformers 5.24.0, CPU, eager, fp32. Cached decode versus the full forward, max absolute logit error:

| config | weights | activations | max abs |
| --- | --- | --- | --- |
| `inkling2/micro` | 6,670,240 B | 3,334,688 B | 1.35e-7 |
| `inkling2/tiny` | 459,359,248 B | 148,938,016 B | 8.20e-8 |
| `inkling2_loop/micro` | 7,078,072 B | 8,173,368 B | 1.57e-7 |
| `inkling2_loop/tiny` | 552,940,504 B | 226,762,224 B | 7.08e-8 |

For every config, Q, K and relative clipping each fire on some vectors and miss others, separately for sliding and full layers. Even clip-scale heads are multiplied by `clip_scale_multiplier` and odd heads by `0.05`. Gate, up and logit softcaps (all set to `0.05`, not the stock 4 / 25 / 30) each move their inputs by at least `0.1`. The minimum router top-k margin is at least `1e-4`. `rms_norm_eps` is `1e-5`; the depth-mix norm stays at its class default `1e-6`.

Sites are small, often RMS around `1e-2`. Match a Rust dump with something like `max(1e-5, 1e-4 * rms(site))` rather than one absolute tolerance, and keep expert ids exact. The cached-versus-full check stays at `1e-6` absolute because both sides are the same graph.

The process exits if any of these fail:

- cached token-by-token decode differs from the full forward by more than `1e-6` absolute
- for Q, K or relative states, on sliding or full layers, clipping fires on none of the vectors or on all of them
- a softcap moves its input by less than `1e-2`
- the minimum top-k score gap is under `1e-4`
- the eager sliding mask is not `(query - window) < key <= query`, or the full mask is not `key <= query`
- relative bias is nonzero outside its extent
- `model.safetensors` is not the interleaved checkpoint layout of the transformers state dict

Hugging Face init writes ones into clip scales and RMSNorm weights, and zeros into the dynamic-convolution `static_weight`, the router correction bias, and the attention-over-depth queries. After that init, every parameter and every floating buffer is redrawn from `Normal(0, weight_std)`. Clip scales are then spread per head, as above.

## Configs

`inkling2/micro`. Hidden 64, 12 layers, 4 query / 2 KV heads of size 32, 8 experts choose 2, expert width 64, vocab 256 padded / 250 real. Sliding window 8, relative extent 16, log-scale floor 4. Sequence length 48, so the sequence crosses each of those.

`inkling2/tiny`. The tiny-random/inkling-2 text widths: hidden 128, 12 layers, 8/4 heads of 128, 96 experts choose 8, expert width 128, vocab 201024 / 200058. Windows are 32 / 64 / 16 rather than the stock 1024 / 1024 / 128000, and the sequence is 80 tokens (2.5 times the window). Stock numbers are in the manifest. Token ids are the tiny-random/inkling-2 chat template applied to a fixed prompt, tiled to 80.

`inkling2_loop/micro`. Same widths and windows as `inkling2/micro`. Layers `[6, 12)` run 3 times. A 12-layer stack has a pre-loop block and a loop block and no post-loop block, because both loop bounds have to be multiples of 6.

`inkling2_loop/tiny`. Mini's topology at a reduced width: 24 layers, layers `[6, 18)` run 3 times, 24/8 heads, 128 experts choose 8, vocab 201024 / 200058. Hidden 128, head size 32 (stock 128), expert width 64 (stock 1152), gate rank 32 (stock 64). Same small windows and 80-token chat sequence as `inkling2/tiny`. Stock Mini numbers are in the manifest.

Field names are the transformers attributes (`conv_kernel_size`, `moe_intermediate_size`, `loop_layer_index_start`, ...). The checkpoint aliases and the nested `loop_config` object are described in SEMANTICS.md.

## What a later Mini load has to do

`model.safetensors` is already in that checkpoint layout (interleaved gate/up, padded unembed, `model.llm.*` names). `model.transformers.safetensors` is the module `state_dict`. Quantization is not implemented: the real Mini shard stores NVFP4 experts in layers 0–19 and MXFP8 experts in layers 20–23, plus a separate MTP file. SEMANTICS.md records `Inkling2LoopDequantizeExperts`.
