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
| `config.json`, `model.safetensors` | text causal LM. `model_type` is `inkling2_text` or `inkling2_loop_text`. |
| `activations.safetensors` | one full-sequence forward. Keys `L{i}.{site}` or, on looped layers, `L{i}.p{pass}.{site}`. Shape `[T, ...]`, fp32 (expert ids are int64). |
| `manifest.json` | site descriptions, layer table, clip counts, cached-decode comparison, token ids. |
| `input_ids.json` | the token sequence. |

`artifacts/` is gitignored. The two micro checkpoints are about 6–8 MiB each and the activations another 3–8 MiB; the tiny checkpoints are a few hundred MiB. Regeneration is the command above; weights and tokens are seeded.

One thread and MKLDNN disabled. A full-sequence GEMM and a single-token GEMM otherwise disagree by more than `1e-6`.

## Measured

transformers 5.24.0, CPU, eager, fp32. Cached decode versus the full forward, max absolute logit error:

| config | weights | activations | max abs |
| --- | --- | --- | --- |
| `inkling2/micro` | 6,670,240 B | 3,334,688 B | 1.49e-7 |
| `inkling2/tiny` | 459,359,248 B | 148,938,016 B | 1.19e-7 |
| `inkling2_loop/micro` | 7,078,072 B | 8,173,368 B | 2.09e-7 |
| `inkling2_loop/tiny` | 552,940,504 B | 226,762,224 B | 1.04e-7 |

Q clipping fired on essentially every query vector. K clipping fired on 18–47% of key vectors and relative-state clipping on 48–97%, depending on the config. Thresholds are in each `config.json`; they are tighter than the stock 22.627 / 8 so the clip is on the path at this weight scale.

The process exits if any of these fail:

- cached token-by-token decode differs from the full forward by more than `1e-6` absolute
- Q, K, or relative-state L2 clipping never fires
- the eager sliding mask is not `(query - window) < key <= query`, or the full mask is not `key <= query`
- relative bias is nonzero outside its extent

Hugging Face init writes ones into clip scales and RMSNorm weights, and zeros into the dynamic-convolution `static_weight`, the router correction bias, and the attention-over-depth queries. After that init, every parameter and every floating buffer is redrawn from `Normal(0, weight_std)` and the clip scales are multiplied by `clip_scale_multiplier`. Both numbers are in the manifest.

## Configs

`inkling2/micro`. Hidden 64, 12 layers, 4 query / 2 KV heads of size 32, 8 experts choose 2, expert width 64, vocab 256 padded / 250 real. Sliding window 8, relative extent 16, log-scale floor 4. Sequence length 48, so the sequence crosses each of those.

`inkling2/tiny`. The tiny-random/inkling-2 text widths: hidden 128, 12 layers, 8/4 heads of 128, 96 experts choose 8, expert width 128, vocab 201024 / 200058. Windows are 32 / 64 / 16 rather than the stock 1024 / 1024 / 128000, and the sequence is 80 tokens (2.5 times the window). Stock numbers are in the manifest. Token ids are the tiny-random/inkling-2 chat template applied to a fixed prompt, tiled to 80.

`inkling2_loop/micro`. Same widths and windows as `inkling2/micro`. Layers `[6, 12)` run 3 times. A 12-layer stack has a pre-loop block and a loop block and no post-loop block, because both loop bounds have to be multiples of 6.

`inkling2_loop/tiny`. Mini's topology at a reduced width: 24 layers, layers `[6, 18)` run 3 times, 24/8 heads, 128 experts choose 8, vocab 201024 / 200058. Hidden 128, head size 32 (stock 128), expert width 64 (stock 1152), gate rank 32 (stock 64). Same small windows and 80-token chat sequence as `inkling2/tiny`. Stock Mini numbers are in the manifest.

Field names are the transformers attributes (`conv_kernel_size`, `moe_intermediate_size`, `loop_layer_index_start`, ...). The checkpoint aliases and the nested `loop_config` object are described in SEMANTICS.md.

## What a later Mini load has to do

Not done here. The raw `thinkingmachines/Inkling-2-Mini` shard uses different tensor names, an interleaved gate/up expert layout, NVFP4 experts in layers 0–19 and MXFP8 experts in layers 20–23, a padded unembedding, and a separate MTP file. SEMANTICS.md records the transformers conversion, including `Inkling2LoopDequantizeExperts`.
