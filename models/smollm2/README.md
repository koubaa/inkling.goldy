# smollm2

[SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) on [Ammon](https://github.com/koubaa/ammon) and [Goldy](https://github.com/koubaa/goldy): a Hugging Face Llama checkpoint with grouped-query attention (9 query heads over 3 KV heads of 64), RoPE base 100000, RMSNorm eps 1e-5, SiLU-gated MLPs, and tied embeddings. The checkpoint is BF16; this crate widens it to f32 on load, which is exact.

There is no CPU transformer and there are no kernels here. The crate maps the checkpoint onto Ammon modules and records the graph; Goldy's automatic fusion decides the dispatches.

## Assets

```bash
python3 tools/fetch_assets.py
```

| File | Source | SHA-256 |
|------|--------|---------|
| `assets/config.json` | [HuggingFaceTB/SmolLM2-135M@93efa2f0](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/tree/93efa2f097d58c2a74874c7e644dbc9b0cee75a2) | `1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843` |
| `assets/tokenizer.json` | same revision | `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c` |
| `assets/model.safetensors` | same revision, 269 MB BF16 | `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1` |

`assets/` is gitignored. `--only tokenizer.json` fetches just the tokenizer for the host tests.

## Run

From this directory (the workspace builds into the repository's `target/`):

```bash
cargo run --release --features vulkan -- "The capital of France is" -n 32
```

`--features metal` and `--features cuda` select the other backends. Flags: `--assets DIR`, `-n TOKENS`, `--seq POSITIONS` (KV cache capacity, default 2048). Generation stops at `<|endoftext|>`.

## Tests

```bash
cargo test                                                   # host: config, packing, tokenizer
cargo test --release --features vulkan -- --test-threads=1   # GPU: reference against transformers
cargo test --release --features vulkan --test bench -- --ignored --nocapture
```

`tests/reference.rs` is the correctness gate. `tests/data/reference.json` is what `tools/ref/reference.py` gets from transformers' `LlamaForCausalLM` in float32 on the same weights: for three prompts, the 12-token greedy continuation and the top 8 logits at every position. The model decodes each prompt one token at a time from position 0. It must pick the same greedy tokens, with each reference top logit within 5e-3, on the first pass (authored graph) and again after specialization and fusion compiles settle. On lavapipe the largest logit error is about 7e-5 before fusion and 1.5e-4 after (Relaxed); under `MatchRecorded` the fused graph stays at 7e-5. The smallest greedy margin in the reference is 0.012.

Regenerate or check the reference with the same frozen environment as Inkling:

```bash
python3 -m pip install -r ../inkling/tools/ref/requirements.lock
python3 tools/ref/reference.py --check
```

The bench is steady-state and Relaxed only. It cycles positions over a 256-token context once no compile is outstanding, fails if the timed window records, and prints median, p10, and p90 tok/s.

## Goldy mapping

- Weights: each tensor widened from BF16 to f32 and packed in checkpoint order onto f32 slabs of at most 128 MiB, so each slab binds within lavapipe's storage-buffer range; no tensor straddles slabs. The LM head is a view of the embedding table.
- RoPE layout: Hugging Face Llama rotates each head's halves; Ammon rotates adjacent pairs at the same frequencies. Packing interleaves each head's `q_proj` and `k_proj` rows, which leaves every Q·K score unchanged. V and the output projection keep their order.
- Modules: one shared set (`Embedding`, `CausalAttentionBlock` with `with_rope_theta(100000)`, `SwiGluBlock`, `RmsNorm`, `Linear`) records every layer; a `LayerSpec` is only where a layer's weights live on the slabs.
- State: Ammon `KvCache` (one key and one value tensor over all layers, `[layers, seq, 3, 64]`), the residual `x`, the logits, and the `DecodeStep` parcel. Block intermediates and the final-norm output are `Scheme::temporary_tensor` scratch.
- Scheme: one retained scheme per token under `configure_inference_scheme`, automatic fusion on (`GOLDY_FUSION=0` opts out). Groups are `embed`, `layerN/attn`, `layerN/ffn`, and `tail`. `DecodeStep { token, position }` is tendered on the scheme's deposit; logits are claimed from a host sink.
