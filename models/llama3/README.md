# llama3

FP32 TinyStories generator on [Ammon](https://github.com/koubaa/ammon) and [Goldy](https://github.com/koubaa/goldy), replicating [`llama3.cuda`](https://github.com/likejazz/llama3.cuda) commit `424333d1651d2b0fc17d38e9f790e824947e284b`. Imported with its history from [koubaa/llama3.goldy](https://github.com/koubaa/llama3.goldy).

This is **not** Meta Llama 3. The first checkpoint is Karpathy’s 15M TinyStories model (Llama-2-style MHA, RoPE θ=10,000) served through `llama3.cuda`’s layout and greedy loop. Ammon’s attention kernel keeps `n_kv_heads` generic so a later GQA checkpoint can exercise grouped-query attention. Llama 3 tokenizer / RoPE scaling / GGUF are out of scope.

There is **no CPU transformer**. Host code loads the checkpoint, tokenizes, tenders a `DecodeStep { token, position }` on the scheme's deposit, and greedy-argmaxes claimed logits. RMSNorm / GEMV / RoPE / attention / SwiGLU live in Ammon; this crate records the Llama graph and the llama2.c packed-blob layout. It has no kernels of its own; automatic fusion in Goldy decides the dispatches.

## Assets

```bash
python tools/fetch_assets.py
```

| File | Source | SHA-256 |
|------|--------|---------|
| `models/tokenizer.bin` | [karpathy/llama2.c](https://github.com/karpathy/llama2.c/raw/350e04fe35433e6d2941dce5a1f53308f87058eb/tokenizer.bin) | `50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361` |
| `models/stories15M.bin` | [karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas/resolve/0bd21da7698eaf29a0d7de3992de8a46ef624add/stories15M.bin) | `cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a` |
| `models/stories42M.bin` | [karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas/resolve/0bd21da7698eaf29a0d7de3992de8a46ef624add/stories42M.bin) | `9f65a1000e17d0bc167dd6332e0ce5119a0222a3d920cead5bce413bfab2ee7b` |
| `models/stories110M.bin` | [karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas/resolve/0bd21da7698eaf29a0d7de3992de8a46ef624add/stories110M.bin) | `515267168726a1ed1317a64a408492e6af3b67c1f71c5bd98c01d9d721803a24` |

All four paths are gitignored. The 42M/110M blobs are for the scaling tier in [BENCHMARKS.md](BENCHMARKS.md); `fetch_assets.py --only required` keeps the 15M + tokenizer pair.

## Run

Commands run from this directory; the workspace builds into the repository's `target/`.

Vulkan (lavapipe in CI):

```bash
cargo run --release --features vulkan -- "I have a dream"
```

CUDA (NVIDIA) and Metal (macOS) builds use `--features cuda` and `--features metal`.

Optional flags: `--checkpoint PATH`, `--tokenizer PATH`, `-n 50`.

The JSON benchmark harness is a separate binary (`llama3-goldy-bench`). Protocol, matrix runner, and adapters live in [BENCHMARKS.md](BENCHMARKS.md) and `tools/bench/`.

Expected sample (50 tokens, greedy, llama3.cuda README):

```
I have a dream. He dreams of a big, beautiful garden full of flowers and trees. He dreams of playing with his friends and eating yummy snacks.
One day, he was walking in the garden when he saw
```

Argmax is brittle under FP32 reduction-order differences. If a backend diverges on a later token, compare the prefix and `GOLDY_VALIDATION=api` traces before changing kernels.

## Tests

```bash
cargo test
cargo test --release --features vulkan -- --test-threads=1 --nocapture
```

`tests/generation.rs` is the reference-output gate: greedy decode of `"I have a dream"` must equal the llama3.cuda sample on the first pass, while specialization and fusion compiles land, and again once they have settled. It fails if the assets are missing. CI runs it on lavapipe.

Steady-state decode timing, cycling positions over the whole context once no compile is outstanding:

```bash
cargo test --release --features vulkan --test bench -- --ignored --nocapture
```

Kernel algebra lives in Ammon:

```bash
cargo test --manifest-path ../../../ammon/Cargo.toml --features vulkan --test kernels
```

## Goldy mapping

- Weights: one retained FP32 tensor blob; this crate maps llama2.c offsets to `CausalAttentionBlockWeights` / `SwiGluBlockWeights`
- Graph ops: Ammon modules (`Embedding`, `CausalAttentionBlock`, `SwiGluBlock`, `RmsNorm`, `Linear`); kernels stay inside Ammon
- Scratch: every buffer the scheme binds belongs to the model (`x`, the final-norm output, logits) or to a module it owns, and lives as long as the scheme
- `DecodeStep { token, position }`: Ammon control parcel (`DecodeStep::parcel` / `deposit_target`); `MemoryExchange` deposit on the scheme root. Tender with `<<` each step; one `submit()` per token
- KV cache: Ammon `KvCache`; each layer is `[seq_len, n_kv_heads, head_size]` (no `loff`)
- Scheme: one per token, under Ammon's inference arithmetic policy, with automatic fusion on (`GOLDY_FUSION=0` opts out). Ammon records named groups once (`embed`, `layerN/attn`, `layerN/ffn`, `tail`); parcel accesses derive their ordering. Specialization and fusion promotions re-record; `topology_records == 0`
- Logits: host sink claim `(&mut submission >> &sink).take::<f32>()` after each submit, copied out for `AutoregressiveModel::step`
