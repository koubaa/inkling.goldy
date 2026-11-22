# inkling.goldy

Inkling 2 text decoder on [Ammon](https://github.com/Koubaa/Ammon) and [Goldy](https://github.com/koubaa/goldy). The correctness target is FP32 on the Vulkan backend, matched against the Hugging Face transformers reference on random weights. Readable text is not a goal.

Two architectures:

- **Inkling 2 Small** text decoder (`model_type` `inkling2` in the multimodal checkpoint, `inkling2_text` for the decoder itself). Thinking Machines' Inkling-2-Small stack: sliding and full attention, relative-position bias, sinks, short convolutions, attention over depth, sigmoid top-k experts.
- **Inkling 2 Mini** (`model_type` `inkling2_loop`). The same decoder with 24 physical layers, of which layers 6–17 run three times on shared weights.

Text only. No vision tower, no multi-token prediction.

This tree holds the Python reference harness and the Rust decoder layer: the checkpoint name map, one decoder layer, and the only depth-tape owner (`src/tape.rs`). Ammon supplies the model-agnostic kernels.

## Reference

```bash
python3 -m venv tools/ref/.venv
source tools/ref/.venv/bin/activate
pip install -r tools/ref/requirements.lock
python tools/ref/generate.py
```

Setup, configs, and the cached-decode check are in [tools/ref/README.md](tools/ref/README.md). The decoder behaviour the Rust side has to match, with transformers file and line references, is in [tools/ref/SEMANTICS.md](tools/ref/SEMANTICS.md).

Generated weights and activations go to `artifacts/ref/` and are not committed. `python tools/ref/generate.py` reproduces them. CI runs `cargo check`, the lavapipe decoder-layer tests, and the two micro configs. The tiny configs are local-only because the checkpoints are hundreds of megabytes.

## Crate

Ammon and Goldy are path dependencies, checked out as siblings of this crate. CI pins `AMMON_REF` and `GOLDY_REF`.

```bash
cargo check
cargo test --features vulkan -- --test-threads=1
```

The Vulkan tests need lavapipe (`VK_DRIVER_FILES` pointing at `lvp_icd.json`). They replay a pinned sliding visit and a pinned full visit to 1e-5, including a retained resubmit, and check that two visit states do not share post-conv history. `inkling2_loop/mini` is the 24-layer Mini schedule (48 visits) at a small width; `python tools/ref/generate.py --only inkling2_loop/mini` then `python tools/ref/pack_mini.py` refreshes the committed logits.
