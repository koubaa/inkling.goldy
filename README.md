# inkling.goldy

Inkling 2 text decoder on [Ammon](https://github.com/Koubaa/Ammon) and [Goldy](https://github.com/koubaa/goldy). The correctness target is FP32 on the Vulkan backend, matched against the Hugging Face transformers reference on random weights. Readable text is not a goal.

Two architectures:

- **Inkling 2 Small** text decoder (`model_type` `inkling2` in the multimodal checkpoint, `inkling2_text` for the decoder itself). Thinking Machines' Inkling-2-Small stack: sliding and full attention, relative-position bias, sinks, short convolutions, attention over depth, sigmoid top-k experts.
- **Inkling 2 Mini** (`model_type` `inkling2_loop`). The same decoder with 24 physical layers, of which layers 6–17 run three times on shared weights.

Text only. No vision tower, no multi-token prediction.

This tree currently holds the Python reference harness and a Rust crate skeleton. The Rust model is not implemented yet.

## Reference

```bash
python3 -m venv tools/ref/.venv
source tools/ref/.venv/bin/activate
pip install -r tools/ref/requirements.txt
python tools/ref/generate.py
```

Setup, configs, and the cached-decode check are in [tools/ref/README.md](tools/ref/README.md). The decoder behaviour the Rust side has to match, with transformers file and line references, is in [tools/ref/SEMANTICS.md](tools/ref/SEMANTICS.md).

Generated weights and activations go to `artifacts/ref/` and are not committed. `python tools/ref/generate.py` reproduces them.

## Crate

```bash
cargo check
```

`inkling-goldy` does not link Ammon or Goldy yet. The binary prints a pointer at the harness.
