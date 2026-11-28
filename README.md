# ammon-models

Model-specific decoders on [Ammon](https://github.com/koubaa/ammon) and [Goldy](https://github.com/koubaa/goldy).

The layers:

- **Goldy**: the GPU runtime and linear algebra, with no ML vocabulary: tensors, schemes, retained replay, automatic fusion.
- **Ammon**: model-agnostic inference operators and decode control: RMSNorm, projections, RoPE, attention, gated MLPs, mixture of experts, KV caches, tokenizers, greedy decode.
- **ammon-models** (this repository): one crate per model. A model crate maps its checkpoint layout onto Ammon modules and records its graph. It has no kernels of its own; Goldy's automatic fusion decides the dispatches. Anything generic a model needs goes into Ammon or Goldy.

## Models

| Crate | Model | Reference |
|-------|-------|-----------|
| [`models/inkling`](models/inkling) | Inkling 2 Small and Mini text decoders, FP32 | Hugging Face transformers on random weights, to 1e-5 |
| [`models/llama3`](models/llama3) | Karpathy's stories15M TinyStories checkpoint in llama2.c layout, FP32 | The llama3.cuda README sample, token for token |

## Layout

```
ammon-models/
├── Cargo.toml            workspace; Ammon and Goldy are workspace path dependencies
├── models/
│   ├── inkling/          crate inkling-goldy, plus its Python reference harness
│   └── llama3/           crate llama3-goldy, plus its benchmark harness
└── .github/
    ├── pins.env          the Ammon and Goldy commits CI builds against
    ├── actions/          shared checkout and lavapipe setup
    └── workflows/ci.yml
```

Ammon and Goldy are checked out as siblings of this repository:

```bash
git clone https://github.com/koubaa/ammon-models
git clone https://github.com/koubaa/ammon
git clone https://github.com/koubaa/goldy
```

Check out the commits in [`.github/pins.env`](.github/pins.env) to match CI.

## Build and test

```bash
cargo check --workspace
cargo test -p inkling-goldy --features vulkan -- --test-threads=1
cargo test --release -p llama3-goldy --features vulkan -- --test-threads=1
```

The Vulkan tests need a Vulkan 1.4 driver; CI uses Mesa's lavapipe (`VK_DRIVER_FILES` pointing at `lvp_icd.json`). Each model's README covers its assets: Inkling needs `sh models/inkling/tools/fetch_tokenizer.sh`, TinyStories needs `python3 models/llama3/tools/fetch_assets.py --only required`.

## CI

Every job builds against the pinned Ammon and Goldy commits. A `goldy_main` repository dispatch builds against Goldy main instead, and a manual run can override either ref.

| Job | What it checks |
|-----|----------------|
| `rust` | `cargo check` for the workspace, host tests for both crates |
| `inkling-reference-micro` | The two Inkling micro references and the cached-decode check (CPU, Python) |
| `inkling-lavapipe` | Inkling layer, forward and greedy goldens on lavapipe |
| `tinystories-lavapipe` | The stories15M reference output on lavapipe, then a steady-state decode bench in the job summary |

## License

MIT; see [LICENSE](LICENSE).
