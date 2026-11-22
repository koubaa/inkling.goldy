# Goldy / Ammon / inkling.goldy restructure

Pinned by the architecture review on [Ammon #9](https://github.com/koubaa/ammon/pull/9#issuecomment-7074204072). Ammon #9 stays open and unchanged. Nothing in this series is merged by the implementing agent.

Reference points: Ammon `b0727eb` (PR #9 head), goldy `2354ef6`, this repo `604c7c4`, llama3.goldy `f4ae77e`. Semantics are `tools/ref/SEMANTICS.md` against transformers 5.24.0 `eeee1109`.

llama3.goldy is the pattern. It owns the checkpoint, the layer loop, and `set_automatic_fusion(true)`. Ammon owns `Embedding`, `CausalAttentionBlock`, `SwiGluBlock`, `KvCache`, `Linear`, `RmsNorm`. Goldy owns the tensor algebra those modules record, and fusion turns a SwiGLU epilogue into the same dispatch as the projection.

## Target

**goldy.** Runtime and symbolic algebra. No neural-net layers.

- Elementwise and reductions that already exist (`add`, `mul`, `tanh`, `sigmoid`, `sqrt`, `reciprocal`, `sum`, `mean`, `softmax`, `matmul`) are the implementation of RMSNorm, L2 clip, softcap, sigmoid gates, SwiGLU, and the depth-mix weighted sum.
- New: an indexed matmul/matvec. The weight operand's byte offset is `expert_id * stride`, and `expert_id` is read from a device `u32` buffer. Lowering is the existing matmul path (cuBLAS / CUDA sm80+ tensor cores when the backend has them, matvec on Vulkan).
- A kernel with a tensor formal is not a fusion site (`fusion_plan.rs`, `fusable_kernel`). Ops that must fuse are recorded as tensor algebra, not `#[goldy::compute]` kernels.

**Ammon.** Model-agnostic batch-1 decode on that algebra.

- `Linear`, decode attention (relative bias, sink, GQA, split-K), ring and full KV, short conv as a position-addressed ring, MoE whose experts are the indexed matmul, and a name-free safetensors reader (dtype, shape, bytes, upload).
- No Inkling parameter names, no loop schedule, no depth tape, no `model.llm.*` map.

**inkling.goldy.** The model.

- Checkpoint name map (transformers names and `model.llm.*`, including `w13` de-interleave).
- One decoder layer composition, the loop (24 layers, layers 6–17 × 3), and the only depth-tape owner.
- Message mix. Per-visit state that owns KV, the K/V convs, and the post-attention and post-MLP convs.
- Parity tests against the pinned modules. Compact binaries. Tolerance `max(1e-5, 1e-4 * rms(site))` per site; cached-versus-full stays `1e-6` absolute.

## File moves

| Today (Ammon `b0727eb`, unmerged #9 or already on main) | After |
| --- | --- |
| `src/checkpoint.rs` Inkling name map and `DecodeLayerSpec` flags | `inkling.goldy` `src/checkpoint.rs`. Ammon keeps a name-free reader. |
| `src/decode_layer.rs` layer schedule, message-gate wiring | `inkling.goldy` `src/layer.rs` |
| `DepthPlan`, `DepthRunner`, the tape snapshot and the message-key slice | `inkling.goldy` `src/tape.rs`. One owner. |
| `message_mix` kernel | deleted. The gate is tensor algebra recorded by the layer. |
| `depth_mix` kernel | deleted once the weighted sum is tensor algebra. Ammon may keep a thin helper that records that algebra. |
| `src/moe.rs` `moe_gate_up` / `moe_down_accum` | deleted after experts call indexed matmul. `topk` stays until goldy grows a `u32` top-k; the ids it writes must be a `u32` buffer either way. |
| Hand-written softcap, sigmoid gate, softcapped SwiGLU, L2 clip, RMSNorm kernels | Ammon records the same formulas with `TensorRecorder`. Kernels remain only where fusion cannot see a stateful or attention-shaped op (short conv ring, split-K attention). |
| `DecodeLayerState` (KV, K conv, V conv, split-K partial) | stays in Ammon. Gains the two post conv rings so one visit's four conv states travel with the visit. |
| `tests/gen_layer_goldens.py`, `tests/gen_depth_goldens.py`, the layer safetensors | move with the model. Ammon goldens stay per op. |

`llama3.goldy` does not move. It keeps calling Ammon's generic blocks.

## Goldy APIs this series adds

`TensorRecorder::matmul_indexed_into(label, weight, x, ids, slot, out) -> Tensor`:

- `weight` is a packed `[experts, rows, k]` tensor. `ids` is a rank-1 `u32` tensor. `slot` selects which id this dispatch reads.
- The implementation copies `weight[ids[slot]]` on device into the returned slab, then calls `matmul_into` on that slab. An id `>= experts` copies zeros and does not read past `weight`. The caller keeps the returned slab alive for every submit.
- The multiply is the ordinary matmul lowering: Vulkan matvec, and cuBLAS / tensor cores when that desc is eligible. The id itself stays a device buffer, so a replay that rewrites `ids` selects a different expert without re-recording.
- A test records two experts, sets the id to expert 1, and matches that slice times `x`. A second submit with a different id matches the other expert. An out-of-range id yields zeros.

No new elementwise opcode is required for the FP32 formulas:

- RMSNorm: `square = mul(x, x)`, `mean`, `add` eps, `sqrt`, `reciprocal`, `mul` by `x` and by `weight`. Unweighted skips `weight`.
- Softcap: `tanh(x / cap) * cap`.
- Sigmoid gate: `mul(values, sigmoid(logits))`.
- SwiGLU: `mul(mul(softcap(gate), sigmoid(gate)), softcap(up))`.
- L2 clip: `sum` of squares, `sqrt`, `min(1, threshold / (norm + 1e-6))`, optional `log` position scale, `mul`.
- Depth-mix weighted sum: per-key unweighted RMS (`eps = 1e-6`), scores `dot(q, key_hat) * hidden^{-0.5}`, `softmax` over the key axis, then a weighted sum of the normalized keys. Queries are not normalized. No mask and no sink.

If a broadcast or a reduction rank blocks one of those recordings, that gap is fixed in the goldy PR before Ammon depends on it. The acceptance test is a fused `Scheme::fusion_report` region, not a new kernel name.

## Dispatch counts per visit

Counts are recorded scheme nodes on Ammon `b0727eb` `DecodeLayer::record` for one token, before fusion. `Kv` is the KV head count. `K` is MoE top-k. A message gate is 9 nodes (2 unweighted RMS, 2 copies into the concatenated gate input, down projection, two tanh projections, the mix). Attention is 16 nodes plus one value-RMS node per KV head (Q/K/V projections and clips, two convs, KV store, relative projection, sink, split-K partial and combine, output gate, output projection). The tail is the post-attention conv, the boundary store or residual add, the MLP depth mix, one message gate, the post norm, MoE (`3 + 3K`: zero, router, top-k, and per expert a gate-up, a SwiGLU, and a down-accumulate), the post-MLP conv, and the residual add.

| Visit | Recorded nodes today |
| --- | --- |
| Sliding, pass 0 (no messages), boundary | `25 + Kv + 3K` |
| Sliding, later pass (4 message gates), boundary | `61 + Kv + 3K` |
| Full, later pass (dynamic K/V adds 2 tap projections) | `63 + Kv + 3K` |

Mini (`Kv = 8`, `K = 8`): pass 0 is 57 nodes, a later sliding pass is 93, a later full pass is 95.

After this series, fusion is on (llama3.goldy already does this unless `GOLDY_FUSION=0`). The recorded IR stays as the algebra steps; `executed_node_count` is what has to drop. Target for a later-pass sliding visit, once the semantic regions promote:

| Piece | Executed dispatches |
| --- | --- |
| Attention depth mix, MLP depth mix | 2 |
| Four message gates, each one fused region | 4 |
| Q projection+clip+log, K projection, K conv, K clip, V projection, V conv, V RMS, KV store | 8 |
| Relative projection+clip, bias gather, sink | 3 |
| Split-K partial + combine | 2 |
| Output-gate projection, output projection with the sigmoid fused in | 2 |
| Post-attention conv, post-MLP conv (residual adds fuse into the conv) | 2 |
| Boundary tape store | 1 |
| Router projection (post-norm fused in), top-k | 2 |
| `K` gate-up matmuls with SwiGLU fused, `K` down matmuls | `2K` |

That is `26 + 2K` (42 on Mini) against 93 recorded nodes today. The conv rings stay their own dispatches because they are stateful. Split-K attention stays two kernels. The acceptance check is `fusion_report`: each message gate, each SwiGLU, and each RMS-then-projection pair is one semantic region, and a clean replay's `executed_node_count` matches the table.

## Correctness fixes that travel with the Ammon PR

These are not on #9. #9 is not edited.

- **Conv rings are position-addressed.** A submit at position `p` writes slot `p` (modulo the ring) and reads the previous `kernel - 1` slots. Submitting the same `p` again, which warmup and specialization do, rewrites that slot and does not shift the history. The retained replay of the last golden token deposits that same position and the output still matches.
- **One deposit check.** `DecodeKvCache::deposit_position` is the only host check that `position < capacity`. The in-kernel guard still refuses an out-of-range read or write. The two host checks that grew in parallel (record-time and deposit-time) become the deposit check only.
- **Split-K scratch.** `DecodeLayerState::partial` is `[q_heads, attended.div_ceil(8), head_dim + 2]`. `attended` is the sliding window, or the full-cache capacity. The `+ 2` is the running max and the sum of exp. A visit with a different window or capacity does not share that buffer.

The interleaved test lives in inkling.goldy: one layer module, two visit states, tokens alternating between the states. Each state's post-attention conv history matches a solo run of that state. A clean resubmit of the last token still matches the golden.

## PR sequence

Dependencies are git SHAs, not merges. Ammon CI's `GOLDY_REF` is set to the goldy PR head. inkling.goldy will pin both SHAs the same way once it builds the model. The owner merges.

1. **This PR.** The plan. No code.
2. **goldy.** `matmul_indexed_into` and the `u32` id buffer. Tests for expert selection, a second id on replay, and an out-of-range id. Vulkan matvec and, where the runner has CUDA, the same desc the ordinary matmul lowers. No Ammon changes.
3. **Ammon**, off current `main`, not off #9. `GOLDY_REF` moves to the goldy PR head. Tensor-algebra recordings replace the pointwise kernels. MoE experts call indexed matmul. Name-free safetensors reader. Conv ring, single deposit check, split-K comment. Per-op goldens stay `<= 1e-5` against the pinned formulas. Fusion assertions on the algebra sites. Inkling names and the tape schedule are not added here; the copies on #9 are left for the model PR to replace.
4. **inkling.goldy.** Decoder layer, single tape owner, loop schedule, name map, per-visit state with all four conv rings, interleaved two-state test, retained-replay numeric check. Goldens from the pinned modules, compact binary plus safetensors. `fusion_report` meets the table above for one later-pass visit.

## What #9's questions resolve to

- Message-gate RMSNorm uses `rms_norm_eps`. Depth-mix RMSNorm uses `1e-6`. That is `SEMANTICS.md` and stays.
- The outer tape append belongs to the single tape owner in this repo. The layer's MLP mix reads the tape plus, on a boundary, the one local row the layer writes for itself.
- `cache_offset` stays loop schedule. The layer receives the visit state the schedule picked.
- Post-attention and post-MLP conv state move onto that visit state.
