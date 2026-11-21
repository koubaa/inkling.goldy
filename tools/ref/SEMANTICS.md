# Inkling 2 text-decoder semantics

Line numbers are from transformers **v5.24.0**, commit `eeee11099135592f1d03c2871fbd3d8132e5a4d7`. The two modeling files at that commit are what `generate.py` hashes against `pin.json`.

Shared text-decoder behaviour is cited from `src/transformers/models/inkling2/modeling_inkling2.py`. `inkling2_loop` vendors its own copy of those modules (`modeling_inkling2_loop.py`); the formulas match. Loop-only behaviour is cited from the loop file. Mask inequalities are from `src/transformers/masking_utils.py`. Checkpoint loading, which this harness does not implement, is from `src/transformers/conversion_mapping.py` and `src/transformers/core_model_loading.py`.

The reference model is the **text** causal LM (`Inkling2ForCausalLM`, config `model_type = "inkling2_text"`, and `Inkling2LoopForCausalLM`, config `model_type = "inkling2_loop_text"`). There is no vision tower and no multi-token-prediction head. The published checkpoints wrap this decoder in `Inkling2ForConditionalGeneration` / `Inkling2LoopForConditionalGeneration` (`model_type` `inkling2` / `inkling2_loop`) and add both.

Attention is eager. SDPA, flash and flex are disabled on the text stack because every layer adds a relative-position bias and a sink (`modeling_inkling2.py` 1275–1280). The harness forces `_attn_implementation = "eager"`. Do not install `causal-conv1d` or the Hub kernels package; the dump is the pure PyTorch convolution below.

## Layer pattern

`layer_types[i]` is `"hybrid"` when `(i + 1) % n_layers_per_attn_res_block == 0`, otherwise `"hybrid_sliding"` (`configuration_inkling2.py` 172–176, same rule in the loop config). With the stock block size of 6 that is five sliding layers and one full layer, and the full layers are indices 5, 11, 17, 23, ...

`"hybrid_sliding"` builds `Inkling2SlidingDecoderLayer` / the loop equivalent. `"hybrid"` builds the full-attention decoder layer. Sliding layers use `swa_*` head counts and `swa_head_dim`. Full layers use `num_attention_heads`, `num_key_value_heads` and `head_dim` (`modeling_inkling2.py` 554–560 and 647–653). In the released configs those two sets are equal; the micro and tiny configs here keep them equal too. Nothing requires `hidden_size == num_heads * head_dim`.

`local_layer_ids` in the published `config.json` is not a transformers field. The pattern above is what runs.

## Forward, per layer

Hidden states stay in the module dtype. The reference is fp32, so the casts are no-ops, but several blocks compute in fp32 even when the parameters are not (`modeling_inkling2.py` 110–115, 349, 411, 498–500, 773, 809).

Let `x` be the residual stream entering the layer, and `blocks` the attention-over-depth keys accumulated so far (empty at layer 0).

1. **Attention stream.** Layer 0 has no `attention_input` (`modeling_inkling2.py` 867, 924). Every later layer runs attention-over-depth (below) to produce the attention input.
   - Sliding: one stream. It is RMSNorm'd by `input_layernorm` and that tensor feeds Q, K and V (`modeling_inkling2.py` 896–897).
   - Full: three streams (query, key, value), each with its own RMSNorm (`modeling_inkling2.py` 952–962).
   - Loop layers on passes after the first mix a message into each stream *before* that norm. See the loop section.
2. **Attention**, then `o_proj`.
3. **Post-attention short convolution** on the attention output (`conv_idx = 2`).
4. **Block boundary and residual.** `is_block_boundary` is `layer_idx % n_layers_per_attn_res_block == 0` (`modeling_inkling2.py` 866), so the boundary is the *first* layer of a block (0, 6, 12, ...), not the full-attention layer at the end.
   - Boundary: the post-attention convolution is the residual by itself. The layer input `x` is appended to the local `blocks` used by this layer's MLP mix (`modeling_inkling2.py` 906–908). It is not added onto the attention output.
   - Otherwise: residual = `x + post_attention_conv` (`modeling_inkling2.py` 909–910). `blocks` is unchanged inside the layer.
5. **MLP stream.** `mlp_input` attention-over-depth, then (loop, later passes) the MLP message gate, then `post_attention_layernorm`.
6. **Router and experts**, then the post-MLP short convolution (`conv_idx = 3`).
7. **Layer output** = residual + post-MLP convolution (`modeling_inkling2.py` 915).

After the layer returns, the caller appends the *layer input* (not the layer output) to the shared `blocks` when the layer is a boundary (`modeling_inkling2.py` 1433–1435). The inner append and this outer append carry the same vector; the inner one exists so the boundary layer's own MLP can see it, while its attention mix ran before the append.

So a block state is the residual stream *entering* a boundary layer: embeddings at layer 0, and the previous block's output at layers 6, 12, .... The depth mix never sees a layer's own output as a completed block until a later boundary.

After the last layer, `final_input` mixes `blocks` with the last layer output, and `norm` RMSNorms that (`modeling_inkling2.py` 1437–1438). `lm_head` is a bias-free linear to `unpadded_vocab_size` rows, cast to fp32, then softcapped (`modeling_inkling2.py` 1514–1516). Embeddings and the unembedding are not tied (`modeling_inkling2.py` 1447–1448).

## Attention over depth

`Inkling2BlockAttentionResidual` (`modeling_inkling2.py` 833–858):

- Keys and values are `cat([blocks, partial], dim=block)` then an **unweighted** RMSNorm (no gamma) over the hidden axis. The norm is constructed as `Inkling2UnweightedRMSNorm()` with the class default `eps = 1e-6`. It does **not** read `rms_norm_eps`. The message gate's norm does (`modeling_inkling2_loop.py` 83).
- Queries are a learned parameter, broadcast to every token. Sliding attention and the MLP and final mixes use one query of shape `[hidden]`. Full-attention layers use three, shape `[3, hidden]`, and unbind them into the query, key and value streams. Init sets the queries to **zero** (`modeling_inkling2.py` 1312–1313); the harness re-randomizes them.
- Attention is over the block axis only. The token axis is a batch axis. There is no causal mask and no sink. `is_causal = False`, and eager attention applies a mask only when one is passed (`modeling_inkling2.py` 533–534), so every block is visible, including the current partial.
- Scale is `hidden_size ** -0.5`. Softmax is fp32.
- Output streams have shape `[batch, T, hidden]`.

`partial` is the current layer input for the attention mix, and the post-attention residual (see step 4) for the MLP mix. The final mix's partial is the last layer output.

This state is per token. It is not stored in the KV cache. A decode step rebuilds it from that token's own layer inputs.

## Attention

Eager attention (`modeling_inkling2.py` 515–546):

```
scores = (Q K^T) * head_dim**-0.5 + relative_bias + mask
```

`repeat_kv` expands KV heads to query heads by repeating each KV head `num_query_heads / num_kv_heads` times (`modeling_inkling2.py` 503–512).

**Sink.** `sink_proj` is a bias-free linear to `num_heads`, giving one logit per head per query (`modeling_inkling2.py` 622). That logit is concatenated on the key axis. Softmax runs over keys-plus-sink in fp32, and the sink column is dropped. The sink has no value vector; dropped mass is a zero contribution. There is always a sink (`modeling_inkling2.py` 536–539).

**Mask, full layers.** Causal, including the current token (`masking_utils.py` 74–78):

```
key_index <= query_index
```

Indices are absolute positions. A kept position is `0` in the additive mask; a dropped position is `torch.finfo(dtype).min`.

**Mask, sliding layers.** Causal and the window overlay (`masking_utils.py` 90–98 and 132–136):

```
query_index - sliding_window < key_index <= query_index
```

The left edge is strict `>`. A query sees `sliding_window` keys: distances `0, 1, ..., sliding_window - 1`. Distance `== sliding_window` is masked. `sliding_window` is the attribute alias of `sliding_window_size` (`configuration_inkling2.py` 127, `masking_utils.py` 1196–1201).

The cache stores `sliding_window - 1` past keys and returns those plus the new keys (`cache_utils.py`, `DynamicSlidingWindowLayer.update`, around 244–247). `get_mask_sizes` reports `kv_length = sliding_window - 1 + query_length` once `cumulative_length >= sliding_window`, with `kv_offset = max(cumulative_length - sliding_window + 1, 0)` (around 261–271). The visible span is still the inequality above.

On an uncached full sequence the sliding and full relative-distance matrices are the same `query_position - key_position`. The masks differ. `generate.py` checks both inequalities on the dumped masks.

**Relative bias** (`modeling_inkling2.py` 196–227):

- A bias-free linear maps the attention-stream hidden state (the same tensor Q was projected from, not K's stream) to `num_heads * d_rel`, viewed as `[batch, T, heads, d_rel]`.
- Sliding layers L2-clip that with a learned per-head scale (`Inkling2ScaledL2Clip`). Full layers clip and then apply the log scale (`Inkling2LogScaledL2Clip`).
- `proj` has shape `[d_rel, rel_extent]`. The extent is `sliding_window_size` on sliding layers and `rel_extent` on full layers (`modeling_inkling2.py` 560, 653).
- Distance `d = query_position - key_position` is clamped into `[0, rel_extent - 1]` only for the gather. The gathered value is then **replaced by 0** when `d < 0` or `d >= rel_extent` (`modeling_inkling2.py` 225–227). Negative distances and distances past the extent contribute no bias. They are not clamped to the edge profile.
- On sliding layers the extent equals the window, so the bias is nonzero only on keys the mask keeps (distances `0 .. window-1`). On full layers, keys with `distance >= rel_extent` are still attended, with a zero relative bias.

**Log scale**, full layers only, applied to queries and to relative states after their L2 clip, not to keys (`modeling_inkling2.py` 162–176, 190–193):

```
scale(p) = 1 + log_scaling_alpha * log(max((p + 1) / log_scaling_n_floor, 1))
```

`p` is the absolute position id. Positions with `p + 1 <= log_scaling_n_floor` get scale 1. The scale multiplies the `[batch, T, heads, dim]` tensor on the token axis.

**L2 clip** (`modeling_inkling2.py` 133–144, 147–159):

```
y = x                         # K, and the inner clip of a log-scaled projection
y = x * scale                 # Q and relative states; scale is [heads, dim], init ones
y = y * min(1, threshold / (||y||_2 + 1e-6))
```

Queries and keys use `qk_l2_clip_threshold`. Relative states use `rel_l2_clip_threshold`. The clip fires when `||y|| + 1e-6 > threshold`. Values are not L2-clipped; they get an unweighted RMSNorm with `eps = rms_norm_eps` (`modeling_inkling2.py` 466–470).

The published checkpoints use `qk_l2_clip_threshold = 22.627...` (`2 * sqrt(128)`) and `rel_l2_clip_threshold = 8`. The reference configs set both lower, each config to a value inside its own pre-clip norm distribution, so that at the weight scale that keeps cached decode inside 1e-6 of the full forward, Q, K and relative clipping all actually fire. The formula is unchanged; the thresholds are config values.

Q, K and V projections are bias-free. The harness multiplies every learned `scale` by `clip_scale_multiplier` after the Gaussian redraw and refuses to continue if Q, K or relative clipping never fires on the dump sequence.

**Short convolutions.** Kernel size is `conv_kernel_size` (alias `sconv_kernel_size`), stock 4. Four states per layer: K (`conv_idx 0`), V (`conv_idx 1`), post-attention (`conv_idx 2`), post-MLP (`conv_idx 3`). `number_of_conv_states = 4` is a class attribute, not a serialized field.

Static convolution (sliding K/V, and both post-attention and post-MLP convolutions on every layer), `modeling_inkling2.py` 286–350:

- Depthwise, bias-free, one kernel per channel. Causal via left padding of `kernel - 1` (the `F.conv1d` padding is symmetric; the implementation keeps the left-aligned `seq_len` outputs, 273–279).
- Output is `(conv(x) + x) / sqrt(kernel + 1)`. For kernel 4 that is `/ sqrt(5)`.
- The module is cast to fp32 at init (`modeling_inkling2.py` 464, 873–878).

Dynamic convolution (full-attention K and V only), `modeling_inkling2.py` 353–412:

- `static_weight` has shape `[kernel, hidden]` and **tap 0 is the current token**. Init zeros it (`modeling_inkling2.py` 1310–1311). The harness redraws it.
- Per-token taps are a bias-free linear of the *pre-projection* hidden state (the K or V stream, not the projected state) to `num_kv_heads * kernel`, shared by the channels of a KV head (`modeling_inkling2.py` 480–483, 387–389).
- Weights are `static_weight + dynamic_taps`. The window is flipped so lag 0 is the current token.
- Output is `(conv(x) + x) / sqrt(2 * kernel + 1)`. For kernel 4 that is `/ 3`. This is a different normalizer from the static convolution.

**Output projection** (`modeling_inkling2.py` 488–500). `gate = sigmoid(W_gate x)` in fp32, with `x` the attention-stream hidden state. The attention output (heads flattened) is multiplied by `gate`, then a bias-free linear maps it back to `hidden`.

## Router and experts

`Inkling2TopKRouter.forward` (`modeling_inkling2.py` 807–816), all fp32:

```
logits = x @ W^T                         # W is [n_experts, hidden], no bias parameter
scores = sigmoid(logits)                 # scoring_func, stock "sigmoid"
ids    = topk(scores + e_score_correction_bias, k, sorted=False)
weights = scores.gather(ids)             # bias does not enter the weights
weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
weights = weights * route_scale
```

`e_score_correction_bias` is a buffer of zeros, not a parameter (`modeling_inkling2.py` 805, 1307). The harness redraws it. `route_scale` is the attribute behind the alias `routed_scaling_factor`. `n_routed_experts` is the attribute behind `num_local_experts`.

There is no shared expert in the module. The published `n_shared_experts: 0` is not a config field. The expert loop builds a one-hot of width `n_experts + 1` and skips the extra class (`modeling_inkling2.py` 775–780); top-k indices stay in `0 .. n_experts-1`.

Experts (`modeling_inkling2.py` 757–793): `gate_up_proj` is `[n_experts, 2 * moe_intermediate_size, hidden]` and `down_proj` is `[n_experts, hidden, moe_intermediate_size]`. `F.linear` applies them. `chunk(2)` on the last axis splits **concatenated** `[gate | up]`, not an interleaved layout. The eager loop accumulates `index_add` in fp32 over experts that occur at least once.

SwiGLU (`modeling_inkling2.py` 733–753):

```
softcap(z) = tanh(z / cap) * cap
output = softcap(gate) * sigmoid(gate) * softcap(up)
```

`mlp_gate_softcap` (stock 4) bounds only the linear gate factor. `sigmoid` sees the raw gate. `mlp_up_softcap` (stock 25) bounds `up`. The same softcap, with `final_logit_softcapping` (stock 30), bounds the logits.

`moe_intermediate_size` is the attribute behind the alias `intermediate_size`.

## Norms

Weighted RMSNorm (`modeling_inkling2.py` 101–115), used for the token embedding, the attention and MLP input norms, and the final norm:

```
y = x * rsqrt(mean(x^2) + eps) * weight
```

Computed in fp32. `weight` is initialized to ones; the harness redraws it. `eps` is `rms_norm_eps` (stock `1e-6`).

Unweighted RMSNorm drops `weight`. Depth-mix norms use the default `eps = 1e-6` regardless of `rms_norm_eps`. Value norms and the message-gate norms use `rms_norm_eps`.

## What HF init hides

`_init_weights` (`modeling_inkling2.py` 1297–1313) and the module constructors:

| tensor | init |
| --- | --- |
| RMSNorm `weight`, clip `scale` | ones |
| dynamic conv `static_weight` | zeros |
| `e_score_correction_bias` | zeros |
| attention-over-depth `query`, including the loop message queries | zeros |

A zero message query is not a zero message: the depth-attention score is identically zero, so the softmax is uniform over the previous-pass blocks. A zero `static_weight` leaves only the dynamic taps. A ones `scale` with small projections often never reaches the clip threshold. The harness draws every parameter and every floating buffer from `Normal(0, weight_std)` with a per-config seed, then multiplies clip `scale` parameters by `clip_scale_multiplier`.

## Loop (`inkling2_loop`)

Config fields, after `loop_config` is flattened (`configuration_inkling2_loop.py` 144–154, `modeling_inkling2_loop.py` 1350–1352):

- Layers `[loop_layer_index_start, loop_layer_index_end)` are the loop. Both ends must be positive multiples of `n_layers_per_attn_res_block`, and the start must be `> 0` (`configuration_inkling2_loop.py` 156–170). Mini is start 6, end 18, 24 physical layers, `loop_num_passes = 3`.
- The published key `loop_config.is_varloop` is not read. A variable-pass checkpoint is the same architecture with a different `loop_num_passes`.
- There is **no per-pass embedding**. Passes differ because later passes see messages, and because each pass has its own KV cache and convolution state.

Execution (`modeling_inkling2_loop.py` 1359–1442):

1. Layers `[0, start)` run once. Their block states and the hidden state at `start` are the pre-loop snapshot.
2. Layers `[start, end)` run once with `messages = None` and `cache_offset = 0`. This pass appends its own boundary states on top of the pre-loop blocks.
3. For `loop_idx` in `1 .. num_passes-1`:
   - `previous_block_states = block_states[..., num_pre_loop_blocks + 1 :, :]` (`modeling_inkling2_loop.py` 1398–1400). The slice drops every pre-loop block **and one extra block**, which is the state appended at the first loop boundary (the hidden state that *entered* the loop). Remaining keys are boundary states recorded later in the previous pass. The previous pass's exit hidden state is passed separately as `partial`.
   - On the micro config the loop is a single 6-layer block, so that slice is **empty**. Messages then attend only to the previous pass's exit state. Mini (and the tiny loop config) also records the layer-12 boundary, and that state survives the slice.
   - Hidden state and `blocks` restart from the pre-loop snapshot. Earlier passes do not leave block states behind for the next pass, except through the messages.
   - `cache_offset = num_hidden_layers + (loop_idx - 1) * num_loop_layers - loop_start` (`modeling_inkling2_loop.py` 1405). Layer `i` uses cache slot `i + cache_offset`. Pass 0 uses slot `i`. For Mini that yields 48 cache layers (`24 + 2 * 12`), one per visit.
   - `message_input` (below) produces `[batch, T, 4, hidden]` for this loop layer: query, key, value, MLP. The layer mixes them in.
4. Layers `[end, num_hidden_layers)` run once, on the **last** pass's hidden state and on the block states rebuilt during that last pass (pre-loop blocks plus the last pass's boundary appends). Earlier passes affect post-loop layers only through the last pass.

The first time a cache is used, if it still has exactly `num_hidden_layers` slots, the forward deep-copies the loop layers' slots `num_passes - 1` times and appends them (`modeling_inkling2_loop.py` 1354–1357). Later decode steps see the longer cache and do not copy again. Masks and relative distances are built **once per forward**, from the first sliding cache layer and the first full cache layer, and reused for every layer and every pass (`modeling_inkling2_loop.py` 1332–1348). Passes stay aligned because each pass runs on every token.

**Message retrieval** (`modeling_inkling2_loop.py` 145–173). One query parameter of shape `[num_loop_layers, 4, hidden]`, initialized to zeros (and zeroed again in `_init_weights`, lines 416–419). Index 0..3 is query, key, value, MLP. For loop-layer-local index `j`, the query is `query[j]`. Keys and values are the sliced previous block states plus the previous pass's exit state, unweighted-RMSNorm'd. Same depth-attention rule as above (no mask, scale `hidden**-0.5`). Output is `[batch, T, 4, hidden]`.

**Message gate** (`modeling_inkling2_loop.py` 79–96), fp32:

```
h = unweighted_rmsnorm(hidden, eps=rms_norm_eps)
m = unweighted_rmsnorm(message, eps=rms_norm_eps)
z = down_proj(cat(h, m))          # to loop_gate_rank, bias-free
forget = tanh(W_forget z)         # W_forget, W_message are fp32, bias-free
gain   = tanh(W_message z)
output = (1 - forget) * hidden + gain * message
```

Sliding loop layers have four gates. On later passes the shared `input_layernorm` is applied **three times**, to the query, key and value gate outputs, in that order (`modeling_inkling2_loop.py` 264–270 in the modular source; the generated sliding loop layer matches). The dump records those as `attn_in_{q,k,v}` / `attn_norm_{q,k,v}`. Pass 0 calls the norm once and is recorded as `attn_in` / `attn_norm`. Full loop layers gate the three depth-mix streams and then apply `q_input_layernorm`, `k_input_layernorm`, `v_input_layernorm` (`modeling_inkling2_loop.py` 339–355).

KV and convolution state are **not** shared across passes. Each pass's cache slot starts empty. Depth-attention state is **not** carried across passes except as the message keys described above.

`total_num_hidden_layers` (`configuration_inkling2_loop.py` 172–176) counts visits: `num_hidden_layers + (num_passes - 1) * (end - start)`. `output_hidden_states` returns one entry per visit. This harness instead keys sites by physical layer and pass index.

## Cached decode

`use_cache=False` on the whole sequence is the activation dump. Token-by-token decode passes the returned `cache` back in and must match those logits to `1e-6` absolute. `generate.py` exits if it does not.

Positions, when not supplied, are `arange(query_length) + cache.get_seq_length()` (`modeling_inkling2.py` 1397–1400). `get_seq_length()` is the cumulative token count, not the cropped sliding window.

Single-token decode uses `causal_conv1d_update` for the static convolution when a conv state already exists (`modeling_inkling2.py` 330–333): concatenate the cached channels with the new token, depthwise-convolve, keep the last position, store the last `kernel` input channels. The dynamic convolution does not take that fast path; it always runs the unfold (`modeling_inkling2.py` 394–411) on `cache.update_conv_state`, which returns the cached left context concatenated with the new tokens (`cache_utils.py` around 1260–1282). The stored conv state length is the kernel size.

`create_recurrent_attention_mask` returns `None` for an unpadded sequence and for a single-token decode (`masking_utils.py` 1516–1523). With no padding, convolutions see every token. The harness does not insert padding.

## Checkpoint loading (not implemented)

The published Mini (and Small) weights are not this harness's layout. `conversion_mapping.py` renames `model.llm.*` onto the text module. The names the Rust loader will eventually see in a raw Thinking Machines shard, and where they go:

| checkpoint | transformers parameter |
| --- | --- |
| `model.llm.embed.weight`, `embed_norm.weight` | `embed_tokens.weight`, `embed_tokens.embed_norm.weight` |
| `model.llm.unembed.weight` | `lm_head.weight`, then drop rows past `unpadded_vocab_size` (`Inkling2UnpadVocab`, `core_model_loading.py` 856–878). Saving pads them back with zeros. |
| `layers.{i}.attn.wq_du` / `wk_dv` / `wv_dv` | `self_attn.{q,k,v}_proj.linear` |
| `layers.{i}.attn.wo_ud`, `gate_attn` | `o_proj.linear`, `o_proj.gate_proj` |
| `layers.{i}.attn.wr_du`, `rel_logits_proj.proj` | `relative_position_bias.linear`, `.proj` |
| `q_l2_clip.scale`, `rel_l2_clip.scale` | `q_proj.norm.scale`, `relative_position_bias.norm.scale` |
| `attn_sink_proj` | `sink_proj` |
| `k_sconv.weight` or `k_sconv.static_w` + `weight_proj` | static conv `conv1d.weight`, or dynamic `static_weight` + `weight_proj.weight`. Sliding layers have the static kernel; full layers have `static_w` and `weight_proj`. |
| `attn_norm` / `q_input_norm` / `k_input_norm` / `v_input_norm` | `input_layernorm` or the three full-attention norms |
| `mlp.gate.weight`, `mlp.gate.bias` | `mlp.gate.weight`, `mlp.gate.e_score_correction_bias` |
| `mlp.experts.w13_weight`, `w2_weight` | `gate_up_proj`, `down_proj` |
| `attn_res.attn_queries.{i}` | `layers.{i+1}.attention_input.query` (layer 0 has none; `Inkling2IndexOffsetWeightRename` offset 1, `core_model_loading.py` 1266–1294) |
| `attn_res.mlp_queries.{i}`, `attn_res.output_query` | `layers.{i}.mlp_input.query`, `final_input.query` |
| `controllers.{attn,k,v,mlp}.{down,forget,gain}_weight` | `{q,k,v,mlp}_message_gate.{down_proj,forget_gate_proj,message_gate_proj}.weight`. The checkpoint's `attn` controller is the query gate. |
| `middle_cycle_cross_pass_queries`, `middle_cycle_cross_pass_kv_queries` | merged into `message_input.query` |

`Inkling2LoopMergeMessageQueries` (`core_model_loading.py` 823–849) unflattens each `[2 * num_loop_layers, hidden]` matrix into `[num_loop_layers, 2, hidden]`. The first file holds `(query, mlp)` per loop layer and the second holds `(key, value)`. The merge order is query, key, value, mlp.

`w13` is interleaved `(gate, up, gate, up, ...)` along the fused intermediate axis. `Interleave(dim=1)` (`core_model_loading.py` 232–251) reshapes `[experts, 2I, hidden]` to `[experts, I, 2, hidden]`, swaps the last two groups, and reshapes back, which is concatenated `[gate | up]`. That is the layout `chunk(2)` consumes. `w2` is `down_proj` with no interleave.

Mini expert quantization (`hf_quant_config.json` on `thinkingmachines/Inkling-2-Mini`, `quant_method: "modelopt"`, quant algo NVFP4):

- Layers 0–19 experts are NVFP4. `Inkling2LoopDequantizeExperts` (`core_model_loading.py` 772–816): two E2M1 codes per byte, even element in the low nibble, values `(0, 0.5, 1, 1.5, 2, 3, 4, 6)` and their negatives; one fp8 scale per 16 values; per-expert factor `amax / outer_scale_denominator`. Block shape `[1, 16]`.
- Layers 20–23 experts are MXFP8: fp8 e4m3 values, power-of-two scale `2 ** (scale - 127)` per 32 values, block shape `[32, 32]`.
- The same op returns the tensor unchanged when only the weight is present (already dense). Small's MXFP8 path in `inkling2_text` keeps the scales as `*_scale_inv` and does **not** dequantize; the loop mapping replaces that with the dequant op (`conversion_mapping.py` 2654–2677). The dequant result is still run through `Interleave` for `w13`.
- Other decoder tensors are bf16, except the attention-over-depth queries, the router bias and the two message-query matrices, which are fp32. `unembed` is stored bf16 and computed in fp32.
- `mtp.safetensors` is a separate bf16 module. Out of scope, same as the vision tower (`model.visual.*`).

Published config keys that are not transformers fields: `q_bias`, `o_bias`, `pre_norm_pre_affine`, `final_norm_pre_affine`, `use_value_norm_no_gamma`, `attn_res_normalize_values`, `n_shared_experts`, `use_gate_bias`, `local_layer_ids`, and `loop_config.is_varloop`. The aliases that *are* honored are `sconv_kernel_size`, `intermediate_size`, `model_max_length`, `sliding_window`, `num_local_experts` and `routed_scaling_factor`.
