"""Forward hooks for one teacher-forced FP32 pass.

Keys are `L{layer}.{site}` and, on looped layers, `L{layer}.p{pass}.{site}`.
Pass 0 is the first visit. Tensors are stored as float32 (or int64 for expert
ids) with the batch dimension removed, leading axis the token axis, shape
`[T, ...]`.
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn as nn

from transformers.models.inkling2.modeling_inkling2 import (
    Inkling2L2Clip,
    Inkling2LogScale,
    Inkling2ScaledL2Clip,
)
from transformers.models.inkling2_loop.modeling_inkling2_loop import (
    Inkling2LoopL2Clip,
    Inkling2LoopLogScale,
    Inkling2LoopScaledL2Clip,
)

CLIP_TYPES = (Inkling2L2Clip, Inkling2ScaledL2Clip, Inkling2LoopL2Clip, Inkling2LoopScaledL2Clip)
LOG_SCALE_TYPES = (Inkling2LogScale, Inkling2LoopLogScale)

SITE_DOCS = {
    "embed": "Token embedding after its RMSNorm. [T, hidden]",
    "attn_depth": "Attention-over-depth mix that produces the attention stream, before a message gate. Sliding layers have one stream. [T, hidden]",
    "attn_depth_q": "Full-attention depth mix, query stream, before the message gate. [T, hidden]",
    "attn_depth_k": "Full-attention depth mix, key stream, before the message gate. [T, hidden]",
    "attn_depth_v": "Full-attention depth mix, value stream, before the message gate. [T, hidden]",
    "attn_in": "Attention input after the depth mix and (on later loop passes) the message gate, before the attention norm. [T, hidden]",
    "attn_in_q": "Full-attention query input after depth mix and message gate, before q_input_layernorm. [T, hidden]",
    "attn_in_k": "Full-attention key input after depth mix and message gate, before k_input_layernorm. [T, hidden]",
    "attn_in_v": "Full-attention value input after depth mix and message gate, before v_input_layernorm. [T, hidden]",
    "attn_norm": "Sliding-layer input RMSNorm, the tensor that enters q/k/v projections. [T, hidden]",
    "attn_norm_q": "Full-attention query RMSNorm. [T, hidden]",
    "attn_norm_k": "Full-attention key RMSNorm. [T, hidden]",
    "attn_norm_v": "Full-attention value RMSNorm. [T, hidden]",
    "q": "Queries after the learned scale, L2 clip and (full layers only) log scale. [T, heads, head_dim]",
    "k": "Keys after the short convolution and L2 clip. [T, kv_heads, head_dim]",
    "v": "Values after the short convolution and unweighted RMSNorm. [T, kv_heads, head_dim]",
    "rel_bias": "Relative-position bias added to attention logits. Zero where distance < 0 or distance >= rel_extent. [T, heads, Tk]",
    "sink": "Per-query sink logits from sink_proj, one per head. Concatenated as an extra key with an implicit zero value. [T, heads]",
    "attn_pre_o": "Attention output after the value mix, before the sigmoid gate and o_proj. [T, heads * head_dim]",
    "attn_out": "Attention output after the sigmoid gate and o_proj. [T, hidden]",
    "post_attn_conv": "Residual short convolution on the attention output. [T, hidden]",
    "mlp_depth": "Attention-over-depth mix that produces the MLP stream, before a message gate. [T, hidden]",
    "mlp_in": "MLP input after the depth mix and (on later loop passes) the message gate, before post_attention_layernorm. [T, hidden]",
    "mlp_norm": "post_attention_layernorm, the tensor that enters the router and experts. [T, hidden]",
    "router_logits": "Router linear outputs in fp32, before sigmoid. [T, n_experts]",
    "expert_ids": "Top-k expert indices. Chosen by sigmoid(logits) + correction bias, sorted=False. [T, k]",
    "expert_weights": "Renormalized sigmoid scores of the chosen experts, times route_scale. The bias is not part of the weight. [T, k]",
    "mlp_out": "Mixture-of-experts output before the post-MLP convolution. [T, hidden]",
    "post_mlp_conv": "Residual short convolution on the MLP output. [T, hidden]",
    "layer_out": "Decoder layer output, after the residual add. [T, hidden]",
    "messages": "Four messages (query, key, value, mlp) retrieved from the previous loop pass. Absent on pass 0. [T, 4, hidden]",
    "msg_q": "Message-gate output mixed into the query/attention stream. Later-pass loop layers only. [T, hidden]",
    "msg_k": "Message-gate output mixed into the key stream. [T, hidden]",
    "msg_v": "Message-gate output mixed into the value stream. [T, hidden]",
    "msg_mlp": "Message-gate output mixed into the MLP stream. [T, hidden]",
    "final_depth_mix": "Final attention-over-depth mix of the completed block states and the last layer output, before the final norm. [T, hidden]",
    "final_norm": "Final RMSNorm. This is the tensor the lm_head reads. [T, hidden]",
    "logits_pre_softcap": "lm_head output over unpadded_vocab_size, before the logit softcap. [T, unpadded_vocab]",
    "logits": "Softcapped logits returned by the model. [T, unpadded_vocab]",
    "mask.sliding": "Eager sliding-window mask shared by every sliding layer and every loop pass. 0 keeps a key, finfo.min drops it. [T, Tk]",
    "mask.full": "Eager causal mask shared by every full-attention layer. [T, Tk]",
    "rel_distance.sliding": "Query position minus key position for sliding layers. [T, Tk]",
    "rel_distance.full": "Query position minus key position for full-attention layers. [T, Tk]",
    "log_scale": "Per-position log scale applied to full-attention queries and relative states. [T]",
}


class ActivationDump:
    def __init__(self, seq_len: int):
        self.seq_len = seq_len
        self.tensors: dict[str, torch.Tensor] = {}
        self.clip = defaultdict(lambda: {"clipped": 0, "total": 0})
        self._handles: list = []
        self._wrapped = []

    def save(self, key: str, tensor: torch.Tensor, *, kind: str) -> None:
        if key in self.tensors:
            raise RuntimeError(f"activation key {key} was written twice")
        value = _take(tensor, kind, self.seq_len)
        self.tensors[key] = value

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for module, name, original in self._wrapped:
            setattr(module, name, original)
        self._wrapped.clear()


def _take(tensor: torch.Tensor, kind: str, seq_len: int) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"expected a tensor, got {type(tensor)}")
    # Clone so aliased sites (depth-mix output and the norm input, on a pass
    # with no message gate) can both be stored in one safetensors file.
    out = tensor.detach().clone()
    if out.dtype.is_floating_point:
        out = out.to(torch.float32)
    out = out.cpu().contiguous()
    if kind == "bth":
        if out.shape[0] != 1 or out.shape[1] != seq_len:
            raise RuntimeError(f"bth shape {tuple(out.shape)} does not start with [1, {seq_len}]")
        out = out[0]
    elif kind == "bhtd":
        if out.shape[0] != 1 or out.shape[2] != seq_len:
            raise RuntimeError(f"bhtd shape {tuple(out.shape)} is not [1, heads, {seq_len}, dim]")
        out = out[0].permute(1, 0, 2).contiguous()
    elif kind == "bhtk":
        if out.shape[0] != 1 or out.shape[2] != seq_len:
            raise RuntimeError(f"bhtk shape {tuple(out.shape)} is not [1, heads, {seq_len}, Tk]")
        out = out[0].permute(1, 0, 2).contiguous()
    elif kind == "flat":
        if out.shape[0] != seq_len:
            raise RuntimeError(f"flat shape {tuple(out.shape)} does not start with {seq_len}")
    elif kind == "dist":
        if out.shape[0] != seq_len:
            raise RuntimeError(f"distance shape {tuple(out.shape)} does not start with {seq_len}")
    elif kind == "scale":
        out = out.reshape(-1)
        if out.shape[0] != seq_len:
            raise RuntimeError(f"scale shape {tuple(out.shape)} is not [{seq_len}]")
    else:
        raise ValueError(kind)
    return out.contiguous()


def _site(module: nn.Module, name: str) -> str:
    layer = module._ref_layer
    if layer._ref_is_loop:
        return f"L{layer._ref_idx}.p{layer._ref_pass}.{name}"
    return f"L{layer._ref_idx}.{name}"


def _as_streams(output) -> tuple[torch.Tensor, ...]:
    if isinstance(output, torch.Tensor):
        return (output,)
    return tuple(output)


def install_hooks(model: nn.Module, dump: ActivationDump, *, loop_start: int | None, num_loop_layers: int) -> None:
    """Register hooks. Call `dump.close()` before the cached decode."""

    text = model.model
    layers = list(text.layers)
    loop_start = 0 if loop_start is None else loop_start

    def pre_layer(layer, args):
        layer._ref_pass = layer._ref_calls
        layer._ref_calls += 1

    for index, layer in enumerate(layers):
        layer._ref_idx = index
        layer._ref_is_loop = loop_start <= index < loop_start + num_loop_layers if num_loop_layers else False
        layer._ref_calls = 0
        layer._ref_pass = 0
        for module in layer.modules():
            module._ref_layer = layer
        layer._ref_shared_norm_in = []
        layer._ref_shared_norm_out = []
        dump._handles.append(layer.register_forward_pre_hook(pre_layer))
        dump._handles.append(layer.register_forward_hook(lambda mod, inputs, output: _finish_layer(dump, mod, output)))
        _hook_layer(layer, dump)

    dump._handles.append(
        text.embed_tokens.register_forward_hook(lambda mod, inputs, output: dump.save("embed", output, kind="bth"))
    )
    dump._handles.append(
        text.final_input.register_forward_hook(
            lambda mod, inputs, output: dump.save("final_depth_mix", _as_streams(output)[0], kind="bth")
        )
    )
    dump._handles.append(
        text.norm.register_forward_hook(lambda mod, inputs, output: dump.save("final_norm", output, kind="bth"))
    )
    dump._handles.append(
        model.lm_head.register_forward_hook(
            lambda mod, inputs, output: dump.save("logits_pre_softcap", output, kind="bth")
        )
    )

    if hasattr(text, "message_input"):
        _wrap_messages(text, dump, loop_start=loop_start, num_loop_layers=num_loop_layers)

    _hook_clips(model, dump)


def _hook_layer(layer: nn.Module, dump: ActivationDump) -> None:
    if layer.attention_input is not None:
        dump._handles.append(layer.attention_input.register_forward_hook(lambda mod, inputs, output: _save_streams(dump, mod, "attn_depth", output)))

    dump._handles.append(layer.mlp_input.register_forward_hook(lambda mod, inputs, output: _save_streams(dump, mod, "mlp_depth", output)))

    if hasattr(layer, "input_layernorm"):
        # One call on a normal sliding layer. On a later loop pass the same norm
        # is applied separately to the query, key and value streams, in that order.
        _hook_shared_norm(dump, layer.input_layernorm)
    else:
        _hook_norm(dump, layer.q_input_layernorm, "attn_in_q", "attn_norm_q")
        _hook_norm(dump, layer.k_input_layernorm, "attn_in_k", "attn_norm_k")
        _hook_norm(dump, layer.v_input_layernorm, "attn_in_v", "attn_norm_v")

    _hook_norm(dump, layer.post_attention_layernorm, "mlp_in", "mlp_norm")

    attn = layer.self_attn
    dump._handles.append(attn.q_proj.register_forward_hook(lambda mod, inputs, output: dump.save(_site(mod, "q"), output, kind="bhtd")))
    dump._handles.append(attn.k_proj.register_forward_hook(lambda mod, inputs, output: dump.save(_site(mod, "k"), output, kind="bhtd")))
    dump._handles.append(attn.v_proj.register_forward_hook(lambda mod, inputs, output: dump.save(_site(mod, "v"), output, kind="bhtd")))
    dump._handles.append(attn.relative_position_bias.register_forward_hook(lambda mod, inputs, output: dump.save(_site(mod, "rel_bias"), output, kind="bhtk")))
    dump._handles.append(attn.sink_proj.register_forward_hook(lambda mod, inputs, output: dump.save(_site(mod, "sink"), output, kind="bth")))
    dump._handles.append(attn.o_proj.register_forward_pre_hook(lambda mod, args: dump.save(_site(mod, "attn_pre_o"), args[0], kind="bth")))
    dump._handles.append(attn.o_proj.register_forward_hook(lambda mod, inputs, output: dump.save(_site(mod, "attn_out"), output, kind="bth")))
    dump._handles.append(attn.register_forward_pre_hook(_mask_hook(dump), with_kwargs=True))

    dump._handles.append(
        layer.post_attention_convolution.register_forward_hook(
            lambda mod, inputs, output: dump.save(_site(mod, "post_attn_conv"), output, kind="bth")
        )
    )
    dump._handles.append(
        layer.post_mlp_convolution.register_forward_hook(
            lambda mod, inputs, output: dump.save(_site(mod, "post_mlp_conv"), output, kind="bth")
        )
    )
    dump._handles.append(layer.mlp.register_forward_hook(lambda mod, inputs, output: dump.save(_site(mod, "mlp_out"), output, kind="bth")))
    dump._handles.append(layer.mlp.gate.register_forward_hook(lambda mod, inputs, output: _save_router(dump, mod, output)))

    for gate_name, site in (
        ("q_message_gate", "msg_q"),
        ("k_message_gate", "msg_k"),
        ("v_message_gate", "msg_v"),
        ("mlp_message_gate", "msg_mlp"),
    ):
        gate = getattr(layer, gate_name, None)
        if gate is not None:
            dump._handles.append(gate.register_forward_hook(lambda mod, inputs, output, site=site: dump.save(_site(mod, site), output, kind="bth")))


def _finish_layer(dump: ActivationDump, layer: nn.Module, output) -> None:
    dump.save(_site(layer, "layer_out"), output, kind="bth")
    inputs = layer._ref_shared_norm_in
    outputs = layer._ref_shared_norm_out
    if not inputs and not outputs:
        return
    if len(inputs) != len(outputs):
        raise RuntimeError(f"layer {layer._ref_idx} shared-norm inputs/outputs diverged")
    if len(inputs) == 1:
        dump.save(_site(layer, "attn_in"), inputs[0], kind="bth")
        dump.save(_site(layer, "attn_norm"), outputs[0], kind="bth")
    elif len(inputs) == 3:
        for suffix, tensor in zip(("q", "k", "v"), inputs):
            dump.save(_site(layer, f"attn_in_{suffix}"), tensor, kind="bth")
        for suffix, tensor in zip(("q", "k", "v"), outputs):
            dump.save(_site(layer, f"attn_norm_{suffix}"), tensor, kind="bth")
    else:
        raise RuntimeError(f"layer {layer._ref_idx} input_layernorm ran {len(inputs)} times")
    layer._ref_shared_norm_in = []
    layer._ref_shared_norm_out = []


def _hook_shared_norm(dump: ActivationDump, norm: nn.Module) -> None:
    def pre(module, args):
        module._ref_layer._ref_shared_norm_in.append(args[0])

    def post(module, inputs, output):
        module._ref_layer._ref_shared_norm_out.append(output)

    dump._handles.append(norm.register_forward_pre_hook(pre))
    dump._handles.append(norm.register_forward_hook(post))


def _hook_norm(dump: ActivationDump, norm: nn.Module, input_site: str, output_site: str) -> None:
    def pre(module, args):
        dump.save(_site(module, input_site), args[0], kind="bth")

    def post(module, inputs, output):
        dump.save(_site(module, output_site), output, kind="bth")

    dump._handles.append(norm.register_forward_pre_hook(pre))
    dump._handles.append(norm.register_forward_hook(post))


def _save_streams(dump: ActivationDump, module: nn.Module, stem: str, output) -> None:
    streams = _as_streams(output)
    if len(streams) == 1:
        dump.save(_site(module, stem), streams[0], kind="bth")
        return
    if len(streams) != 3:
        raise RuntimeError(f"{stem} returned {len(streams)} streams")
    for suffix, stream in zip(("q", "k", "v"), streams):
        dump.save(_site(module, f"{stem}_{suffix}"), stream, kind="bth")


def _save_router(dump: ActivationDump, module: nn.Module, output) -> None:
    logits, weights, indices = output
    dump.save(_site(module, "router_logits"), logits, kind="flat")
    dump.save(_site(module, "expert_weights"), weights, kind="flat")
    dump.save(_site(module, "expert_ids"), indices.to(torch.int64), kind="flat")


def _mask_hook(dump: ActivationDump):
    def pre(module, args, kwargs):
        kind = "sliding" if module.is_sliding else "full"
        mask_key = f"mask.{kind}"
        if mask_key not in dump.tensors:
            mask = kwargs.get("attention_mask")
            if mask is None:
                raise RuntimeError(f"{kind} attention was called without a materialized mask")
            dump.save(mask_key, mask.squeeze(1) if mask.shape[1] == 1 else mask[:, 0], kind="bth")
            distance = kwargs["relative_distance"]
            dump.save(f"rel_distance.{kind}", distance, kind="dist")
        return None

    return pre


def _wrap_messages(text: nn.Module, dump: ActivationDump, *, loop_start: int, num_loop_layers: int) -> None:
    module = text.message_input
    original = module.forward
    state = {"calls": 0}

    def wrapped(block_states, partial_block_states, loop_layer_idx, *args, **kwargs):
        call = state["calls"]
        state["calls"] += 1
        pass_idx = 1 + call // num_loop_layers
        layer_idx = loop_start + int(loop_layer_idx)
        output = original(block_states, partial_block_states, loop_layer_idx, *args, **kwargs)
        dump.save(f"L{layer_idx}.p{pass_idx}.messages", output, kind="bth")
        return output

    module.forward = wrapped
    dump._wrapped.append((module, "forward", original))


def _hook_clips(model: nn.Module, dump: ActivationDump) -> None:
    for name, module in model.named_modules():
        if isinstance(module, CLIP_TYPES):
            dump._handles.append(module.register_forward_hook(lambda mod, inputs, output, name=name: _count_clip(dump, name, mod, inputs)))
        elif isinstance(module, LOG_SCALE_TYPES) and "q_proj" in name and "log_scale" not in dump.tensors:
            dump._handles.append(module.register_forward_pre_hook(lambda mod, args: _save_log_scale(dump, mod, args)))


def _count_clip(dump: ActivationDump, name: str, module: nn.Module, inputs) -> None:
    hidden = inputs[0].detach().float()
    if hasattr(module, "scale"):
        hidden = hidden * module.scale.detach().float()
    norm = torch.linalg.vector_norm(hidden, dim=-1)
    clipped = norm + 1e-6 > module.threshold
    if ".k_proj" in name:
        kind = "k"
    elif ".q_proj" in name:
        kind = "q"
    elif "relative_position_bias" in name:
        kind = "r"
    else:
        kind = "other"
    bucket = dump.clip[kind]
    bucket["clipped"] += int(clipped.sum().item())
    bucket["total"] += int(clipped.numel())


def _save_log_scale(dump: ActivationDump, module: nn.Module, args) -> None:
    if "log_scale" in dump.tensors:
        return
    position_ids = args[1]
    scale = 1.0 + module.log_scaling_alpha * torch.log(
        ((position_ids + 1).float() / module.log_scaling_n_floor).clamp(min=1.0)
    )
    dump.save("log_scale", scale, kind="scale")
