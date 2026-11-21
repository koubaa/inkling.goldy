#!/usr/bin/env python3
"""Generate seeded FP32 reference weights and activation dumps.

Run from the repository root, after installing tools/ref/requirements.txt:

    python tools/ref/generate.py
    python tools/ref/generate.py --only inkling2/micro inkling2_loop/micro

Artifacts land in artifacts/ref/<family>/<size>/ and are gitignored.
The process exits non-zero if cached decode disagrees with the full forward
by more than 1e-6, if Q/K/R clipping never fires, or if the sliding-window
mask does not match the documented boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

# One thread, set before torch is imported, so the dumps do not depend on a
# parallel reduction order.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import torch

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
# MKLDNN picks a different kernel for a full sequence than for a single token,
# which moves cached decode off the full forward by more than 1e-6.
torch.backends.mkldnn.enabled = False

REPO_ROOT = Path(__file__).resolve().parents[2]
REF_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REF_DIR))

from capture import SITE_DOCS, ActivationDump, install_hooks  # noqa: E402
from configs import SPECS, TINY_PROMPT, TOKENIZER_REPO, Spec  # noqa: E402

PARITY_ATOL = 1e-6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Config names to generate (default: all). Example: inkling2/micro",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "artifacts" / "ref",
        help="Output directory (default: artifacts/ref)",
    )
    args = parser.parse_args()
    names = list(SPECS) if not args.only else args.only
    unknown = [name for name in names if name not in SPECS]
    if unknown:
        raise SystemExit(f"unknown config {unknown}; choose from {list(SPECS)}")

    pin = verify_pin()
    args.out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name in names:
        print(f"\n=== {name} ===", flush=True)
        summaries.append(generate_one(SPECS[name], args.out, pin))
    summary_path = args.out / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"\nWrote {summary_path}")
    for item in summaries:
        parity = item["parity"]
        print(
            f"{item['name']}: weights {item['bytes']['model_safetensors']} B, "
            f"activations {item['bytes']['activations_safetensors']} B, "
            f"cached-vs-full max_abs {parity['max_abs']:.3e} "
            f"({parity['status']})"
        )


def verify_pin() -> dict:
    pin = json.loads((REF_DIR / "pin.json").read_text())
    import transformers
    from transformers.models.inkling2 import modeling_inkling2
    from transformers.models.inkling2_loop import modeling_inkling2_loop

    version = transformers.__version__
    if not version.startswith(pin["transformers_version"]):
        raise SystemExit(
            f"transformers {version} is installed; pin.json requires {pin['transformers_version']} "
            f"(commit {pin['transformers_commit']})"
        )
    for label, module, expected in (
        ("modeling_inkling2", modeling_inkling2, pin["modeling_inkling2_sha256"]),
        ("modeling_inkling2_loop", modeling_inkling2_loop, pin["modeling_inkling2_loop_sha256"]),
    ):
        digest = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        if digest != expected:
            raise SystemExit(
                f"{label} sha256 {digest} does not match pin.json ({expected}). "
                f"The semantics notes are line-pinned to commit {pin['transformers_commit']}."
            )
    return pin


def generate_one(spec: Spec, out_root: Path, pin: dict) -> dict:
    model, config = build_model(spec)
    rerandomize(model, seed=spec.weight_seed, std=spec.weight_std, clip_scale_multiplier=spec.clip_scale_multiplier)
    assert_not_left_at_hf_init(model)

    out_dir = out_root / spec.family / spec.size
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    check_config_roundtrip(spec, out_dir)

    input_ids, token_meta = make_tokens(spec)
    (out_dir / "input_ids.json").write_text(
        json.dumps({"input_ids": input_ids[0].tolist(), **token_meta}, indent=2) + "\n"
    )

    dump = ActivationDump(spec.seq_len)
    start = spec.hf.get("loop_layer_index_start")
    end = spec.hf.get("loop_layer_index_end")
    num_loop = 0 if start is None else int(end) - int(start)
    install_hooks(model, dump, loop_start=start, num_loop_layers=num_loop)

    with torch.inference_mode():
        full = model(input_ids=input_ids, use_cache=False)
    dump.save("logits", full.logits, kind="bth")
    check_sites(spec, dump)
    check_masks_and_bias(spec, dump)
    check_log_scale(spec, dump)
    check_clipping(dump)
    if num_loop:
        check_message_gates_feed_norms(dump, start, end)

    dump.close()
    with torch.inference_mode():
        decoded = cached_decode(model, input_ids)
    parity = compare_logits(full.logits[0], decoded)

    from safetensors.torch import save_file

    save_file(dump.tensors, str(out_dir / "activations.safetensors"))
    manifest = build_manifest(spec, config, model, dump, token_meta, input_ids[0].tolist(), parity, pin)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    sizes = {
        "model_safetensors": (out_dir / "model.safetensors").stat().st_size,
        "activations_safetensors": (out_dir / "activations.safetensors").stat().st_size,
        "config_json": (out_dir / "config.json").stat().st_size,
        "manifest_json": (out_dir / "manifest.json").stat().st_size,
    }
    print(
        f"saved {out_dir}  model={sizes['model_safetensors']} B  "
        f"activations={sizes['activations_safetensors']} B  "
        f"parity max_abs={parity['max_abs']:.3e}"
    )
    return {"name": spec.name, "dir": str(out_dir.relative_to(REPO_ROOT)), "bytes": sizes, "parity": parity, "clip": manifest["clip"]}


def build_model(spec: Spec):
    if spec.family == "inkling2":
        from transformers import Inkling2ForCausalLM, Inkling2TextConfig

        config_cls, model_cls = Inkling2TextConfig, Inkling2ForCausalLM
    elif spec.family == "inkling2_loop":
        from transformers import Inkling2LoopForCausalLM, Inkling2LoopTextConfig

        config_cls, model_cls = Inkling2LoopTextConfig, Inkling2LoopForCausalLM
    else:
        raise ValueError(spec.family)

    torch.manual_seed(spec.weight_seed)
    config = config_cls(**spec.hf)
    if hasattr(config, "validate_architecture"):
        config.validate_architecture()
    config._attn_implementation = "eager"
    expected_types = [
        "hybrid" if (i + 1) % config.n_layers_per_attn_res_block == 0 else "hybrid_sliding"
        for i in range(config.num_hidden_layers)
    ]
    if list(config.layer_types) != expected_types:
        raise RuntimeError(f"layer_types {config.layer_types} != {expected_types}")
    if config.sliding_window != spec.sliding_window_size:
        raise RuntimeError("sliding_window alias does not read sliding_window_size")
    if config.num_local_experts != spec.hf["n_routed_experts"]:
        raise RuntimeError("num_local_experts alias does not read n_routed_experts")
    if config.routed_scaling_factor != spec.hf["route_scale"]:
        raise RuntimeError("routed_scaling_factor alias does not read route_scale")
    if config.sconv_kernel_size != spec.hf["conv_kernel_size"]:
        raise RuntimeError("sconv_kernel_size alias does not read conv_kernel_size")

    model = model_cls(config)
    model = model.float().eval()
    if model.config._attn_implementation != "eager":
        raise RuntimeError(f"attention implementation is {model.config._attn_implementation}, want eager")
    for parameter in model.parameters():
        if parameter.dtype != torch.float32:
            raise RuntimeError(f"parameter dtype {parameter.dtype}")
    return model, config


def rerandomize(model, *, seed: int, std: float, clip_scale_multiplier: float) -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    for parameter in model.parameters():
        parameter.data.normal_(mean=0.0, std=std, generator=generator)
    for buffer in model.buffers():
        if buffer.is_floating_point() and buffer.numel() > 0:
            buffer.normal_(mean=0.0, std=std, generator=generator)
    scaled = 0
    for name, parameter in model.named_parameters():
        if name.endswith(".scale"):
            parameter.data.mul_(clip_scale_multiplier)
            scaled += 1
    if scaled == 0:
        raise RuntimeError("no clip scale parameters were found")


def assert_not_left_at_hf_init(model) -> None:
    """HF init zeros these. Re-randomization has to move every one of them."""
    interesting = []
    for name, parameter in model.named_parameters():
        if (
            name.endswith("static_weight")
            or name.endswith(".query")
            or name.endswith(".scale")
            or name.endswith("embed_norm.weight")
        ):
            interesting.append(name)
            if parameter.numel() > 1 and torch.count_nonzero(parameter) == 0:
                raise RuntimeError(f"{name} is still all zeros after re-randomization")
            if name.endswith(".scale") and torch.allclose(parameter, torch.ones_like(parameter)):
                raise RuntimeError(f"{name} is still all ones")
    for name, buffer in model.named_buffers():
        if name.endswith("e_score_correction_bias"):
            interesting.append(name)
            if torch.count_nonzero(buffer) == 0:
                raise RuntimeError(f"{name} is still all zeros")
    if not any(name.endswith("static_weight") for name in interesting):
        raise RuntimeError("model has no dynamic-conv static_weight; full-attention layers are missing")
    if not any(name.endswith("e_score_correction_bias") for name in interesting):
        raise RuntimeError("router correction bias buffer is missing")


def check_config_roundtrip(spec: Spec, out_dir: Path) -> None:
    if spec.family == "inkling2":
        from transformers import Inkling2TextConfig as config_cls
    else:
        from transformers import Inkling2LoopTextConfig as config_cls

    loaded = config_cls.from_pretrained(out_dir)
    for key, expected in spec.hf.items():
        got = getattr(loaded, key)
        if isinstance(expected, float):
            if not math.isclose(float(got), expected, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(f"config {key} reloaded as {got}, expected {expected}")
        elif got != expected:
            raise RuntimeError(f"config {key} reloaded as {got}, expected {expected}")


def make_tokens(spec: Spec) -> tuple[torch.Tensor, dict]:
    if spec.use_tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO)
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": TINY_PROMPT}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )
        raw = encoded["input_ids"]
        if hasattr(raw, "tolist"):
            raw = raw.tolist()
        if raw and isinstance(raw[0], list):
            raw = raw[0]
        raw = [int(token) for token in raw]
        vocab = int(spec.hf["vocab_size"])
        if any(token < 0 or token >= vocab for token in raw):
            raise RuntimeError("tokenizer produced an id outside vocab_size")
        tiled: list[int] = []
        while len(tiled) < spec.seq_len:
            tiled.extend(raw)
        ids = tiled[: spec.seq_len]
        meta = {
            "tokenizer_repo": TOKENIZER_REPO,
            "prompt": TINY_PROMPT,
            "chat_template": True,
            "add_generation_prompt": True,
            "untiled_length": len(raw),
            "tiled_to": spec.seq_len,
        }
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(spec.token_seed)
        ids = torch.randint(
            0,
            int(spec.hf["unpadded_vocab_size"]),
            (spec.seq_len,),
            generator=generator,
        ).tolist()
        meta = {
            "tokenizer_repo": None,
            "prompt": None,
            "chat_template": False,
            "token_seed": spec.token_seed,
            "id_range": [0, int(spec.hf["unpadded_vocab_size"])],
        }
    return torch.tensor([ids], dtype=torch.long), meta


def cached_decode(model, input_ids: torch.Tensor) -> torch.Tensor:
    pieces = []
    cache = None
    for step in range(input_ids.shape[1]):
        output = model(input_ids=input_ids[:, step : step + 1], cache=cache, use_cache=True)
        cache = output.cache
        if output.logits.shape[1] != 1:
            raise RuntimeError(f"decode step {step} returned logits {tuple(output.logits.shape)}")
        pieces.append(output.logits[0, 0].float().cpu())
    return torch.stack(pieces, dim=0)


def compare_logits(full: torch.Tensor, decoded: torch.Tensor) -> dict:
    full = full.float().cpu()
    diff = (decoded - full).abs()
    max_abs = float(diff.max().item())
    # Where the full logit is away from zero, also report the worst relative error.
    denom = full.abs().clamp(min=1e-6)
    max_rel = float((diff / denom).max().item())
    position = int(diff.view(diff.shape[0], -1).max(dim=-1).values.argmax().item())
    status = "ok" if max_abs <= PARITY_ATOL else "FAIL"
    report = {
        "atol": PARITY_ATOL,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "worst_position": position,
        "status": status,
    }
    if status != "ok":
        raise SystemExit(
            f"cached decode does not match the full forward: max_abs {max_abs:.3e} > {PARITY_ATOL:.0e} "
            f"at position {position}"
        )
    return report


def check_sites(spec: Spec, dump: ActivationDump) -> None:
    required = ["embed", "final_depth_mix", "final_norm", "logits_pre_softcap", "logits", "log_scale"]
    required += ["mask.sliding", "mask.full", "rel_distance.sliding", "rel_distance.full"]
    n_layers = int(spec.hf["num_hidden_layers"])
    start = spec.hf.get("loop_layer_index_start")
    end = spec.hf.get("loop_layer_index_end")
    passes = int(spec.hf.get("loop_num_passes", 1))
    for index in range(n_layers):
        is_full = (index + 1) % int(spec.hf["n_layers_per_attn_res_block"]) == 0
        is_loop = start is not None and int(start) <= index < int(end)
        n_pass = passes if is_loop else 1
        for pass_idx in range(n_pass):
            prefix = f"L{index}.p{pass_idx}." if is_loop else f"L{index}."
            required.append(prefix + "layer_out")
            required.append(prefix + "mlp_depth")
            required.append(prefix + "mlp_in")
            required.append(prefix + "mlp_norm")
            required.append(prefix + "mlp_out")
            required.append(prefix + "router_logits")
            required.append(prefix + "expert_ids")
            required.append(prefix + "expert_weights")
            required.append(prefix + "q")
            required.append(prefix + "k")
            required.append(prefix + "v")
            required.append(prefix + "rel_bias")
            required.append(prefix + "sink")
            required.append(prefix + "attn_pre_o")
            required.append(prefix + "attn_out")
            required.append(prefix + "post_attn_conv")
            required.append(prefix + "post_mlp_conv")
            if is_full:
                for stem in ("attn_in", "attn_norm", "attn_depth"):
                    for suffix in ("q", "k", "v"):
                        # Layer 0 has no attention_input module. None of our
                        # configs put a full layer there; kept for the pattern.
                        if stem == "attn_depth" and index == 0:
                            continue
                        required.append(prefix + f"{stem}_{suffix}")
            elif is_loop and pass_idx > 0:
                # The shared sliding norm runs once per stream on later passes.
                for suffix in ("q", "k", "v"):
                    required.append(prefix + f"attn_in_{suffix}")
                    required.append(prefix + f"attn_norm_{suffix}")
                if index != 0:
                    required.append(prefix + "attn_depth")
            else:
                required.append(prefix + "attn_in")
                required.append(prefix + "attn_norm")
                if index != 0:
                    required.append(prefix + "attn_depth")
            if is_loop and pass_idx > 0:
                required.append(prefix + "messages")
                required += [prefix + name for name in ("msg_q", "msg_k", "msg_v", "msg_mlp")]
    missing = [key for key in required if key not in dump.tensors]
    if missing:
        raise RuntimeError(f"missing {len(missing)} activation sites, first ones: {missing[:12]}")


def check_masks_and_bias(spec: Spec, dump: ActivationDump) -> None:
    window = spec.sliding_window_size
    sliding = dump.tensors["mask.sliding"]
    full = dump.tensors["mask.full"]
    length = spec.seq_len
    if sliding.shape != (length, length) or full.shape != (length, length):
        raise RuntimeError(f"mask shapes sliding {tuple(sliding.shape)} full {tuple(full.shape)}")
    sliding_on = sliding > -1e20
    full_on = full > -1e20
    for query in range(length):
        for key in range(length):
            expect_full = key <= query
            expect_sliding = (key > query - window) and (key <= query)
            if bool(full_on[query, key]) != expect_full or bool(sliding_on[query, key]) != expect_sliding:
                raise RuntimeError(
                    f"mask mismatch at query {query} key {key}: "
                    f"full {bool(full_on[query, key])} (want {expect_full}), "
                    f"sliding {bool(sliding_on[query, key])} (want {expect_sliding})"
                )
    distance = dump.tensors["rel_distance.full"]
    if not torch.equal(distance, dump.tensors["rel_distance.sliding"]):
        raise RuntimeError("full and sliding relative distances differ on an uncached forward")
    positions = torch.arange(length)
    expected = positions[:, None] - positions[None, :]
    if not torch.equal(distance.long(), expected):
        raise RuntimeError("relative distance is not query_position - key_position")

    block = int(spec.hf["n_layers_per_attn_res_block"])
    first_full = block - 1
    _assert_bias_support(dump, f"L0.rel_bias", distance, window)
    _assert_bias_support(dump, f"L{first_full}.rel_bias", distance, spec.rel_extent)


def _assert_bias_support(dump: ActivationDump, key: str, distance: torch.Tensor, extent: int) -> None:
    bias = dump.tensors[key]
    outside = (distance < 0) | (distance >= extent)
    outside = outside.unsqueeze(1)
    leaked = bias.masked_fill(~outside, 0)
    if torch.count_nonzero(leaked) != 0:
        raise RuntimeError(f"{key} is nonzero outside rel extent {extent}")
    inside = ~outside
    if torch.count_nonzero(bias.masked_fill(~inside, 0)) == 0:
        raise RuntimeError(f"{key} is entirely zero inside rel extent {extent}; the bias bank is dead")


def check_log_scale(spec: Spec, dump: ActivationDump) -> None:
    positions = torch.arange(spec.seq_len, dtype=torch.float32)
    floor = float(spec.log_scaling_n_floor)
    alpha = float(spec.hf["log_scaling_alpha"])
    expected = 1.0 + alpha * torch.log(((positions + 1) / floor).clamp(min=1.0))
    actual = dump.tensors["log_scale"]
    if not torch.allclose(actual, expected, atol=1e-6, rtol=0.0):
        raise RuntimeError("captured log scale does not match 1 + alpha * log(clamp((pos+1)/floor, 1))")
    # The sequence has to contain both an unscaled prefix and a scaled tail.
    if not torch.any(expected == 1) or not torch.any(expected > 1):
        raise RuntimeError(f"sequence does not cross log_scaling_n_floor={floor}")


def check_clipping(dump: ActivationDump) -> None:
    for kind in ("q", "k", "r"):
        stats = dump.clip.get(kind)
        if stats is None or stats["clipped"] <= 0:
            raise RuntimeError(f"{kind} L2 clipping never fired ({dict(dump.clip)})")


def check_message_gates_feed_norms(dump: ActivationDump, start: int, end: int) -> None:
    """On passes after the first, each message-gate output is the matching norm input.

    Sliding loop layers have one shared norm, called on the query, key and value
    gate outputs in that order. Full loop layers have three norms. Both also
    gate the MLP stream.
    """
    block = 6
    for index in range(int(start), int(end)):
        prefix = f"L{index}.p1."
        pairs = [
            ("msg_q", "attn_in_q"),
            ("msg_k", "attn_in_k"),
            ("msg_v", "attn_in_v"),
            ("msg_mlp", "mlp_in"),
        ]
        if (index + 1) % block != 0 and f"{prefix}attn_in_q" not in dump.tensors:
            pairs[0] = ("msg_q", "attn_in")
        for left, right in pairs:
            diff = (dump.tensors[prefix + left] - dump.tensors[prefix + right]).abs().max().item()
            if diff != 0.0:
                raise RuntimeError(f"{prefix}{left} and {prefix}{right} differ by {diff}")


def build_manifest(spec, config, model, dump, token_meta, input_ids, parity, pin) -> dict:
    layers = []
    for index, layer in enumerate(model.model.layers):
        layers.append(
            {
                "index": index,
                "class": type(layer).__name__,
                "attention_type": layer.attention_type,
                "layer_type": config.layer_types[index],
                "is_block_boundary": bool(layer.is_block_boundary),
                "is_loop": bool(getattr(layer, "_ref_is_loop", False)),
                "dynamic_kv_conv": layer.attention_type == "full_attention",
            }
        )
    sites = {}
    for key, tensor in dump.tensors.items():
        doc_key = key if key in SITE_DOCS else key.split(".")[-1]
        sites[key] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "description": SITE_DOCS.get(doc_key, ""),
        }
    n_params = sum(parameter.numel() for parameter in model.parameters())
    return {
        "name": spec.name,
        "family": spec.family,
        "size": spec.size,
        "notes": spec.notes,
        "transformers_commit": pin["transformers_commit"],
        "transformers_version": pin["transformers_version"],
        "dtype": "float32",
        "attention": "eager",
        "threads": 1,
        "weight_seed": spec.weight_seed,
        "weight_std": spec.weight_std,
        "clip_scale_multiplier": spec.clip_scale_multiplier,
        "seq_len": spec.seq_len,
        "tokens": token_meta,
        "input_ids": input_ids,
        "stock_windows": spec.stock_windows,
        "windows_used": {
            "sliding_window_size": spec.sliding_window_size,
            "rel_extent": spec.rel_extent,
            "log_scaling_n_floor": spec.log_scaling_n_floor,
            "log_scaling_alpha": spec.hf["log_scaling_alpha"],
        },
        "parameter_count": n_params,
        "layers": layers,
        "clip": {kind: dict(stats) for kind, stats in dump.clip.items()},
        "parity": parity,
        "sites": sites,
        "model_class": type(model).__name__,
        "config_model_type": config.model_type,
    }


if __name__ == "__main__":
    main()
