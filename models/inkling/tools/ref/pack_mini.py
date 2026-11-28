#!/usr/bin/env python3
"""Pack inkling2_loop/mini logits into the compact golden the Rust test reads.

Run after `python tools/ref/generate.py --only inkling2_loop/mini`.
Writes tests/data/mini_model.safetensors and tests/data/mini_forward.bin.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from configs import SPECS  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

SPEC = SPECS["inkling2_loop/mini"]
SRC = REPO / "artifacts" / "ref" / "inkling2_loop" / "mini"
DST = REPO / "tests" / "data"


def main() -> None:
    acts = load_file(SRC / "activations.safetensors")
    logits = acts["logits"].float().reshape(-1).contiguous()
    ids = json.loads((SRC / "input_ids.json").read_text())["input_ids"]
    hf = SPEC.hf
    header = [
        0x494E4B4D,
        1,
        int(hf["hidden_size"]),
        int(hf["num_hidden_layers"]),
        int(hf["n_layers_per_attn_res_block"]),
        int(hf["loop_layer_index_start"]),
        int(hf["loop_layer_index_end"]),
        int(hf["loop_num_passes"]),
        int(hf["num_attention_heads"]),
        int(hf["num_key_value_heads"]),
        int(hf["head_dim"]),
        int(hf["d_rel"]),
        int(hf["rel_extent"]),
        int(hf["sliding_window_size"]),
        int(hf["conv_kernel_size"]),
        int(hf["n_routed_experts"]),
        int(hf["num_experts_per_tok"]),
        int(hf["moe_intermediate_size"]),
        int(hf["loop_gate_rank"]),
        int(hf["unpadded_vocab_size"]),
        int(SPEC.seq_len),
    ]
    floats = [
        float(hf["rms_norm_eps"]),
        float(hf["qk_l2_clip_threshold"]),
        float(hf["rel_l2_clip_threshold"]),
        float(hf["route_scale"]),
        float(hf["mlp_gate_softcap"]),
        float(hf["mlp_up_softcap"]),
        float(hf["final_logit_softcapping"]),
        float(hf["log_scaling_alpha"]),
        float(hf["log_scaling_n_floor"]),
    ]
    if len(ids) != SPEC.seq_len or logits.numel() != SPEC.seq_len * int(hf["unpadded_vocab_size"]):
        raise SystemExit(f"logits {tuple(logits.shape)} tokens {len(ids)}")
    DST.mkdir(parents=True, exist_ok=True)
    blob = struct.pack(f"<{len(header)}I", *header)
    blob += struct.pack(f"<{len(floats)}f", *floats)
    blob += struct.pack(f"<{len(ids)}I", *[int(token) for token in ids])
    blob += logits.numpy().astype("<f4", copy=False).tobytes()
    (DST / "mini_forward.bin").write_bytes(blob)
    (DST / "mini_model.safetensors").write_bytes((SRC / "model.transformers.safetensors").read_bytes())
    print(f"wrote {DST / 'mini_forward.bin'} ({len(blob)} B) and the state dict")


if __name__ == "__main__":
    main()
