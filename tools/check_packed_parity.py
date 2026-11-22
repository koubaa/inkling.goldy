#!/usr/bin/env python3
"""Opt-in parity for a few real Inkling-2-Mini tensors.

Not used in CI. Range-fetches expert 0 of layers 0, 10, and 19 (`w13`, NVFP4)
plus the first four rows of layer 0 `wq_du`, dequantizes with the transformers
5.24.0 ops, and writes the f32 matvecs the packed Rust path checks.

    INKLING_REAL_PACK=/tmp/realpack python tools/check_packed_parity.py

The directory must already contain `l{0,10,19}/{codes,scales,amax,denom}.bin`
and `wq_rows.bin` (see the offsets in SEMANTICS / the safetensors header).
File offsets are `8 + header_len + data_offset`.
"""

import os
import struct
import sys

import torch
from transformers.core_model_loading import Inkling2LoopDequantizeExperts, Interleave


def write_f32(path, values):
    flat = values.detach().float().cpu().contiguous()
    with open(path, "wb") as handle:
        handle.write(struct.pack(f"<{flat.numel()}f", *flat.tolist()))


def pattern(n):
    return torch.tensor([((i % 17) - 8) / 8 for i in range(n)], dtype=torch.float32)


class FloatParam:
    def get_parameter(self, _name):
        return torch.zeros(1, dtype=torch.float32)


def blob_u8(blob):
    buf = bytearray(blob)
    return torch.frombuffer(buf, dtype=torch.uint8).clone()


def dequant(codes, scales, amax, denom, dtype_model):
    weight = blob_u8(codes).view(1, 2304, 1152)
    scale = blob_u8(scales).view(torch.float8_e4m3fn).view(1, 2304, 144)
    amax_t = torch.tensor([struct.unpack("<f", amax)[0]])
    denom_t = torch.tensor([struct.unpack("<f", denom)[0]])
    op = Inkling2LoopDequantizeExperts()
    kwargs = {}
    if dtype_model is not None:
        kwargs = {"model": dtype_model, "full_layer_name": "w"}
    out = op.convert(
        {"weight": weight, "scale": scale, "amax": amax_t, "denominator": denom_t},
        ["weight", "scale", "amax", "denominator"],
        ["out"],
        **kwargs,
    )["out"]
    return Interleave(dim=1).convert({"w": out}, ["w"], ["out"])["out"]


def main():
    root = os.environ.get("INKLING_REAL_PACK", "/tmp/realpack")
    x = pattern(2304)
    rows = []
    for layer in (0, 10, 19):
        folder = os.path.join(root, f"l{layer}")
        codes = open(os.path.join(folder, "codes.bin"), "rb").read()
        scales = open(os.path.join(folder, "scales.bin"), "rb").read()
        amax = open(os.path.join(folder, "amax.bin"), "rb").read()
        denom = open(os.path.join(folder, "denom.bin"), "rb").read()
        dense_f32 = dequant(codes, scales, amax, denom, FloatParam())
        dense_bf16 = dequant(codes, scales, amax, denom, None)
        y = torch.matmul(dense_f32[0].float(), x)
        y_bf = torch.matmul(dense_bf16[0].float(), x)
        gap = (y - y_bf).abs().max().item()
        write_f32(os.path.join(folder, "expected.bin"), y)
        rows.append((layer, y.abs().max().item(), gap))
        print(f"layer {layer} expert0 |y|max {rows[-1][1]:.6g} f32-vs-bf16-store {gap:.6g}")

    wq = open(os.path.join(root, "wq_rows.bin"), "rb").read()
    bf = blob_u8(wq).view(torch.bfloat16).view(4, 2304)
    y = torch.matmul(bf.float(), x)
    write_f32(os.path.join(root, "wq_expected.bin"), y)
    print(f"wq rows |y|max {y.abs().max().item():.6g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
