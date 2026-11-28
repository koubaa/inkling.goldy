#!/usr/bin/env python3
"""Hugging Face transformers reference for SmolLM2-135M greedy decode.

Loads the pinned checkpoint from `assets/` (run `tools/fetch_assets.py` first)
as `LlamaForCausalLM` in float32. BF16 weights widen to float32 exactly, so the
reference and the Goldy model multiply the same values. For each prompt it
decodes greedily, then runs one teacher-forced forward over the prompt and the
continuation and keeps the top logits at every position.

    python tools/ref/reference.py            # rewrite tests/data/reference.json
    python tools/ref/reference.py --check    # fail if the committed file differs

`--check` compares token ids exactly and logits to 1e-4, since CPU kernels may
round differently across machines.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM

ROOT = pathlib.Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"
FIXTURE = ROOT / "tests" / "data" / "reference.json"

sys.path.insert(0, str(ROOT / "tools"))
from fetch_assets import REPO, REVISION  # noqa: E402

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time, there was a little",
]
NEW_TOKENS = 12
TOP_K = 8
CHECK_ATOL = 1e-4


def generate() -> dict:
    torch.manual_seed(0)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    tokenizer = AutoTokenizer.from_pretrained(ASSETS)
    model = LlamaForCausalLM.from_pretrained(
        ASSETS, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    cfg = model.config
    cases = []
    with torch.no_grad():
        for text in PROMPTS:
            prompt = tokenizer(text, add_special_tokens=True)["input_ids"]
            ids = list(prompt)
            greedy = []
            for _ in range(NEW_TOKENS):
                logits = model(torch.tensor([ids])).logits[0, -1]
                token = int(torch.argmax(logits))
                greedy.append(token)
                ids.append(token)
            # Logits at every position the decoder emits: the prompt tokens and every
            # generated token except the last.
            logits = model(torch.tensor([ids[:-1]])).logits[0]
            positions = []
            for row in logits:
                top = torch.topk(row, TOP_K)
                positions.append(
                    {
                        "top_ids": [int(i) for i in top.indices],
                        "top_logits": [float(v) for v in top.values],
                    }
                )
            for step, token in enumerate(greedy):
                top1 = positions[len(prompt) - 1 + step]["top_ids"][0]
                if top1 != token:
                    raise SystemExit(
                        f"{text!r}: teacher-forced argmax {top1} != greedy {token} at step {step}"
                    )
            cases.append(
                {
                    "prompt": text,
                    "prompt_ids": prompt,
                    "greedy_ids": greedy,
                    "greedy_text": tokenizer.decode(greedy),
                    "positions": positions,
                }
            )
    return {
        "generator": "tools/ref/reference.py",
        "repo": REPO,
        "revision": REVISION,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "dtype": "float32",
        "config": {
            "hidden_size": cfg.hidden_size,
            "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": cfg.num_key_value_heads,
            "num_hidden_layers": cfg.num_hidden_layers,
            "rope_theta": float(
                getattr(cfg, "rope_theta", None)
                or cfg.rope_parameters["rope_theta"]
            ),
            "tie_word_embeddings": cfg.tie_word_embeddings,
        },
        "top_k": TOP_K,
        "new_tokens": NEW_TOKENS,
        "cases": cases,
    }


def check(fresh: dict, committed: dict) -> list[str]:
    problems = []
    if len(fresh["cases"]) != len(committed["cases"]):
        return ["case count differs"]
    for a, b in zip(fresh["cases"], committed["cases"]):
        name = a["prompt"]
        for key in ("prompt", "prompt_ids", "greedy_ids"):
            if a[key] != b[key]:
                problems.append(f"{name!r}: {key} {a[key]} != {b[key]}")
        for pos, (pa, pb) in enumerate(zip(a["positions"], b["positions"])):
            if pa["top_ids"] != pb["top_ids"]:
                problems.append(f"{name!r} position {pos}: top ids differ")
                continue
            err = max(abs(x - y) for x, y in zip(pa["top_logits"], pb["top_logits"]))
            if err > CHECK_ATOL:
                problems.append(f"{name!r} position {pos}: logits differ by {err:.2e}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    fresh = generate()
    for case in fresh["cases"]:
        margins = [
            p["top_logits"][0] - p["top_logits"][1]
            for p in case["positions"][len(case["prompt_ids"]) - 1 :]
        ]
        print(
            f"{case['prompt']!r} -> {case['greedy_text']!r} "
            f"(smallest greedy margin {min(margins):.4f})"
        )
    if args.check:
        problems = check(fresh, json.loads(FIXTURE.read_text()))
        if problems:
            raise SystemExit("reference drifted:\n" + "\n".join(problems))
        print(f"{FIXTURE.relative_to(ROOT)} matches")
        return
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(fresh, indent=1) + "\n")
    print(f"wrote {FIXTURE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
