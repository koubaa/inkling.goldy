#!/usr/bin/env python3
"""Fetch and hash-check SmolLM2-135M at a pinned Hugging Face revision."""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"

REPO = "HuggingFaceTB/SmolLM2-135M"
REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"

FILES: dict[str, dict[str, object]] = {
    "config.json": {
        "sha256": "1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843",
        "bytes": 704,
    },
    "tokenizer.json": {
        "sha256": "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
        "bytes": 2_104_556,
    },
    "model.safetensors": {
        "sha256": "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1",
        "bytes": 269_060_552,
    },
}


def url(name: str) -> str:
    return f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(name: str, path: pathlib.Path) -> str | None:
    """Return why `path` does not match its pin, or None when it does."""
    spec = FILES[name]
    size = path.stat().st_size
    if size != spec["bytes"]:
        return f"{size} bytes, expected {spec['bytes']}"
    got = sha256(path)
    if got != spec["sha256"]:
        return f"sha256 {got}, expected {spec['sha256']}"
    return None


def fetch(name: str) -> None:
    dest = ASSETS / name
    if dest.exists() and check(name, dest) is None:
        print(f"ok {name}")
        return
    ASSETS.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url(name)}")
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    urllib.request.urlretrieve(url(name), tmp)
    problem = check(name, tmp)
    if problem is not None:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"{name}: {problem}")
    tmp.replace(dest)
    print(f"ok {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=sorted(FILES), help="fetch only these files")
    args = parser.parse_args()
    for name in args.only or FILES:
        fetch(name)


if __name__ == "__main__":
    main()
