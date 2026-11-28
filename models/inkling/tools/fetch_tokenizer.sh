#!/bin/sh
# Pinned Inkling tokenizer: tiny-random/inkling-2 @ 0a57c72e4e5b68aafe1d983dbb62980344a5fcd7.
set -eu
root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
dest=$root/assets/tokenizer.json
sum=7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6
hash_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}
if [ -f "$dest" ] && [ "$(hash_file "$dest")" = "$sum" ]; then
  exit 0
fi
mkdir -p "$root/assets"
curl -fsSL -o "$dest" \
  "https://huggingface.co/tiny-random/inkling-2/resolve/0a57c72e4e5b68aafe1d983dbb62980344a5fcd7/tokenizer.json"
got=$(hash_file "$dest")
if [ "$got" != "$sum" ]; then
  echo "tokenizer hash mismatch: $got != $sum" >&2
  exit 1
fi
