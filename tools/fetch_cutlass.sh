#!/usr/bin/env bash
set -euo pipefail
if (( $# != 1 )); then
  echo 'Usage: bash tools/fetch_cutlass.sh DESTINATION' >&2
  exit 2
fi
revision=1afc6d355b8cb992398f17beb4425702f1d57ddd
if [[ ! -e "$1" ]]; then
  git clone --branch v4.3.6 --depth 1 https://github.com/NVIDIA/cutlass.git "$1"
fi
actual=$(git -C "$1" rev-parse HEAD)
if [[ "$actual" != "$revision" ]]; then
  echo "Expected CUTLASS v4.3.6 commit $revision; found $actual" >&2
  exit 1
fi
if [[ -n $(git -C "$1" status --porcelain --untracked-files=no) ]]; then
  echo 'CUTLASS tracked files are modified; use a clean pinned checkout.' >&2
  exit 1
fi
printf 'Pinned CUTLASS: %s\n' "$actual"
