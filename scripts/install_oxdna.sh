#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Build oxDNA at the commit seq2oxdna was validated with.
#
#   scripts/install_oxdna.sh [--cuda] [--prefix DIR] [--jobs N]
#
# CPU-only by default (enough for `seq2oxdna prepare` and the MC stage). `run` needs MD
# on CUDA, so on a machine with an NVIDIA GPU and the CUDA toolkit pass --cuda.
# The binary ends up at DIR/build/bin/oxDNA (default DIR: ./oxDNA); point seq2oxdna at it
# with --oxdna-bin, OXDNA_BIN, or by adding DIR/build/bin to PATH.
#
# Validated toolchain: Ubuntu 22.04, gcc 11.4, cmake 3.22, CUDA 12.3.
set -euo pipefail

OXDNA_REPO=${OXDNA_REPO:-https://github.com/lorenzo-rovigatti/oxDNA.git}
OXDNA_COMMIT=dbdc72658ff0325910c50586e16427c45a767f44   # v3.6.1-18-gdbdc7265
PREFIX=./oxDNA
CUDA=OFF
JOBS=4

while [ $# -gt 0 ]; do
    case "$1" in
        --cuda) CUDA=ON ;;
        --prefix) PREFIX=$2; shift ;;
        --jobs) JOBS=$2; shift ;;
        -h|--help) sed -n '3,12p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

for tool in git cmake make; do
    command -v "$tool" >/dev/null || { echo "ERROR: '$tool' is required" >&2; exit 1; }
done

if [ ! -d "$PREFIX/.git" ]; then
    git clone --quiet "$OXDNA_REPO" "$PREFIX"
fi
git -C "$PREFIX" fetch --quiet origin "$OXDNA_COMMIT" 2>/dev/null || true
git -C "$PREFIX" checkout --quiet "$OXDNA_COMMIT"

mkdir -p "$PREFIX/build"
# Physical path (-P): CMake records the folder it was configured in, and a later run through a
# symlinked path (as scripts/check_readme.py makes) would otherwise rewrite it to a temporary
# path that no longer exists next time.
cd -P "$PREFIX/build"
cmake .. -DCUDA="$CUDA"
make -j "$JOBS" oxDNA

echo
echo "Built $(pwd)/bin/oxDNA (commit ${OXDNA_COMMIT:0:8}, CUDA=$CUDA)"
echo "Use it with:  export OXDNA_BIN=$(pwd)/bin/oxDNA"
