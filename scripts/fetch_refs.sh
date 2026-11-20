#!/bin/sh
# Fetch the third-party sources that the tests run against.  They are not redistributed in this
# repository; each comes with its own license.
#
#   sh scripts/fetch_refs.sh
#
# Creates, under refs/:
#   cutlass/            NVIDIA/cutlass at the commit used in the paper (sparse checkout:
#                       the CuTe headers, the bundled Python module, the CuTe documentation)
#   repos/nvlabs-cute/  NVlabs/CuTe (the reference implementation "PyCuTe")
#   cuda-headers/       NVIDIA's redistributable CUDA header archives (headers only; nothing
#                       from them is executed -- they let the CuTe headers compile on a host)
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REFS="$ROOT/refs"
mkdir -p "$REFS/repos" "$REFS/cuda-headers/dl"

CUTLASS_COMMIT=af951046cd9cac9301d8e3e90863fd8d7edc9fa9
PYCUTE_COMMIT=56cc3d0c0b3cc906dd8771b98181524b82781049

if [ ! -d "$REFS/cutlass/.git" ]; then
  git clone --filter=blob:none --no-checkout https://github.com/NVIDIA/cutlass.git "$REFS/cutlass"
  git -C "$REFS/cutlass" sparse-checkout set include/cute include/cutlass media/docs/cpp/cute \
      python/pycute test/unit/cute/core
  git -C "$REFS/cutlass" checkout --quiet "$CUTLASS_COMMIT"
fi

if [ ! -d "$REFS/repos/nvlabs-cute/.git" ]; then
  git clone --quiet https://github.com/NVlabs/CuTe.git "$REFS/repos/nvlabs-cute"
  git -C "$REFS/repos/nvlabs-cute" checkout --quiet "$PYCUTE_COMMIT"
fi

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

# CUDA 13.5.0 redistributable archives (linux-x86_64; only the headers are used)
BASE=https://developer.download.nvidia.com/compute/cuda/redist
fetch() {  # <name> <relative path> <sha256>
  out="$REFS/cuda-headers/dl/$1.tar.xz"
  [ -f "$out" ] || curl -sSfL "$BASE/$2" -o "$out"
  got="$(sha256 "$out")"
  if [ "$got" != "$3" ]; then
    echo "checksum mismatch for $1: $got" >&2
    exit 1
  fi
  rm -rf "$REFS/cuda-headers/$1"
  mkdir -p "$REFS/cuda-headers/$1"
  tar -xJf "$out" -C "$REFS/cuda-headers/$1" --strip-components=1
}
fetch cccl        cccl/linux-x86_64/cccl-linux-x86_64-13.3.5.0.6-archive.tar.xz \
      5139f641e72df6b0c64ee9e68cabaeafa9a5e36a4b1f7d370a1dc0dac62dd72f
fetch cuda_crt    cuda_crt/linux-x86_64/cuda_crt-linux-x86_64-13.5.37-archive.tar.xz \
      0df5c0ca27392152f370751c82b27ff1ab67661e4810cfc7248e2e03479c9ae0
fetch cuda_cudart cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-13.5.37-archive.tar.xz \
      26ca1b8963ce1b73e0e77113c711730545cacaaee7233c0526477d96d4c621fe

echo "third-party sources are in $REFS"
