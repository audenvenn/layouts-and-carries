#!/bin/sh
# Install elan (Lean toolchain manager) without touching shell profiles, then create the
# Lean project in ./lean with a Mathlib dependency.  The Mathlib build cache is NOT fetched
# here (MATHLIB_NO_CACHE_ON_UPDATE=1); fetch only the modules we import afterwards with
#   lake exe cache get <modules>
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="${1:?usage: setup_lean.sh <tmpdir>}"
export PATH="$HOME/.elan/bin:$PATH"
if ! command -v elan >/dev/null 2>&1; then
  curl -sSfL https://elan.lean-lang.org/elan-init.sh -o "$TMP/elan-init.sh"
  sh "$TMP/elan-init.sh" -y --no-modify-path --default-toolchain none
fi
elan --version
cd "$ROOT/lean"
if [ ! -f lakefile.toml ] && [ ! -f lakefile.lean ]; then
  MATHLIB_NO_CACHE_ON_UPDATE=1 lake +leanprover-community/mathlib4:lean-toolchain init CuteLayout math
fi
cat lean-toolchain
MATHLIB_NO_CACHE_ON_UPDATE=1 lake update
echo "SETUP_DONE"
du -sh "$HOME/.elan" "$ROOT/lean" 2>/dev/null
