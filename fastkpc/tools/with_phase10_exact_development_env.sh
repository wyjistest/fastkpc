#!/bin/sh
# Development configuration qualified on the RTX 4090, canonical 351x48.
# Set CUDA connections before the child creates a CUDA context. Explicit
# caller settings take precedence; this does not change library defaults.
set -eu

if [ "$#" -eq 0 ]; then
  echo "usage: sh fastkpc/tools/with_phase10_exact_development_env.sh COMMAND [ARG ...]" >&2
  exit 2
fi

: "${CUDA_VISIBLE_DEVICES:=0}"
: "${CUDA_DEVICE_MAX_CONNECTIONS:=32}"
: "${OPENBLAS_NUM_THREADS:=1}"
: "${OMP_NUM_THREADS:=1}"
: "${MKL_NUM_THREADS:=1}"
: "${FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES:=11811160064}"
: "${FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS:=1}"
: "${FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES:=8}"
export CUDA_VISIBLE_DEVICES CUDA_DEVICE_MAX_CONNECTIONS
export OPENBLAS_NUM_THREADS OMP_NUM_THREADS MKL_NUM_THREADS
export FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES
export FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS
export FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES

printf 'Phase 10 development: GPU=%s connections=%s cache_bytes=%s bypass=%s eig_lanes=%s\n' \
  "$CUDA_VISIBLE_DEVICES" "$CUDA_DEVICE_MAX_CONNECTIONS" \
  "$FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES" \
  "$FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS" \
  "$FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES" >&2
exec "$@"
