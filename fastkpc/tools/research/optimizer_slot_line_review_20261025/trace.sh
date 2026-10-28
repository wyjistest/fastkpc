#!/bin/sh
# One production canonical run under the minimal CUPTI injection tracer (GPU1).
# sh trace.sh ARTIFACT_DIR [RUN_NAME]   (expects ARTIFACT_DIR/ktrace.so built from ktrace.cpp)
# Other FASTKPC_* settings in the caller environment are inherited by the run.
set -eu
art=$(cd "$1" && pwd)
cd "$(dirname "$0")/../../../.."
out="$art/${2:-ktrace-run}"
test ! -e "$out"
mkdir -p "$out"
export CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_MAX_CONNECTIONS=32
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES=11811160064
export FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS=1
export FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES=8
nvidia-smi -i 1 --query-gpu=timestamp,uuid,name,driver_version,clocks.sm,temperature.gpu --format=csv > "$out/environment.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv >> "$out/environment.csv"
KTRACE_DIR="$out" CUDA_INJECTION64_PATH="$art/ktrace.so" taskset -c 0-19 \
  Rscript --vanilla fastkpc/tools/research/optimizer_slot_line_review_20261025/run_profile.R "$out"
