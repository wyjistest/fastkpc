#!/bin/sh
# nsys CUDA-only trace of one production canonical run on GPU1 (review diagnostic).
# sh profile.sh ARTIFACT_DIR
set -eu
out=$1
cd "$(dirname "$0")/../../../.."
test ! -e "$out/canonical.nsys-rep"
mkdir -p "$out"
export CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_MAX_CONNECTIONS=32
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES=11811160064
export FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS=1
export FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES=8
nvidia-smi -i 1 --query-gpu=timestamp,uuid,name,driver_version,clocks.sm,temperature.gpu --format=csv > "$out/environment.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv >> "$out/environment.csv"
taskset -c 0-19 nsys profile --sample=none --cpuctxsw=none --trace=cuda \
  --output="$out/canonical" \
  Rscript --vanilla fastkpc/tools/research/optimizer_slot_line_review_20261025/run_profile.R "$out"
