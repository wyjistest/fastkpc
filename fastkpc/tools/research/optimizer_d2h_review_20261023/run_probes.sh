#!/bin/sh
# Build and run the standalone driver probes on idle GPU0; outputs go to OUT.
set -e
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUT=$1; BIN=$(mktemp -d)
mkdir -p "$OUT"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
for p in pageable_d2h_convoy queue_policy_probe queue_refill_probe queue_mapping_probe; do
  /usr/local/cuda/bin/nvcc -O2 -arch=sm_89 -std=c++17 -o "$BIN/$p" "$HERE/$p.cu"
done
export CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_MAX_CONNECTIONS=32
{ nvidia-smi --query-gpu=name,driver_version --format=csv,noheader -i 0; /usr/local/cuda/bin/nvcc --version | tail -2; } > "$OUT/environment.txt"
taskset -c 0-19 "$BIN/pageable_d2h_convoy" 8 1000 50 3 > "$OUT/pageable_d2h_convoy.txt"
taskset -c 0-19 "$BIN/pageable_d2h_convoy" 32 1000 50 3 > "$OUT/pageable_d2h_convoy_32threads.txt"
"$BIN/queue_policy_probe" > "$OUT/queue_policy_probe.txt"
"$BIN/queue_refill_probe" > "$OUT/queue_refill_probe.txt"
for m in none forward reverse; do "$BIN/queue_mapping_probe" 128 $m; done > "$OUT/queue_mapping_probe.txt"
"$BIN/queue_mapping_probe" 32 forward >> "$OUT/queue_mapping_probe.txt"
"$BIN/queue_mapping_probe" 64 forward >> "$OUT/queue_mapping_probe.txt"
rm -rf "$BIN"
echo PROBES_DONE
