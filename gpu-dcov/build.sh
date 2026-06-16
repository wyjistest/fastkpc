#!/bin/sh
# Build dcov_gpu.so for R (.Call interface), sm_89 = RTX 4090
set -e
cd "$(dirname "$0")"
/usr/local/cuda/bin/nvcc -O3 -arch=sm_89 -Xcompiler -fPIC -shared \
  -I"$(R CMD config --cppflags | sed 's/^-I//')" \
  dcov_gpu.cu -o dcov_gpu.so \
  -L"$(R RHOME)/lib" -lR
echo "built: $(pwd)/dcov_gpu.so"
