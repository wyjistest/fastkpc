#!/bin/sh
set -e

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
rm -f "$ROOT"/build/*.o
rm -f "$ROOT"/build/fastkpc_cuda.so
