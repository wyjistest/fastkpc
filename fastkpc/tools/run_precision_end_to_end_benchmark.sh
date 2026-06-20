#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
Rscript fastkpc/tools/run_precision_end_to_end_benchmark.R "${1:-fastkpc/artifacts/precision_end_to_end_benchmark}"
