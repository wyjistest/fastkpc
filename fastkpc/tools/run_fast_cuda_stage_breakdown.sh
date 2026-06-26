#!/usr/bin/env bash
set -euo pipefail

Rscript fastkpc/tools/run_fast_cuda_stage_breakdown.R "$@"
