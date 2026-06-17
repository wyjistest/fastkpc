#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
out="${1:-fastkpc/artifacts/mgcv_gate_b}"
Rscript fastkpc/tools/run_mgcv_gate_b_campaign.R "$out"
