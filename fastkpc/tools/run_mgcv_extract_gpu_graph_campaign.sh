#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
Rscript fastkpc/tools/run_mgcv_extract_gpu_graph_campaign.R "${1:-fastkpc/artifacts/mgcv_extract_gpu_graph}"
