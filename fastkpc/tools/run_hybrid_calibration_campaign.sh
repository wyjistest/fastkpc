#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
Rscript fastkpc/tools/run_hybrid_calibration_campaign.R "${1:-fastkpc/artifacts/hybrid_calibration}"
