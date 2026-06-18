#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
Rscript fastkpc/tools/run_precision_ladder_timing_campaign.R "$@"
