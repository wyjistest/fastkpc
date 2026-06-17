#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
out="${1:-fastkpc/artifacts/hybrid_compatibility}"
Rscript fastkpc/tools/run_hybrid_compatibility_campaign.R "$out"
