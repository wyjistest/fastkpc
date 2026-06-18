#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
Rscript fastkpc/tools/run_tprs_approx_decision_memo.R "${1:-fastkpc/artifacts/tprs_approx_decision}"
