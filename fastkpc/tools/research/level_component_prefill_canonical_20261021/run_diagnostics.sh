#!/bin/sh
# Run only after the uninstrumented canonical off/on pair has completed.
set -eu
LSCP_WORK=${1:?usage: run_diagnostics.sh WORK_DIRECTORY}
LSCP_NATIVE=$(realpath fastkpc/build/fastkpc_cuda.so)
LSCP_TOOLS=fastkpc/tools/research/level_component_prefill_canonical_20261021
Rscript -e 'a<-commandArgs(TRUE); for(p in a){x<-readRDS(p);stopifnot(all(unlist(x$gates)))}' \
  "$LSCP_WORK/canonical-off/receipt.rds" "$LSCP_WORK/canonical-on/receipt.rds"

for LSCP_SCOPE in probe audit; do
  for LSCP_MODE in off on; do
    LSCP_OUTPUT="$LSCP_WORK/$LSCP_SCOPE-$LSCP_MODE"
    test ! -e "$LSCP_OUTPUT/receipt.rds"
    mkdir -p "$LSCP_OUTPUT/raw"
    LSCP_AUDIT_MODE=capture
    if [ "$LSCP_MODE" = on ]; then LSCP_AUDIT_MODE=compare; fi
    if [ "$LSCP_SCOPE" = probe ]; then
      LSCP_REFERENCE=-
      if [ "$LSCP_MODE" = on ]; then LSCP_REFERENCE="$LSCP_WORK/probe-off/receipt.rds"; fi
      env LD_PRELOAD="$LSCP_WORK/audit_payload.so" \
        FASTKPC_LSCP_TEST_NATIVE="$LSCP_NATIVE" \
        FASTKPC_LSCP_CANONICAL_AUDIT_DIR="$LSCP_OUTPUT" \
        FASTKPC_LSCP_CANONICAL_AUDIT_MODE="$LSCP_AUDIT_MODE" \
        FASTKPC_LSCP_CANONICAL_REFERENCE="$LSCP_WORK/probe-off" \
        FASTKPC_LSCP_PAYLOAD_LOG="$LSCP_OUTPUT/external-diagnostic-only" \
        sh fastkpc/tools/with_phase10_exact_development_env.sh \
        Rscript fastkpc/tools/research/level_component_prefill_20261021/run.R \
        "$LSCP_OUTPUT" "$LSCP_MODE" "$LSCP_REFERENCE" 3 > "$LSCP_WORK/$LSCP_SCOPE-$LSCP_MODE.log" 2>&1
    else
      env LD_PRELOAD="$LSCP_WORK/audit_payload.so" \
        FASTKPC_LSCP_TEST_NATIVE="$LSCP_NATIVE" \
        FASTKPC_LSCP_CANONICAL_AUDIT_DIR="$LSCP_OUTPUT" \
        FASTKPC_LSCP_CANONICAL_AUDIT_MODE="$LSCP_AUDIT_MODE" \
        FASTKPC_LSCP_CANONICAL_REFERENCE="$LSCP_WORK/audit-off" \
        sh fastkpc/tools/with_phase10_exact_development_env.sh \
        Rscript "$LSCP_TOOLS/run.R" "$LSCP_OUTPUT" "$LSCP_MODE" \
        "$LSCP_WORK/canonical-off/receipt.rds" "$LSCP_WORK/canonical-$LSCP_MODE/receipt.rds" \
        > "$LSCP_WORK/$LSCP_SCOPE-$LSCP_MODE.log" 2>&1
    fi
  done
  if [ "$LSCP_SCOPE" = probe ]; then
    Rscript -e 'p<-commandArgs(TRUE)[1]; x<-read.delim(p); stopifnot(nrow(x)==4148L,all(x$residual_changed==0),all(x$fitted_changed==0),all(x$vectors_changed==0),all(x$values_changed==0),all(x$total_changed==0),all(x$self_changed==0));cat("PASS observer smoke: 4148 residuals and four-field payloads exact\n")' \
      "$LSCP_WORK/probe-on/comparison.tsv" > "$LSCP_WORK/observer-smoke.log"
  fi
done
Rscript "$LSCP_TOOLS/attribute.R" "$LSCP_WORK" > "$LSCP_WORK/attribution.log" 2>&1
