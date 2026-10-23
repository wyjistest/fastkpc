#!/usr/bin/env python3
"""Build an isolated, namespaced copy of the production optimizer.

Only observation hooks and the candidate's terminal shadow-confirmation switch
are changed. The driver launches the primary kernel directly (no replay/merge).
No production source, object, shared library, or default is overwritten.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[4]
here = Path(__file__).resolve().parent
out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=True)
source = root / "fastkpc/src/cuda/mgcv_multi_penalty_gcv.cu"
text = source.read_text()
source_sha = hashlib.sha256(source.read_bytes()).hexdigest()

def replace(old, new):
    global text
    if text.count(old) != 1:
        raise RuntimeError(f"non-unique instrumentation anchor: {old[:100]!r}")
    text = text.replace(old, new)

helper = r'''
// Isolated diagnostic hooks. No exact-shadow value is a candidate guard input.
struct ProbeRecord {
  int kind, iteration, index, decision, route, rank, aggregate_rank;
  double score, condition, delta, noise, margin1, margin2;
  double log_sp[kMultiPenaltyGcvMaximumPenaltyCount];
};
__device__ ProbeRecord* probe_records = nullptr;
__device__ unsigned int* probe_counts = nullptr;
constexpr unsigned int kProbeCapacity = 4096;
__device__ void probe_log(int target, int kind, int iteration, int index,
    int decision, const DeviceTargetEvaluation& value, const double* log_sp,
    int penalty_count, double delta, double yy, int n,
    double margin1 = 0.0, double margin2 = 0.0) {
  if (probe_records == nullptr) return;
  unsigned int slot = probe_counts[target]++;
  if (slot >= kProbeCapacity) return;
  ProbeRecord& record = probe_records[target * kProbeCapacity + slot];
  record.kind = kind; record.iteration = iteration; record.index = index;
  record.decision = decision; record.route = value.decomposition_route;
  record.rank = value.numerical_rank;
  record.aggregate_rank = value.aggregate_penalty_rank;
  record.score = value.score; record.condition = value.condition;
  record.delta = delta;
  const double denominator = static_cast<double>(n) - value.edf;
  record.noise = 2.0 * kLapackEpsilon * yy * n /
    (denominator * denominator);
  record.margin1 = margin1; record.margin2 = margin2;
  for (int k = 0; k < penalty_count; ++k) record.log_sp[k] = log_sp[k];
}

'''
replace("__global__ void optimize_multi_penalty_targets_kernel(",
        helper + "__global__ void optimize_multi_penalty_targets_kernel(")
replace("  if (state->optimizer_status != 0) return;\n\n  for (int iteration = 1;",
        """  if (state->optimizer_status != 0) return;
  if (threadIdx.x == 0) probe_log(target, 0, 0, 0, 0, state->current,
    state->log_sp, penalty_count, 0.0, target_squared_norm, n);

  for (int iteration = 1;""")
replace("            state->trial_accepted = accepted ? 1 : 0;",
        """            state->trial_accepted = accepted ? 1 : 0;
            probe_log(target, 1, iteration, trial_index, accepted,
              state->trial, state->trial_log_sp, penalty_count,
              score_delta, target_squared_norm, n,
              maximum_step - convergence_step,
              score_delta - comparison_slack);""")
replace("        if (state->tries == max_step_halving) state->converged = 1;",
        """        if (state->tries == max_step_halving) state->converged = 1;
        probe_log(target, 2, iteration, state->tries, state->converged,
          state->current, state->log_sp, penalty_count,
          state->score_reduction, target_squared_norm, n,
          state->score_reduction /
            (convergence_tolerance * (1.0 + state->minimum_score)) - 1.0,
          gradient_norm /
            (gradient_tolerance_factor * (1.0 + fabs(state->minimum_score))) - 1.0);""")
replace("        if (state->use_steepest_descent == 0) {\n          double spectral_step",
        """        if (probe_records != nullptr) {
          double smallest = CUDART_INF, largest = 0.0;
          for (int k = 0; k < penalty_count; ++k) {
            smallest = fmin(smallest, fabs(state->hessian_eigenvalues[k]));
            largest = fmax(largest, fabs(state->hessian_eigenvalues[k]));
          }
          probe_log(target, 3, iteration, 0, state->use_steepest_descent,
            state->current, state->log_sp, penalty_count, 0.0,
            target_squared_norm, n, smallest / largest);
        }
        if (state->use_steepest_descent == 0) {
          double spectral_step""")
replace("          if (maximum_component > max_newton_step) {",
        """          probe_log(target, 4, iteration, 0,
            maximum_component > max_newton_step, state->current,
            state->log_sp, penalty_count, 0.0, target_squared_norm, n,
            maximum_component / max_newton_step - 1.0);
          if (maximum_component > max_newton_step) {""")
replace("            !target_force_stable_svd &&\n            (score_delta >= 0.0 ||",
        """            false && !target_force_stable_svd &&
            (score_delta >= 0.0 ||""")
replace("        if (state->optimizer_status == 0) {\n          if (accepted) {",
        """        probe_log(target, 5, coordinate + 1, probe, accepted,
          state->trial, state->trial_log_sp, penalty_count,
          primary_score_delta, target_squared_norm, n);
        if (state->optimizer_status == 0) {
          if (accepted) {""")

generated = out / "instrumented_optimizer.cu"
generated.write_text(text)
command = [
    "/usr/local/cuda/bin/nvcc", "-O3", "-arch=sm_89", "--fmad=false",
    "-diag-suppress", "2464", "-std=c++17", "-shared", "-Xcompiler", "-fPIC",
    "-I/usr/local/lib/R/include",
    "-I/home/amax/R/x86_64-pc-linux-gnu-library/4.4/Rcpp/include",
    f"-I{root}/fastkpc/src/cuda", f"-I{out}",
    str(here / "probe.cu"), "-lcublas", "-o", str(out / "guarded_qr_probe.so"),
]
manifest = dict(source=str(source.relative_to(root)), source_sha256=source_sha,
                generated_sha256=hashlib.sha256(generated.read_bytes()).hexdigest(),
                command=command, production_modified=False)
(out / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")
# CUDA translation already embeds the device image; link that translation
# directly so the expensive ptxas pass runs only once.
host_command = command.copy()
host_command[host_command.index("-shared")] = "--cuda"
host_command[-1] = str(out / "probe.host.ii")
host_command.remove("-lcublas")
subprocess.run(host_command, check=True)
subprocess.run(["g++", "-std=c++17", "-fPIC", "-fsyntax-only", "-x", "c++",
                str(out / "probe.host.ii")], check=True)
link_command = ["g++", "-O3", "-std=c++17", "-shared", "-fPIC", "-x", "c++",
                str(out / "probe.host.ii"), "-x", "none", "-L/usr/local/cuda/lib64",
                "-Wl,-rpath,/usr/local/cuda/lib64", "-lcudart", "-lcublas",
                "-o", str(out / "guarded_qr_probe.so")]
manifest["command"] = host_command
manifest["link_command"] = link_command
manifest["host_translation_sha256"] = hashlib.sha256((out / "probe.host.ii").read_bytes()).hexdigest()
subprocess.run(link_command, check=True)
manifest["probe_sha256"] = hashlib.sha256((out / "guarded_qr_probe.so").read_bytes()).hexdigest()
(out / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
