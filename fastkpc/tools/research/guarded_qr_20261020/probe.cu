// No symbols or ABI are shared with the installed production optimizer.
#define fastkpc guarded_qr_probe
#include "instrumented_optimizer.cu"
#undef fastkpc
#include <Rcpp.h>

namespace gp = guarded_qr_probe;

extern "C" SEXP C_guarded_qr_probe(SEXP prepared_s, SEXP Y_s,
                                   SEXP exact_s, SEXP repeats_s) {
  BEGIN_RCPP
  Rcpp::List p(prepared_s);
  Rcpp::NumericMatrix X(p["X"]), qr(p["magic_qr_packed"]), r(p["magic_r"]), Y(Y_s);
  Rcpp::NumericVector tau(p["magic_tau"]), initial(p["initial_log_sp"]);
  Rcpp::IntegerVector pivot(p["magic_pivot"]), ranks_r(p["penalty_ranks"]);
  Rcpp::List roots_r(p["penalty_roots"]), matrices_r(p["penalty_matrices"]);
  const int n = X.nrow(), q = X.ncol(), m = roots_r.size(), nt = Y.ncol();
  const int repeats = Rcpp::as<int>(repeats_s);
  const bool exact = Rcpp::as<bool>(exact_s);
  if (nt != 2 || n != Y.nrow() || q > 80 || m > 8 ||
      repeats < 1 || repeats > 11) Rcpp::stop("probe envelope exceeded");
  std::vector<int> piv(pivot.begin(), pivot.end()), ranks(ranks_r.begin(), ranks_r.end());
  for (int& value : piv) --value;
  std::vector<std::vector<double>> roots, matrices;
  for (int k = 0; k < m; ++k) {
    roots.push_back(Rcpp::as<std::vector<double>>(roots_r[k]));
    matrices.push_back(Rcpp::as<std::vector<double>>(matrices_r[k]));
  }
  auto prepared = gp::create_multi_penalty_gcv_cuda_prepared(
    X.begin(), qr.begin(), tau.begin(), r.begin(), piv.data(), roots,
    matrices, ranks, initial.begin(), n, q, m, nt, 0);
  auto stream = prepared->stream.get();
  gp::check_cuda(cudaMemcpyAsync(prepared->d_Y.get(), Y.begin(), n * nt * sizeof(double),
    cudaMemcpyHostToDevice, stream), "probe Y upload");
  gp::target_squared_norm_kernel<<<1, gp::kBlockSize, 0, stream>>>(
    prepared->d_Y.get(), prepared->d_squared_norm.get(), n, nt);
  gp::magic_qt_y_kernel<<<nt, gp::kBlockSize, 0, stream>>>(
    prepared->d_qr_packed.get(), prepared->d_tau.get(), prepared->d_Y.get(),
    prepared->d_qt_work.get(), prepared->d_y0.get(), prepared->d_workspaces.get(), n, q, nt);
  gp::check_cuda(cudaGetLastError(), "probe response preparation");
  gp::MultiPenaltyGcvCudaDiagnostics diag;
  gp::DeviceBuffer<gp::ProbeRecord> records;
  gp::DeviceBuffer<unsigned int> counts;
  records.allocate(nt * gp::kProbeCapacity, &diag);
  counts.allocate(nt, &diag);
  auto record_ptr = records.get(); auto count_ptr = counts.get();
  gp::check_cuda(cudaMemcpyToSymbol(gp::probe_counts, &count_ptr, sizeof(count_ptr)), "probe count binding");
  gp::check_cuda(cudaMemsetAsync(count_ptr, 0, nt * sizeof(unsigned int), stream), "probe count reset");
  auto launch = [&]() {
    gp::optimize_multi_penalty_targets_kernel<<<nt, gp::kBlockSize, 0, stream>>>(
      prepared->d_r.get(), prepared->d_pivot.get(), prepared->d_roots.get(),
      prepared->d_root_offsets.get(), prepared->d_ranks.get(), prepared->d_matrices.get(),
      prepared->d_y0.get(), prepared->d_squared_norm.get(), prepared->d_initial_log_sp.get(),
      prepared->d_workspaces.get(), prepared->d_states.get(), nullptr, n, q, m, nt,
      1e-7, std::pow(1e-7, 1.0 / 3.0), 25, 400, 5.0, 2.0, 5,
      1.4901161193847656e-8, false, false, exact, true, nullptr, nullptr, 0);
    gp::check_cuda(cudaGetLastError(), "probe primary kernel");
  };
  gp::check_cuda(cudaMemcpyToSymbol(gp::probe_records, &record_ptr, sizeof(record_ptr)), "probe trace binding");
  launch();
  std::vector<gp::DeviceOptimizerState> traced(nt), timed(nt);
  std::vector<unsigned int> host_counts(nt);
  std::vector<gp::ProbeRecord> host_records(nt * gp::kProbeCapacity);
  gp::check_cuda(cudaMemcpyAsync(traced.data(), prepared->d_states.get(), nt * sizeof(traced[0]),
    cudaMemcpyDeviceToHost, stream), "probe traced states");
  gp::check_cuda(cudaMemcpyAsync(host_counts.data(), count_ptr, nt * sizeof(unsigned int),
    cudaMemcpyDeviceToHost, stream), "probe trace counts");
  gp::check_cuda(cudaMemcpyAsync(host_records.data(), record_ptr, host_records.size() * sizeof(host_records[0]),
    cudaMemcpyDeviceToHost, stream), "probe trace download");
  gp::check_cuda(cudaStreamSynchronize(stream), "probe trace completion");
  record_ptr = nullptr;
  gp::check_cuda(cudaMemcpyToSymbol(gp::probe_records, &record_ptr, sizeof(record_ptr)), "disable probe trace");
  gp::TimingEvent start, stop;
  Rcpp::NumericVector device_ms(repeats), host_ms(repeats);
  bool deterministic = true;
  for (int repetition = 0; repetition < repeats; ++repetition) {
    auto began = std::chrono::steady_clock::now();
    gp::check_cuda(cudaEventRecord(start.get(), stream), "probe timing start");
    launch();
    gp::check_cuda(cudaEventRecord(stop.get(), stream), "probe timing stop");
    gp::check_cuda(cudaEventSynchronize(stop.get()), "probe timing wait");
    host_ms[repetition] = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - began).count();
    float ms = 0;
    gp::check_cuda(cudaEventElapsedTime(&ms, start.get(), stop.get()), "probe device timing");
    device_ms[repetition] = ms;
    gp::check_cuda(cudaMemcpy(timed.data(), prepared->d_states.get(), nt * sizeof(timed[0]),
      cudaMemcpyDeviceToHost), "probe timed states");
    for (int t = 0; t < nt; ++t) {
      deterministic &= std::memcmp(traced[t].log_sp, timed[t].log_sp, m * sizeof(double)) == 0;
      deterministic &= std::memcmp(traced[t].current.coefficients, timed[t].current.coefficients,
        q * sizeof(double)) == 0 && traced[t].optimizer_status == timed[t].optimizer_status;
    }
  }
  Rcpp::NumericMatrix sp(m, nt), coefficients(q, nt), gradients(m, nt);
  Rcpp::NumericVector score(nt), edf(nt), rss(nt);
  Rcpp::IntegerVector status(nt), rank(nt), info(nt), iterations(nt), qr_calls(nt), svd_calls(nt);
  Rcpp::List traces(nt);
  for (int t = 0; t < nt; ++t) {
    const auto& state = traced[t];
    for (int k = 0; k < m; ++k) { sp(k,t) = state.log_sp[k]; gradients(k,t) = state.current.gradient[k]; }
    for (int k = 0; k < q; ++k) coefficients(k,t) = state.current.coefficients[k];
    score[t] = state.current.score; edf[t] = state.current.edf; rss[t] = state.current.rss;
    status[t] = state.optimizer_status; rank[t] = state.current.numerical_rank;
    info[t] = state.current.solver_info; iterations[t] = state.optimizer_iterations;
    qr_calls[t] = state.phase_timing.guarded_qr_evaluation_count;
    svd_calls[t] = state.phase_timing.stable_svd_evaluation_count;
    if (host_counts[t] > gp::kProbeCapacity) Rcpp::stop("probe trace overflow");
    Rcpp::NumericMatrix trace(host_counts[t], 13 + m);
    for (unsigned int j = 0; j < host_counts[t]; ++j) {
      const auto& a = host_records[t * gp::kProbeCapacity + j];
      double values[] = {double(a.kind),double(a.iteration),double(a.index),double(a.decision),
        double(a.route),double(a.rank),double(a.aggregate_rank),a.score,a.condition,
        a.delta,a.noise,a.margin1,a.margin2};
      for (int k = 0; k < 13; ++k) trace(j,k) = values[k];
      for (int k = 0; k < m; ++k) trace(j,13+k) = a.log_sp[k];
    }
    traces[t] = trace;
  }
  return Rcpp::List::create(Rcpp::Named("log_sp")=sp,
    Rcpp::Named("coefficients")=coefficients, Rcpp::Named("gradient")=gradients,
    Rcpp::Named("score")=score, Rcpp::Named("edf")=edf, Rcpp::Named("rss")=rss,
    Rcpp::Named("status")=status, Rcpp::Named("rank")=rank, Rcpp::Named("info")=info,
    Rcpp::Named("iterations")=iterations, Rcpp::Named("qr_calls")=qr_calls,
    Rcpp::Named("svd_calls")=svd_calls, Rcpp::Named("trace")=traces,
    Rcpp::Named("device_ms")=device_ms, Rcpp::Named("host_ms")=host_ms,
    Rcpp::Named("deterministic")=deterministic);
  END_RCPP
}
