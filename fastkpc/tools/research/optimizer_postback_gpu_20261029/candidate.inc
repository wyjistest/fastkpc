// Research only: include inside the authenticated LAPACK namespace, after
// dgesdd_bdspac. Original reference DLARFT/DLARFB bodies are unchanged.
#ifndef POSTBACK_PROFILE
#define POSTBACK_PROFILE 0
#endif
#ifndef POSTBACK_SLICES
#define POSTBACK_SLICES 1
#endif
#if POSTBACK_PROFILE
// q37/q46; Q/immediate P/deferred P; calls/blocks/total/FT/FB cycles.
__device__ unsigned long long postback_stats[2][3][5];
#endif

template<bool Slice>
__device__ integer postback_apply(
    bool left, integer m, integer q, doublereal* a, integer lda,
    doublereal* tau, doublereal* c, integer ldc, doublereal* work,
    integer deferred = 0) {
  const integer lane = static_cast<integer>(threadIdx.x) & 31;
  const integer lwork = dgesdd_bdspac(q);
  const integer nb = lwork < q * 32 + 4160 ? (lwork - 4160) / q : 32;
  const integer reflectors = left ? q : q - 1;
  const integer base = left ? 0 : 1;
  integer ldt = 65, ldwork = q;
  char side = left ? 'L' : 'R', trans = left ? 'N' : 'T';
  char forward = 'F', store = left ? 'C' : 'R';
  doublereal* t = work + q * nb;
  unsigned long long total_start = 0, ft_cycles = 0, fb_cycles = 0, blocks = 0;
#if POSTBACK_PROFILE
  if (lane == 0) total_start = clock64();
#endif
  if (lane == 0) {
    // Preserve DORMBR/DORMQR/DORMLQ workspace metadata lifetime.
    work[0] = static_cast<doublereal>(q * 32);
    work[0] = 1.0;
    work[0] = static_cast<doublereal>(q * 32 + 4160);
  }
  __syncwarp();
  for (integer offset = (reflectors - 1) / nb * nb; offset >= 0; offset -= nb) {
    integer ib = reflectors - offset < nb ? reflectors - offset : nb;
    integer order = left ? m - offset : q - 1 - offset;
    doublereal* v = a + offset + lda * (base + offset);
    unsigned long long start = 0;
    if (lane == 0) {
#if POSTBACK_PROFILE
      start = clock64();
#endif
      reference_blocked_reflector::dlarft_(
        &forward, &store, &order, &ib,
        v, &lda, tau + offset, t, &ldt, 1, 1);
#if POSTBACK_PROFILE
      ft_cycles += clock64() - start;
#endif
    }
    __syncwarp();
#if POSTBACK_PROFILE
    if (lane == 0) start = clock64();
#endif
    doublereal* output = c + (left ? offset : ldc * (1 + offset));
    if constexpr (Slice) {
      for (integer vector = lane; vector < q; vector += 32) {
        integer rows = left ? order : 1;
        integer columns = left ? 1 : order;
        reference_blocked_reflector::dlarfb_(
          &side, &trans, &forward, &store,
          &rows, &columns, &ib, v, &lda, t, &ldt,
          output + (left ? vector * ldc : vector), &ldc,
          work + vector, &ldwork, 1, 1, 1, 1);
      }
    } else if (lane == 0) {
      integer rows = left ? order : q;
      integer columns = left ? q : order;
      reference_blocked_reflector::dlarfb_(
        &side, &trans, &forward, &store,
        &rows, &columns, &ib, v, &lda, t, &ldt, output, &ldc,
        work, &ldwork, 1, 1, 1, 1);
    }
    // All 32 lanes participate, including lanes without vector+32 work.
    __syncwarp();
#if POSTBACK_PROFILE
    if (lane == 0) { fb_cycles += clock64() - start; ++blocks; }
#endif
  }
  if (lane == 0) {
    work[0] = static_cast<doublereal>(q * 32 + 4160);
    work[0] = static_cast<doublereal>(q * 32);
#if POSTBACK_PROFILE
    auto* stats = postback_stats[q == 37 ? 0 : 1][left ? 0 : (deferred ? 2 : 1)];
    const unsigned long long total = clock64() - total_start;
    atomicAdd(stats, 1ULL); atomicAdd(stats + 1, blocks);
    atomicAdd(stats + 2, total); atomicAdd(stats + 3, ft_cycles);
    atomicAdd(stats + 4, fb_cycles);
#endif
  }
  __syncwarp();
  return 0;
}
