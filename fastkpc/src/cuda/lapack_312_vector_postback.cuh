// Included inside the selected LAPACK namespace, only by the independent
// vector-postback backend. The original backend never compiles this path.
// Keep the full DORMBR workspace contract, not a sliced DORMBR invocation.
template<bool Slice>
__device__ integer postback_apply(
    bool left, integer m, integer q, doublereal* a, integer lda,
    doublereal* tau, doublereal* c, integer ldc, doublereal* work) {
  constexpr integer nbmax = kReferenceReflectorBlockSize;
  constexpr integer ldtri = kReferenceReflectorLeadingDimension;
  constexpr integer tsize = ldtri * (2 * nbmax);
  static_assert(nbmax == 32 && ldtri == 65 && tsize == 4160,
                "requalify vector postback when the reference block contract changes");
  const integer lane = static_cast<integer>(threadIdx.x) & 31;
  const integer lwork = dgesdd_bdspac(q);
  const integer nb = lwork < q * nbmax + tsize ? (lwork - tsize) / q : nbmax;
  const integer reflectors = left ? q : q - 1;
  const integer base = left ? 0 : 1;
  integer ldt = ldtri, ldwork = q;
  char side = left ? 'L' : 'R', trans = left ? 'N' : 'T';
  char forward = 'F', store = left ? 'C' : 'R';
  doublereal* t = work + q * nb;
  if (lane == 0) {
    work[0] = static_cast<doublereal>(q * nbmax);
    work[0] = 1.0;
    work[0] = static_cast<doublereal>(q * nbmax + tsize);
  }
  __syncwarp();
  for (integer offset = (reflectors - 1) / nb * nb; offset >= 0; offset -= nb) {
    integer ib = reflectors - offset < nb ? reflectors - offset : nb;
    integer order = left ? m - offset : q - 1 - offset;
    doublereal* v = a + offset + lda * (base + offset);
    if (lane == 0) {
      reference_blocked_reflector::dlarft_(
        &forward, &store, &order, &ib, v, &lda, tau + offset, t, &ldt, 1, 1);
    }
    __syncwarp();
    doublereal* output = c + (left ? offset : ldc * (1 + offset));
    if constexpr (Slice) {
      for (integer vector = lane; vector < q; vector += 32) {
        integer rows = left ? order : 1, columns = left ? 1 : order;
        reference_blocked_reflector::dlarfb_(
          &side, &trans, &forward, &store, &rows, &columns, &ib,
          v, &lda, t, &ldt, output + (left ? vector * ldc : vector), &ldc,
          work + vector, &ldwork, 1, 1, 1, 1);
      }
    } else if (lane == 0) {
      integer rows = left ? order : q, columns = left ? q : order;
      reference_blocked_reflector::dlarfb_(
        &side, &trans, &forward, &store, &rows, &columns, &ib,
        v, &lda, t, &ldt, output, &ldc, work, &ldwork, 1, 1, 1, 1);
    }
    // Includes lanes with no second vector. Next block may overwrite T/W.
    __syncwarp();
  }
  if (lane == 0) {
    work[0] = static_cast<doublereal>(q * nbmax + tsize);
    work[0] = static_cast<doublereal>(q * nbmax);
  }
  __syncwarp();
  return 0;
}
