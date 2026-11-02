// Candidate sweep-only helper. Direct 2x2/scal/swap keep their original wrappers.
__device__ void ip_consume_sweep(RotationLog* log, double* x, double* y,
                               double c, double s) {
  if (c == 1.0 && s == 0.0) return;
  record_rotation(log, int(x-log->base), int(y-log->base), c, s);
  const double a=*x, b=*y;
  *x=c*a+s*b;
  *y=c*b-s*a;
}
