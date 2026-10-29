// Host cross-check: full reference DLARFB vs per-vector DLARFB calls, bitwise, IEEE doubles.
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <iterator>
#include <openssl/sha.h>
typedef int integer; typedef int logical; typedef int ftnlen; typedef double doublereal;
#define __device__
using std::min; using std::max;
logical lsame_(char* a, char* b, ftnlen, ftnlen) { return std::toupper(*a) == std::toupper(*b); }
int xerbla_(char* name, integer* info, ftnlen) { std::fprintf(stderr, "xerbla %s %d\n", name, *info); std::abort(); }
#include VENDOR_INC

static const double kNaN = std::numeric_limits<double>::quiet_NaN();
static const int G = 64;  // guard elements on each side
struct Buf {
  std::vector<double> d; int n;
  explicit Buf(int count) : d(count + 2 * G, kNaN), n(count) {}
  double* p() { return d.data() + G; }
};
static bool same_bits(const Buf& a, const Buf& b) { return std::memcmp(a.d.data(), b.d.data(), a.d.size() * sizeof(double)) == 0; }
static bool guards_ok(const Buf& a, const Buf& orig) {
  return std::memcmp(a.d.data(), orig.d.data(), G * sizeof(double)) == 0 &&
         std::memcmp(a.d.data() + G + a.n, orig.d.data() + G + a.n, G * sizeof(double)) == 0;
}
int main(int argc, char** argv) {
  if (argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "--validate-only")) {
    std::fprintf(stderr, "usage: slice_check BLOCKS_TSV [--validate-only]\n"); return 2;
  }
  std::ifstream file(argv[1], std::ios::binary);
  if (!file) { std::fprintf(stderr, "cannot open input\n"); return 2; }
  const std::string bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  unsigned char digest[SHA256_DIGEST_LENGTH]; char hex[SHA256_DIGEST_LENGTH * 2 + 1];
  SHA256(reinterpret_cast<const unsigned char*>(bytes.data()), bytes.size(), digest);
  for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) std::sprintf(hex + i * 2, "%02x", digest[i]);
  if (std::string(hex) != "5dc6a5d3bd88b97e0f5fd093f5eb997fa5c5972b7370ede4c2930b1d6e2f8ab2") {
    std::fprintf(stderr, "inventory TSV identity mismatch\n"); return 2;
  }
  std::istringstream in(bytes); std::string line;
  long blocks = 0, cases = 0, fail = 0, finite_output_nan_count = 0, vectors = 0;
  while (std::getline(in, line)) {
    std::istringstream r(line); int q, rows, offset, ib, m, n, ldv, ldc; std::string side, path;
    if (!(r >> q >> side >> rows >> path >> offset >> ib >> m >> n >> ldv >> ldc) ||
        !(r >> std::ws).eof() || (q != 37 && q != 46) || (side != "Q" && side != "P") ||
        m < 1 || n < 1 || ib < 1 || ib > 32 || ldv < 1 || ldc < m) {
      std::fprintf(stderr, "invalid block row\n"); return 2;
    }
    ++blocks;
    if (argc == 3) continue;
    const bool left = side == "Q"; int k = ib, ldt = 65, ldw = q;
    for (int pattern = 0; pattern < 4; ++pattern) for (int seed = 0; seed < 3; ++seed) {
      std::mt19937_64 rng(1000003ULL * cases + 7 * pattern + seed);
      std::normal_distribution<double> nd; std::uniform_int_distribution<int> u(0, 99);
      auto val = [&](bool allow_special) {
        int z = u(rng);
        if (pattern >= 1 && z < 30) return (u(rng) & 1) ? 0.0 : -0.0;
        if (pattern == 3 && allow_special && z >= 97) { int s = u(rng) % 3; return s == 0 ? INFINITY : s == 1 ? -INFINITY : kNaN; }
        return nd(rng);
      };
      Buf V(ldv * (left ? k : n)), T(ldt * k), C(ldc * n), W(ldw * k);
      for (int j = 0; j < (left ? k : n); ++j) for (int i = 0; i < (left ? m : k); ++i) {
        bool unref = left ? (i < k && i <= j) : (j < k && i >= j);
        if (!unref) V.p()[i + ldv * j] = val(false);   // unreferenced V1 parts stay NaN sentinels
      }
      for (int j = 0; j < k; ++j) for (int i = 0; i <= j; ++i)
        T.p()[i + ldt * j] = (pattern == 2 && i == j) ? double(u(rng) & 1) : val(false);  // lower part stays NaN
      for (int j = 0; j < n; ++j) for (int i = 0; i < m; ++i) C.p()[i + ldc * j] = val(true);
      const Buf V0 = V, T0 = T, C0 = C, W0 = W;
      Buf Cs = C, Ws = W;
      char s = left ? 'L' : 'R', tr = left ? 'N' : 'T', dir = 'F', st = left ? 'C' : 'R';
      dlarfb_(&s, &tr, &dir, &st, &m, &n, &k, V.p(), &ldv, T.p(), &ldt, C.p(), &ldc, W.p(), &ldw, 1, 1, 1, 1);
#ifndef SENSITIVITY_CONTROL
      for (int lane = 31; lane >= 0; --lane) for (int v = lane; v < q; v += 32) {
        int sm = left ? m : 1, sn = left ? 1 : n;
        dlarfb_(&s, &tr, &dir, &st, &sm, &sn, &k, V.p(), &ldv, T.p(), &ldt, Cs.p() + (left ? v * ldc : v), &ldc, Ws.p() + v, &ldw, 1, 1, 1, 1);
        ++vectors;
      }
#else
      // Sensitivity control: flip the lowest mantissa bit of one finite nonzero C input before slicing.
      for (int j = 0, done = 0; j < n && !done; ++j) for (int i = 0; i < m && !done; ++i) {
        double& x = Cs.p()[i + ldc * j];
        if (std::isfinite(x) && x != 0.0) { uint64_t bits; std::memcpy(&bits, &x, 8); bits ^= 1; std::memcpy(&x, &bits, 8); done = 1; }
      }
      for (int lane = 31; lane >= 0; --lane) for (int v = lane; v < q; v += 32) {
        int sm = left ? m : 1, sn = left ? 1 : n;
        dlarfb_(&s, &tr, &dir, &st, &sm, &sn, &k, V.p(), &ldv, T.p(), &ldt, Cs.p() + (left ? v * ldc : v), &ldc, Ws.p() + v, &ldw, 1, 1, 1, 1);
        ++vectors;
      }
#endif
      bool ok = same_bits(C, Cs) && same_bits(W, Ws) && same_bits(V, V0) && same_bits(T, T0) &&
                guards_ok(C, C0) && guards_ok(W, W0) && guards_ok(Cs, C0) && guards_ok(Ws, W0);
      if (pattern < 3) {  // Observe NaN propagation, not memory reads.
        for (int j = 0; j < n; ++j) for (int i = 0; i < m; ++i) if (std::isnan(C.p()[i + ldc * j])) { ++finite_output_nan_count; ok = false; }
      }
      ++cases; if (!ok) { ++fail; if (fail < 5) std::fprintf(stderr, "FAIL %s\n", line.c_str()); }
    }
  }
  if (blocks != 1553 || (argc == 2 && (cases != 18636 || vectors != 709404))) {
    std::fprintf(stderr, "incomplete coverage\n"); return 2;
  }
  if (argc == 3) { std::printf("PASS input identity and 1553 block rows\n"); return 0; }
  std::printf("cases %ld failures %ld finite_output_nan_count %ld vector_calls %ld\n", cases, fail, finite_output_nan_count, vectors);
  return fail != 0;
}
