"""Mechanically specialize the vendored LAPACK DBDSQR for a rotation log.

The numerical iteration is unchanged. Only the four operations on VT are
redirected to wrappers that both apply P^T to one vector and record the
transform for reverse application to the requested eigenvectors.
Usage: python3 generate_recorded_dbdsqr.py OUTPUT
"""
import hashlib
from pathlib import Path
import sys

source = Path("fastkpc/src/cuda/third_party/lapack_3_12_small_dgesdd_device.inc")
text = source.read_text()
start = text.index("__device__ int dbdsqr_(")
end = text.index("} /* dbdsqr_ */", start) + len("} /* dbdsqr_ */")
body = text[start:end].replace("dbdsqr_(", "recorded_dbdsqr_(", 1)
needle = "ftnlen uplo_len)"
assert body.count(needle) == 1
body = body.replace(needle, "ftnlen uplo_len, RotationLog* log)")
for name in ("drot_", "dlasr_", "dscal_", "dswap_"):
    assert name + "(" in body
    body = body.replace(name + "(", "record_" + name + "(log, ")
header = ("// Generated from the repository's LAPACK 3.12 device port.\n"
          "// Original LAPACK license and authorship remain applicable.\n"
          "// Source SHA256: " + hashlib.sha256(source.read_bytes()).hexdigest() + "\n")
Path(sys.argv[1]).write_text(header + body + "\n")
print("generated", sys.argv[1], "bytes", len(header + body))
