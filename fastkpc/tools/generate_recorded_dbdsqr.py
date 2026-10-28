"""Specialize the vendored LAPACK DBDSQR for the structured rotation log.

The numerical iteration is unchanged. Only operations on VT are redirected
to wrappers that apply and record the transform. The input is resolved from
this file, so a production source snapshot needs no research directory or
particular working directory.

Usage: python3 generate_recorded_dbdsqr.py OUTPUT [FIXED_VT_OUTPUT]
"""
import hashlib
from pathlib import Path
import sys


def replace_once(text, old, new, count=1):
    assert text.count(old) == count, (old, text.count(old), count)
    return text.replace(old, new)


def fixed_vt_work_pair(text):
    """Fixed L/1/0/0 entry: retain each consumed rotation pair, including signs.

    Forward VT consumes W0/W1; backward VT consumes W2/W3. Reassign only
    backward stores/consumers. Scalar iteration, helpers and sorting stay exact.
    This is not a general DBDSQR with four aliased work segments.
    """
    for forward, vals in [(True, ('cs', 'sn', 'oldcs', 'oldsn')),
                          (False, ('cs', '-sn', 'oldcs', '-oldsn')),
                          (True, ('cosr', 'sinr', 'cosl', 'sinl')),
                          (False, ('cosr', '-sinr', 'cosl', '-sinl'))]:
        idx = 'i__ - ll + 1' if forward else 'i__ - ll'
        rows = ['\t\twork[' + idx + suffix + '] = ' + value + ';'
                for suffix, value in zip(['', ' + nm1', ' + nm12', ' + nm13'], vals)]
        kept = rows[:2] if forward else [rows[2].replace(' + nm12', ''),
                                         rows[3].replace(' + nm13', ' + nm1')]
        text = replace_once(text, '\n'.join(rows), '\n'.join(kept))
    text = replace_once(text,
        'record_dlasr_(log, "L", "V", "B", &i__1, ncvt, &work[nm12 + 1], &work[\n\t\t\tnm13 + 1]',
        'record_dlasr_(log, "L", "V", "B", &i__1, ncvt, &work[1], &work[*n]', 2)
    text = replace_once(text, '    /* System generated locals */',
        "    if (*uplo != 'L' || *ncvt != 1 || *nru != 0 || *ncc != 0) { *info = -101; return 0; }\n"
        '    /* System generated locals */')
    return replace_once(text, 'recorded_dbdsqr_(', 'recorded_dbdsqr_fixed_vt_(')


def main():
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: generate_recorded_dbdsqr.py OUTPUT [FIXED_VT_OUTPUT]")
    source = (Path(__file__).resolve().parents[1] / "src/cuda/third_party/"
              "lapack_3_12_small_dgesdd_device.inc")
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
    if len(sys.argv) == 3:
        Path(sys.argv[2]).write_text(fixed_vt_work_pair(header + body + "\n"))
    print("generated", sys.argv[1], "bytes", len(header + body))


if __name__ == "__main__":
    main()
