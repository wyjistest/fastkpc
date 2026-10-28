"""Source-specific, non-numerical audit of fixed-VT recorded DBDSQR scratch."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]
PARENT = ROOT / "fastkpc/artifacts/full_cuda_ci/optimizer_l3_slot_runtime_20261028"
ACTIVITY = ROOT / "fastkpc/artifacts/full_cuda_ci/stage_b_attribution_20261028"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def compact(value):
    return re.sub(r"\s+", "", value)


def line_at(text, offset):
    return text.count("\n", 0, offset) + 1


def calls(text, name):
    """Balanced arguments in this fixed generated function (no C++ parsing)."""
    result = []
    for match in re.finditer(r"\b" + name + r"\(", text):
        begin = match.end()
        pos, depth = begin, 1
        while depth:
            if text[pos] == "(":
                depth += 1
            elif text[pos] == ")":
                depth -= 1
            pos += 1
        args, start, depth = [], begin, 0
        for i in range(begin, pos - 1):
            if text[i] in "([":
                depth += 1
            elif text[i] in ")]":
                depth -= 1
            elif text[i] == "," and depth == 0:
                args.append(compact(text[start:i]))
                start = i + 1
        args.append(compact(text[start:pos - 1]))
        result.append({"line": line_at(text, match.start()), "args": args})
    return result


BRANCHES = [
    ("zero_forward", "F", ("cs", "sn", "oldcs", "oldsn"), 0),
    ("zero_backward", "B", ("cs", "-sn", "oldcs", "-oldsn"), 2),
    ("shifted_forward", "F", ("cosr", "sinr", "cosl", "sinl"), 0),
    ("shifted_backward", "B", ("cosr", "-sinr", "cosl", "-sinl"), 2),
]


def symbolic_sweep(reference, packed, p, width, branch, epoch):
    """Only symbolic tokens: no floating-point arithmetic or QR iteration."""
    name, direction, expressions, selected = branch
    order = range(width) if direction == "F" else range(width - 1, -1, -1)
    for j in order:
        for segment, expression in enumerate(expressions):
            token = (epoch, name, j, expression)
            reference[segment * p + j] = token
            if selected <= segment < selected + 2:
                packed[(segment - selected) * p + j] = token
    reads = range(width) if direction == "F" else range(width - 1, -1, -1)
    for j in reads:
        old = (reference[selected * p + j], reference[(selected + 1) * p + j])
        new = (packed[j], packed[p + j])
        assert old == new
        assert all(token[0] == epoch for token in new), "read before current sweep write"


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: audit.py NEW_OUTPUT_DIRECTORY")
    out = Path(sys.argv[1]).resolve()
    assert not out.exists(), "preserve existing evidence"
    build = json.loads((PARENT / "build.json").read_text())
    closure = json.loads((ACTIVITY / "closure.json").read_text())
    native = ROOT / "fastkpc/build/fastkpc_cuda.so"
    assert sha(native) == build["native_sha256"] == closure["native_sha256"]
    assert len(build["source_sha256"]) == 103
    assert all(sha(ROOT / p) == h for p, h in build["source_sha256"].items())

    kernel_path = ROOT / "fastkpc/src/cuda/full_cuda_ci_structured_eig.cu"
    generated_path = ROOT / "fastkpc/build/recorded_dbdsqr.inc"
    vendor_path = ROOT / "fastkpc/src/cuda/third_party/lapack_3_12_small_dgesdd_device.inc"
    generator_path = ROOT / "fastkpc/tools/generate_recorded_dbdsqr.py"
    kernel, generated, vendor = [p.read_text() for p in (kernel_path, generated_path, vendor_path)]
    assert sha(generated_path) == build["generator_output_sha256"]
    begin = vendor.index("__device__ int dbdsqr_(")
    end = vendor.index("} /* dbdsqr_ */", begin) + len("} /* dbdsqr_ */")
    body = vendor[begin:end].replace("dbdsqr_(", "recorded_dbdsqr_(", 1)
    assert body.count("ftnlen uplo_len)") == 1
    body = body.replace("ftnlen uplo_len)", "ftnlen uplo_len, RotationLog* log)")
    for name in ("drot_", "dlasr_", "dscal_", "dswap_"):
        body = body.replace(name + "(", "record_" + name + "(log, ")
    regenerated = ("// Generated from the repository's LAPACK 3.12 device port.\n"
                   "// Original LAPACK license and authorship remain applicable.\n"
                   "// Source SHA256: " + sha(vendor_path) + "\n" + body + "\n")
    assert regenerated == generated
    caller = compact(kernel[kernel.index("__global__ void bidiagonal"):kernel.index("__device__ double secular_value")])
    assert 'intr=z.m-1,one=1,zero=0,info=0;' in caller
    assert 'recorded_dbdsqr_(const_cast<char*>("L"),&r,&one,&zero,&zero,d,e,q,&r,dummy+2,&one,dummy+2,&one,work,&info,1,&log);' in caller
    assert 'if(threadIdx.x!=0)return;' in caller
    assert 'double*d=local;double*e=d+n;double*q=e+n;double*work=q+n;' in caller

    stores = [{"line": line_at(generated, m.start()), "index": compact(m[1]), "value": compact(m[2])}
              for m in re.finditer(r"\bwork\s*\[([^]]+)\]\s*=\s*([^;]+);", generated)]
    expected = [("i__", "cs"), ("nm1+i__", "sn")]
    for _, direction, expressions, _ in BRANCHES:
        index = "i__-ll+1" if direction == "F" else "i__-ll"
        expected.extend((index + suffix, value) for suffix, value in
                        zip(("", "+nm1", "+nm12", "+nm13"), expressions))
    assert [(s["index"], s["value"]) for s in stores] == expected
    consumers = calls(generated, "record_dlasr_")
    assert len(consumers) == 14
    live = [c for c in consumers if c["args"][5] == "ncvt"]
    assert len(live) == 4
    for c, (_, direction, _, selected) in zip(live, BRANCHES):
        pointers = ["&work[1]", "&work[*n]"] if selected == 0 else ["&work[nm12+1]", "&work[nm13+1]"]
        assert c["args"][1:4] == ['"L"', '"V"', '"' + direction + '"']
        assert c["args"][6:8] == pointers
    for c in consumers:
        c["reachable_fixed_vt"] = c["args"][5] == "ncvt"
    work_references = list(re.finditer(r"\bwork\s*\[([^]]+)\]", generated))
    pointers = list(re.finditer(r"&work\s*\[([^]]+)\]", generated))
    assert len(work_references) == 47 and len(stores) == 18 and len(pointers) == 29
    assert len(calls(generated, "dlasq1_")) == 1
    assert "rotate=*ncvt>0||*nru>0||*ncc>0;" in compact(generated)
    inventory = {"stores": stores, "rotation_consumers": consumers,
                 "work_reference_count": 47, "store_count": 18,
                 "pointer_reference_count": 29, "unreachable_dqds_work_consumers": 1,
                 "scope": "Exact source inventory; reachability is justified in the accompanying human audit."}

    # Exhaust all canonical dimensions and active widths. ll affects neither
    # prefix bounds nor operand order once width=M-ll is fixed.
    address_cases = 0
    for r in range(2, 351):
        p = r - 1
        for width in range(1, p + 1):
            assert 0 <= width - 1 < p
            for _, _, _, selected in BRANCHES:
                assert selected * p <= selected * p + width - 1 < (selected + 1) * p
                assert (selected + 1) * p + width - 1 < 4 * p
                assert p + width - 1 < 2 * p <= 2 * 351
                address_cases += 1
    token_cases, token_reads = 0, 0
    for width in range(1, 350):
        for branch in BRANCHES:
            symbolic_sweep([None] * (4 * width), [None] * (2 * width), width, width, branch, 0)
            token_cases += 1
            token_reads += 2 * width
    reference, packed = [None] * (4 * 349), [None] * (2 * 349)
    transitions = [(BRANCHES[0], 349), (BRANCHES[1], 3), (BRANCHES[2], 27),
                   (BRANCHES[3], 2), (BRANCHES[1], 348), (BRANCHES[0], 1)]
    for epoch, (branch, width) in enumerate(transitions):
        symbolic_sweep(reference, packed, 349, width, branch, epoch)
    alias_counterexamples = []
    for name, direction, expressions, selected in BRANCHES:
        aliased = [None, None]
        for segment, token in enumerate(expressions):
            aliased[segment % 2] = token
        expected_pair = list(expressions[selected:selected + 2])
        if aliased != expected_pair:
            alias_counterexamples.append({"branch": name, "expected": expected_pair, "blind_alias_reads": aliased})
    assert len(alias_counterexamples) == 2

    hardware = json.loads((ACTIVITY / "attempt3/trace/hardware.json").read_text())
    assert hardware["shared_per_sm"] == 102400 and hardware["reserved_shared_per_block"] == 1024
    shared_rows = []
    for name, size in [("current_7N", 7 * 351 * 8), ("trim_only_work_padding", (3 * 351 + 4 * 349) * 8),
                       ("five_block_threshold", 19456), ("threshold_plus_one", 19457),
                       ("proposed_5N", 5 * 351 * 8), ("recover_unchanged", 36 * 351 * 8)]:
        allocated = ((size + 1024 + 127) // 128) * 128
        shared_rows.append({"name": name, "dynamic_bytes": size, "allocated_bytes": allocated,
                            "shared_only_blocks_per_sm_upper": 102400 // allocated})
    assert [r["shared_only_blocks_per_sm_upper"] for r in shared_rows] == [4, 4, 5, 4, 6, 1]

    input_files = [PARENT / "build.json", ACTIVITY / "closure.json", ACTIVITY / "manifest.json",
                   ACTIVITY / "attempt3/analysis.json", ACTIVITY / "attempt3/trace/hardware.json",
                   ROOT / "fastkpc/docs/PHASE10_STAGE_B_ATTRIBUTION_REVIEW_20261028.md",
                   kernel_path, generated_path, vendor_path, generator_path,
                   Path("/usr/local/cuda/include/cuda_occupancy.h"), native]
    input_hashes = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha(p) for p in input_files}
    result = {"schema": "fixed-vt-dbdsqr-work-static-audit-v1", "status": "STATIC_WORK_LIFETIME_AUDIT_COMPLETE",
              "recommendation": "ISOLATED_FIXED_VT_WORK_PAIR_COMPACTION_QUALIFICATION",
              "gpu_execution": False, "cpu_numerical_evaluation": False, "prototype_implemented": False,
              "gpu_bitwise_qualification": "NOT_RUN", "performance_claim": False,
              "native_sha256": sha(native), "producer_head": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "production_inputs_verified": len(build["source_sha256"]), "generator_text_exact": True,
              "scope": {"N": 351, "distinct_values": [2, 351], "r": [1, 350], "uplo": "L", "ncvt": 1, "nru": 0, "ncc": 0},
              "address_cases": address_cases, "symbolic_sweep_cases": token_cases,
              "symbolic_operand_reads": token_reads, "direction_transition_cases": len(transitions),
              "blind_alias_counterexamples": alias_counterexamples, "shared_bounds": shared_rows,
              "model_limit": "Source-specific integer/token checks do not execute numerical branches or establish compiled GPU parity.",
              "accepted_development_median_unchanged": 329.078,
              "phase10": "ACTIVE", "candidate": "NOT_FROZEN", "holdout": "SEALED", "recommended": False}
    # Recheck all authenticated inputs before publishing new output.
    assert all(sha(ROOT / p) == h for p, h in build["source_sha256"].items())
    assert all(sha(Path(p) if p.startswith("/") else ROOT / p) == h for p, h in input_hashes.items())
    out.mkdir(parents=True)
    snapshots = out / "input-source"
    snapshots.mkdir()
    for p in (kernel_path, generated_path, generator_path, Path("/usr/local/cuda/include/cuda_occupancy.h")):
        shutil.copyfile(p, snapshots / p.name)
    for name, value in [("audit.json", result), ("work-accesses.json", inventory)]:
        with (out / name).open("x") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
    manifest = {"schema": "fixed-vt-dbdsqr-work-audit-manifest-v1", "inputs": input_hashes,
                "production_source_sha256": build["source_sha256"],
                "tools": {str(p.relative_to(ROOT)): sha(p) for p in sorted(Path(__file__).parent.iterdir()) if p.is_file()},
                "files": {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}}
    with (out / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "address_cases": address_cases, "symbolic_sweep_cases": token_cases,
                      "symbolic_operand_reads": token_reads, "audit_sha256": sha(out / "audit.json"),
                      "manifest_sha256": sha(out / "manifest.json")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
