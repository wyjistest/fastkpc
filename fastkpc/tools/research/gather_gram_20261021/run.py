"""Path (i) step 0: three fields from wide prefill batches, gathered into the
production buffers in the ORIGINAL cache-miss grouping/order, then the verbatim
production Gram + self-moment stage.  Compares every field byte for byte with
the accepted-native cache payload capture."""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from collections import defaultdict, Counter
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("frozen", type=Path, help="frozen default-Inf capture dir (capture.json + raw/finalize-*)")
parser.add_argument("native_anchor", type=Path, help="native payload capture dir (capture.json + raw/*.json,*.f64)")
parser.add_argument("build", type=Path)
parser.add_argument("out", type=Path)
parser.add_argument("--smoke", type=int, default=0, help="restrict to the first N original miss cohorts (sanitizer smoke)")
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
native = Path("fastkpc/build/fastkpc_cuda.so").resolve()
binding = json.loads((args.build / "build.json").read_text())
assert sha(native) == binding["native_sha256"]
r_library = Path(subprocess.check_output(["R", "RHOME"], text=True).strip()) / "lib"
if str(r_library) not in os.getenv("LD_LIBRARY_PATH", "").split(":"):
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = str(r_library) + ":" + environment.get("LD_LIBRARY_PATH", "")
    os.execve(sys.executable, [sys.executable, *sys.argv], environment)
ct.CDLL(str(r_library / "libR.so"), mode=ct.RTLD_GLOBAL)
ct.CDLL(str(native), mode=ct.RTLD_GLOBAL)
lib = ct.CDLL(str((args.build / "probe_gather.so").resolve()))
ptr = ct.POINTER(ct.c_double)
lib.gather_probe_create.argtypes = [ct.c_int] * 4 + [ct.c_ulonglong]
lib.gather_probe_create.restype = ct.c_void_p
lib.gather_probe_run_full.argtypes = [ct.c_void_p, ptr, ct.c_int, ptr, ct.POINTER(ct.c_int)]
lib.gather_probe_run_full.restype = ct.c_int
lib.gather_probe_run_gram.argtypes = [ct.c_void_p, ptr, ptr, ct.c_int, ptr]
lib.gather_probe_run_gram.restype = ct.c_int
lib.gather_probe_destroy.argtypes = [ct.c_void_p]
lib.gather_probe_snapshot.argtypes = [ct.POINTER(ct.c_longlong)]
def snapshot():
    s = (ct.c_longlong * 4)()
    assert lib.gather_probe_snapshot(s) == 0
    return list(s)

# ---- frozen residuals (same loader as the payload gate) ----
frozen_binding = json.loads((args.frozen / "capture.json").read_text())
assert frozen_binding["native_sha256"] == binding["native_sha256"]
records, ordered, by_key = [], [], {}
for path in sorted((args.frozen / "raw").glob("finalize-*.json")):
    m = json.loads(path.read_text())
    n, q, count = (m[x] for x in ("n", "q", "targets"))
    a = np.fromfile(path.with_suffix(".f64"), dtype="<f8")
    assert len(a) == n * q + q * count + 3 * n * count and n == 351
    r = a[-n * count:].reshape((n, count), order="F")
    for j, key in enumerate(m["keys"]):
        if key not in by_key:
            by_key[key] = len(ordered)
            ordered.append(dict(key=key, q=q, residual=r[:, j].copy()))
        else:
            assert ordered[by_key[key]]["residual"].tobytes() == r[:, j].tobytes()
    records.append(dict(source=path.stem, q=q))
assert len(ordered) == 6632 and len(records) == 2592
n, k, capacity = 351, 35, 22
entry_values = n * k + k + 2
fields = dict(vectors=slice(0, n * k), selected_values=slice(n * k, n * k + k),
              total=slice(n * k + k, n * k + k + 1), self_moment=slice(-1, None))

# ---- native anchor: actual cache payloads and original miss cohorts ----
capture_binding = json.loads((args.native_anchor / "capture.json").read_text())
assert capture_binding["native_sha256"] == binding["native_sha256"]
assert all(capture_binding["gates"].values())
native_metadata, native_groups = {}, defaultdict(list)
reference = np.empty((len(ordered), entry_values), dtype=np.float64)
for path in sorted((args.native_anchor / "raw").glob("*.json")):
    m = json.loads(path.read_text())
    i = by_key[m["key"]]
    assert m["configured_lanes"] == 8 and m["n"] == n and m["num_col"] == k
    assert path.with_suffix(".residual.f64").read_bytes() == ordered[i]["residual"].tobytes()
    native_metadata[i] = m
    native_groups[m["request"]].append(i)
    payload = np.fromfile(path.with_suffix(".f64"), dtype="<f8")
    assert payload.size == entry_values
    reference[i] = payload
assert len(native_metadata) == len(ordered)
groups = []
for request, ids in native_groups.items():
    ids.sort(key=lambda i: native_metadata[i]["component_position"])
    assert len(ids) == native_metadata[ids[0]]["component_misses"]
    assert [native_metadata[i]["component_position"] for i in ids] == list(range(len(ids)))
    groups.append(ids)
assert len(groups) == 2592 and sum(len(g) for g in groups) == 6632
if args.smoke:
    # Sanitizer smoke: a prefix of cohorts covering every original miss width present in it.
    groups = groups[:args.smoke]
    keep = sorted({i for g in groups for i in g})
    remap = {old: new for new, old in enumerate(keep)}
    ordered = [ordered[i] for i in keep]
    reference = reference[keep]
    native_metadata = {remap[i]: m for i, m in native_metadata.items() if i in remap}
    groups = [[remap[i] for i in g] for g in groups]
miss_histogram = Counter(len(g) for g in groups)

before = snapshot()
probe = lib.gather_probe_create(n, k, capacity, 8, 11811160064)
assert probe
started = time.monotonic()
full_batches = gram_batches = 0
def run_full(ids):
    global full_batches
    x = np.asfortranarray(np.column_stack([ordered[i]["residual"] for i in ids]))
    result = np.empty((len(ids), entry_values), dtype=np.float64)
    lanes = ct.c_int()
    assert lib.gather_probe_run_full(probe, x.ctypes.data_as(ptr), len(ids), result.ctypes.data_as(ptr), ct.byref(lanes)) == 0
    assert lanes.value == min(8, len(ids)) and np.isfinite(result).all()
    full_batches += 1
    return result
def run_gram(ids, source):
    global gram_batches
    vec = np.ascontiguousarray(np.concatenate([source[i, fields["vectors"]] for i in ids]))
    sel = np.ascontiguousarray(np.concatenate([source[i, fields["selected_values"]] for i in ids]))
    out = np.empty(len(ids), dtype=np.float64)
    assert lib.gather_probe_run_gram(probe, vec.ctypes.data_as(ptr), sel.ctypes.data_as(ptr), len(ids), out.ctypes.data_as(ptr)) == 0
    gram_batches += 1
    return out
bits = lambda a: a.view("u8")
report = dict(schema="gather-gram-step0-v1", native_sha256=binding["native_sha256"],
              build_sha256=sha(args.build / "build.json"), frozen_capture_sha256=sha(args.frozen / "capture.json"),
              native_capture_sha256=sha(args.native_anchor / "capture.json"),
              original_component_miss_histogram=dict(sorted(miss_histogram.items())),
              unique_targets=len(ordered), original_miss_cohorts=len(groups), capacity=capacity, configured_lanes=8)
try:
    # Phase 1: wide prefill (22-wide, natural order) producing all four fields.
    prefill = np.empty_like(reference)
    order = list(range(len(ordered)))
    for offset in range(0, len(order), capacity):
        ids = order[offset:offset + capacity]
        prefill[ids] = run_full(ids)
    phase1 = {}
    for field, section in fields.items():
        mism = [i for i in range(len(ordered)) if not np.array_equal(bits(prefill[i, section]), bits(reference[i, section]))]
        phase1[field] = dict(mismatching_targets=len(mism),
                             by_original_miss_width=dict(Counter(native_metadata[i]["component_misses"] for i in mism)),
                             max_abs=float(max((np.max(np.abs(prefill[i, section] - reference[i, section])) for i in mism), default=0.0)))
    report["phase1_wide_prefill_vs_native"] = phase1
    print("phase 1 (22-wide prefill vs native):", {f: v["mismatching_targets"] for f, v in phase1.items()}, flush=True)

    # Phase 2: gather prefilled three fields in the ORIGINAL miss grouping/order, run Gram.
    def phase_gather(label):
        gathered = np.empty(len(ordered), dtype=np.float64)
        for ids in groups:
            gathered[ids] = run_gram(ids, prefill)
        mism = [i for i in range(len(ordered)) if not np.array_equal(bits(gathered[i:i+1]), bits(reference[i, fields["self_moment"]]))]
        result = dict(mismatching_targets=len(mism),
                      by_original_miss_width=dict(Counter(native_metadata[i]["component_misses"] for i in mism)),
                      max_abs=float(max((abs(gathered[i] - reference[i, -1]) for i in mism), default=0.0)))
        report[label] = result
        print(label + ":", result, flush=True)
        return gathered
    g1 = phase_gather("phase2_gather_original_cohorts_vs_native")
    # Phase 3: control — Gram in 22-wide grouping must reproduce the known singleton effect.
    wide = np.empty(len(ordered), dtype=np.float64)
    for offset in range(0, len(order), capacity):
        ids = order[offset:offset + capacity]
        wide[ids] = run_gram(ids, prefill)
    mism = [i for i in range(len(ordered)) if not np.array_equal(bits(wide[i:i+1]), bits(reference[i, fields["self_moment"]]))]
    report["phase3_control_wide_gram_vs_native"] = dict(mismatching_targets=len(mism),
        by_original_miss_width=dict(Counter(native_metadata[i]["component_misses"] for i in mism)),
        max_abs=float(max((abs(wide[i] - reference[i, -1]) for i in mism), default=0.0)),
        wide_gram_equals_wide_prefill_self_moment=int(np.array_equal(bits(wide), bits(prefill[:, -1]))))
    print("phase 3 (control, 22-wide gram):", report["phase3_control_wide_gram_vs_native"], flush=True)
    # Phase 4: repeat phase 2 for determinism, plus reversed cohort processing order.
    g2 = phase_gather("phase4_repeat_gather_original_cohorts_vs_native")
    groups.reverse()
    g3 = phase_gather("phase5_reversed_cohort_order_vs_native")
    report["gather_repeat_bitwise_identical"] = int(np.array_equal(bits(g1), bits(g2)) and np.array_equal(bits(g1), bits(g3)))
finally:
    lib.gather_probe_destroy(probe)
after = snapshot()
assert before == after and sha(native) == binding["native_sha256"]
report["smoke_cohorts"] = args.smoke
passed = (all(v["mismatching_targets"] == 0 for f, v in report["phase1_wide_prefill_vs_native"].items() if f != "self_moment")
          and report["phase2_gather_original_cohorts_vs_native"]["mismatching_targets"] == 0
          and report["phase4_repeat_gather_original_cohorts_vs_native"]["mismatching_targets"] == 0
          and report["phase5_reversed_cohort_order_vs_native"]["mismatching_targets"] == 0
          and report["gather_repeat_bitwise_identical"] == 1)
report.update(status="PASS" if passed else "STOP_GATHER_GRAM_BITS", full_batches=full_batches, gram_batches=gram_batches,
              resources_before=before, resources_after=after, elapsed_diagnostic_only=time.monotonic() - started,
              environment={x: os.getenv(x, "") for x in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_MAX_CONNECTIONS")},
              shadow_policy="explicit diagnostic H2D/D2H of fields; CPU compares bytes only; source-shared probe, not native integration")
(args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(report["status"], "full batches", full_batches, "gram batches", gram_batches, "elapsed %.1f s" % report["elapsed_diagnostic_only"], flush=True)
