"""Compare every field of real-residual cache payloads across production shapes."""
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
parser.add_argument("capture", type=Path)
parser.add_argument("build", type=Path)
parser.add_argument("out", type=Path)
parser.add_argument("--full", action="store_true")
parser.add_argument("--prepare-capture", action="store_true")
parser.add_argument("--native-anchor",type=Path)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
native = Path("fastkpc/build/fastkpc_cuda.so").resolve()
binding = json.loads((args.build / "build.json").read_text())
assert sha(native) == binding["native_sha256"]
r_library = Path(subprocess.check_output(["R","RHOME"],text=True).strip()) / "lib"
if str(r_library) not in os.getenv("LD_LIBRARY_PATH", "").split(":"):
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = str(r_library)+":"+environment.get("LD_LIBRARY_PATH","")
    os.execve(sys.executable, [sys.executable, *sys.argv], environment)
ct.CDLL(str(r_library/"libR.so"), mode=ct.RTLD_GLOBAL)
ct.CDLL(str(native), mode=ct.RTLD_GLOBAL)
lib = ct.CDLL(str((args.build / "probe.so").resolve()))
ptr = ct.POINTER(ct.c_double)
lib.payload_probe_create.argtypes = [ct.c_int]*4+[ct.c_ulonglong]
lib.payload_probe_create.restype = ct.c_void_p
lib.payload_probe_run.argtypes = [ct.c_void_p, ptr, ct.c_int, ptr, ct.POINTER(ct.c_int)]
lib.payload_probe_run.restype = ct.c_int
lib.payload_probe_destroy.argtypes = [ct.c_void_p]
lib.payload_probe_snapshot.argtypes = [ct.POINTER(ct.c_longlong)]
def snapshot():
    s = (ct.c_longlong*4)()
    assert lib.payload_probe_snapshot(s) == 0
    return list(s)

records, ordered, by_key = [], [], {}
for path in sorted((args.capture / "raw").glob("finalize-*.json")):
    m = json.loads(path.read_text())
    n, q, count = (m[x] for x in ("n", "q", "targets"))
    a = np.fromfile(path.with_suffix(".f64"), dtype="<f8")
    expected = n*q + q*count + 3*n*count
    assert len(a) == expected and n == 351
    r = a[-n*count:].reshape((n,count), order="F")
    ids = []
    for j,key in enumerate(m["keys"]):
        if key not in by_key:
            by_key[key] = len(ordered)
            ordered.append(dict(key=key,q=q,residual=r[:,j].copy(),source=path.stem))
        else:
            assert ordered[by_key[key]]["residual"].tobytes() == r[:,j].tobytes()
        ids.append(by_key[key])
    records.append(dict(source=path.stem, q=q, ids=ids, sha256=sha(path),
                        payload_sha256=sha(path.with_suffix(".f64"))))
assert len(ordered) == 6632 and len(records) == 2592
if args.prepare_capture:
    (args.out/"keys.txt").write_text("".join(x["key"]+"\n" for x in ordered))
    (args.out/"raw").mkdir(exist_ok=True)
    print("prepared capture of",len(ordered),"actual component keys")
    sys.exit(0)
if args.full:
    selected = list(range(len(ordered)))
else:
    selected = []
    for q in sorted({x["q"] for x in ordered}):
        selected += [i for i,x in enumerate(ordered) if x["q"] == q][:44]
selected_set = set(selected)
n, k, capacity = 351, 35, 22
entry_values = n*k+k+2
fields = dict(vectors=slice(0,n*k), selected_values=slice(n*k,n*k+k),
              total=slice(n*k+k,n*k+k+1), self_moment=slice(-1,None))
before = snapshot()
probe = lib.payload_probe_create(n,k,capacity,8,11811160064)
assert probe
rows, witnesses, reference = [], [], {}
changed_keys = {field:set() for field in fields}
maximum_error = {field:0.0 for field in fields}
native_metadata, native_groups = {}, defaultdict(list)
if args.native_anchor:
    capture_binding = json.loads((args.native_anchor/"capture.json").read_text())
    assert capture_binding["native_sha256"] == binding["native_sha256"]
    assert all(capture_binding["gates"].values())
    for path in sorted((args.native_anchor/"raw").glob("*.json")):
        m = json.loads(path.read_text())
        i = by_key[m["key"]]
        assert m["configured_lanes"] == 8 and m["n"] == n and m["num_col"] == k
        assert path.with_suffix(".residual.f64").read_bytes() == ordered[i]["residual"].tobytes()
        native_metadata[i] = m
        native_groups[m["request"]].append(i)
        if i in selected_set:
            reference[i] = np.fromfile(path.with_suffix(".f64"),dtype="<f8")
            assert reference[i].size == entry_values
    assert len(native_metadata) == len(ordered) and set(reference) == selected_set
active_peak, batches = 0, 0
started = time.monotonic()
def gpu(ids):
    global active_peak, batches
    x = np.asfortranarray(np.column_stack([ordered[i]["residual"] for i in ids]))
    result = np.empty((len(ids),entry_values), dtype=np.float64)
    lanes = ct.c_int()
    assert lib.payload_probe_run(probe,x.ctypes.data_as(ptr),len(ids),
                                 result.ctypes.data_as(ptr),ct.byref(lanes)) == 0
    assert lanes.value == min(8,len(ids)) and np.isfinite(result).all()
    active_peak = max(active_peak,lanes.value)
    batches += 1
    return result
def compare(label, ids):
    values = gpu(ids)
    mismatches = {field:0 for field in fields}
    mismatch_origins = Counter()
    for j,i in enumerate(ids):
        ref = reference[i]
        for field,section in fields.items():
            a,b = values[j,section],ref[section]
            exact = np.array_equal(a.view("u8"),b.view("u8"))
            mismatches[field] += not exact
            if not exact:
                changed_keys[field].add(ordered[i]["key"])
                maximum_error[field] = max(maximum_error[field],float(np.max(np.abs(a-b))))
                mismatch_origins[native_metadata.get(i,{}).get("component_misses",-1)] += 1
            if not exact and len(witnesses) < 20:
                witnesses.append(dict(shape=label,width=len(ids),key=ordered[i]["key"],
                  q=ordered[i]["q"],field=field,max_abs=float(np.max(np.abs(a-b))),
                  different_values=int(np.count_nonzero(a.view("u8")!=b.view("u8")))))
    rows.append(dict(shape=label,width=len(ids),targets=len(ids),**mismatches,
                     original_miss_width_of_changes=dict(mismatch_origins)))
try:
    if args.native_anchor:
        # Reproduce actual cache-MISS cohorts, not the larger residual cohorts.
        for ids in native_groups.values():
            ids.sort(key=lambda i:native_metadata[i]["component_position"])
            assert len(ids) == native_metadata[ids[0]]["component_misses"]
            if not any(i in selected_set for i in ids):
                continue
            v = gpu(ids)
            for j,i in enumerate(ids):
                if i in selected_set:
                    assert np.array_equal(reference[i].view("u8"),v[j].view("u8")), (
                        "accepted native original-miss-cohort anchor differs",i)
    else:
        # Preliminary source-only comparison; not a native payload anchor.
        for rec in records:
            if not any(i in selected_set for i in rec["ids"]):
                continue
            v = gpu(rec["ids"])
            for j,i in enumerate(rec["ids"]):
                if i not in selected_set:
                    continue
                if i in reference:
                    assert np.array_equal(reference[i].view("u8"),v[j].view("u8")), (
                        "original-cohort repeated payload differs",rec["source"],i)
                else:
                    reference[i] = v[j].copy()
    assert set(reference) == selected_set
    print("anchored",len(reference),"real targets; batches",batches,flush=True)
    # Distinct real targets across S; no padding or duplicated residual columns.
    for width in ((22,) if args.full else (1,2,3,7,8,9,16,17,19,21,22)):
        for direction,order in (("natural",selected),("reverse",selected[::-1])):
            for offset in range(0,len(order),width):
                compare(f"width{width}_{direction}",order[offset:offset+width])
        print("width",width,"complete; mismatching fields",
              sum(sum(r[f] for f in fields) for r in rows),flush=True)
    for repeat in range(2):
        for offset in range(0,len(selected),22):
            compare(f"width22_repeat{repeat}",selected[offset:offset+22])
finally:
    lib.payload_probe_destroy(probe)
after = snapshot()
assert before == after and sha(native) == binding["native_sha256"]
totals = {f:sum(r[f] for r in rows) for f in fields}
report = dict(schema="component-payload-gate-v1", status="PASS" if not any(totals.values()) else "STOP_PAYLOAD_BITS",
  native_sha256=binding["native_sha256"], build_sha256=sha(args.build/"build.json"),
  capture_sha256=sha(args.capture/"capture.json"),
  native_capture_sha256=sha(args.native_anchor/"capture.json") if args.native_anchor else None,
  original_component_miss_histogram=dict(Counter(m["component_misses"] for m in native_metadata.values())),
  scope="all consumed subset22 default-Inf residuals" if args.full else "four-depth smoke",
  unique_targets=len(selected), q=sorted({ordered[i]["q"] for i in selected}),
  comparisons=sum(r["targets"] for r in rows), batches=batches,
  configured_lanes=8, active_lane_peak=active_peak, capacity=22,
  cache_budget_bytes=11811160064, fields=totals, witnesses=witnesses,
  unique_changed_keys={f:sorted(keys) for f,keys in changed_keys.items()},
  max_abs_error=maximum_error,
  resources_before=before,resources_after=after, rows=rows,
  elapsed_diagnostic_only=time.monotonic()-started,
  environment={x:os.getenv(x,"") for x in ("CUDA_VISIBLE_DEVICES","CUDA_DEVICE_MAX_CONNECTIONS")},
  input_identity=[{x:rec[x] for x in ("source","q","sha256","payload_sha256")} for rec in records],
  shadow_policy="explicit diagnostic D2H of complete payload; CPU compares bytes only; source-shared probe, not native integration qualification")
(args.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
print(report["status"],totals,"comparisons",report["comparisons"],flush=True)
