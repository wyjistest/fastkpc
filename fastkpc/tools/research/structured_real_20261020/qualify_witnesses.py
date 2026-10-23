"""Untimed deeper-level and three-real-input eig qualification."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

spec = importlib.util.spec_from_file_location("replay",Path(__file__).with_name("benchmark.py"))
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
work = Path(sys.argv[1])
smoke = "--sanitizer-smoke" in sys.argv
ref = b.Reference(work/"dense-reference.so")
rows = []
for csv_path in sorted((work/"witnesses").glob("*.csv")):
    n = int(csv_path.stem)
    cases = list(csv.DictReader(csv_path.open()))
    x = np.fromfile(csv_path.with_suffix(".f64"),dtype="<f8").reshape(-1,n)
    if smoke:
        x,cases = x[:2],cases[:1]
    dense,dt = ref.dense(x,1)
    fast,st = b.structured(work/"structured-eig.so",x,170662,1,ref)
    pairs = [(i,i+1) for i in range(0,len(x),2)]
    actual = b.pair_values(fast,pairs,n)
    oracle = np.array([[float(r[k]) for k in ("statistic","mean","variance","authority_p")] for r in cases])
    production = np.array([[float(r[k]) for k in ("gpu_statistic","gpu_mean","gpu_variance","gpu_p")] for r in cases])
    gates = dict(versus_kpcalg=b.compare(actual,oracle),
                 versus_gpu=b.compare(actual,production),
                 versus_dense=b.compare(actual,b.pair_values(dense,pairs,n)))
    row = dict(n=n,pairs=len(cases),ids=[r["id"] for r in cases],
               rotations_max=st["rotations_max"],gates=gates,
               deterministic=st["bitwise_deterministic"] and dt["bitwise_deterministic"],
               pass_=all(g["pass_"] for g in gates.values()))
    rows.append(row)
    print(json.dumps(row),flush=True)
bad = b.gpu.GpuComponents(work/"structured-eig.so",351,35,1,1)
try:
    overflow = bad.run(np.fromfile(work/"residuals.f64",dtype="<f8")[:351][None,:],expect_failure=True)
finally:
    bad.close()
name = "sanitizer-smoke.json" if smoke else "witness-qualification.json"
(work/name).write_text(json.dumps(dict(rows=rows,overflow=overflow,
    pass_=all(r["pass_"] for r in rows),
    source_sha256={str(Path(__file__)):b.sha(Path(__file__))}),indent=2))
if not all(r["pass_"] for r in rows):
    raise SystemExit("STOP: exact witness numerical gate failed")
