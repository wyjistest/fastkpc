"""Nsight-only 640-component replay after warmup, with an 11 GiB footprint."""
import ctypes as ct
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np

spec=importlib.util.spec_from_file_location("replay",Path(__file__).with_name("benchmark.py"))
b=importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
work=Path(sys.argv[1])
ref=b.Reference(work/"dense-reference.so")
profiler=ct.CDLL("/usr/local/cuda/lib64/libcudart.so")
nvtx=ct.CDLL("/usr/local/cuda/targets/x86_64-linux/lib/libnvToolsExt.so.1")
nvtx.nvtxRangePushA.argtypes=[ct.c_char_p]
x=np.fromfile(work/"residuals.f64",dtype="<f8").reshape(640,351)
reservation=ref.reserve(11*2**30)
contexts=[]
try:
    for _ in range(10):
        contexts.append(b.gpu.GpuComponents(work/"structured-eig.so",351,35,64,170662))
    xs=[np.ascontiguousarray(x[i:i+64]) for i in range(0,640,64)]
    baseline=[c.run(chunk)[0] for c,chunk in zip(contexts,xs)]
    signature=b.output_bits([p for parts in baseline for p in parts])
    walls=[]
    assert profiler.cudaProfilerStart()==0
    for trial in range(3):
        nvtx.nvtxRangePushA(f"structured.real640.trial{trial}".encode())
        start=time.perf_counter()
        with ThreadPoolExecutor(max_workers=10) as pool:
            futures=[pool.submit(c.run,chunk) for c,chunk in zip(contexts,xs)]
            runs=[f.result() for f in futures]
        walls.append(time.perf_counter()-start)
        nvtx.nvtxRangePop()
        assert b.output_bits([p for run in runs for p in run[0]])==signature
    assert profiler.cudaProfilerStop()==0
finally:
    for c in contexts:
        c.close()
    ref.release(reservation,11*2**30)
(work/"profile-run.json").write_text(json.dumps(dict(
    schema="structured-real640-nsys-run-v1",profiled=True,
    performance_evidence=False,walls_s=walls,output_sha256=signature,
    sources={str(Path(__file__)):b.sha(Path(__file__))}),indent=2))
