"""Same inputs, isolated sequential measurements of retained eig and prototype.

Dense timing excludes distance construction/centering, so it is a lower bound
for the incumbent complete component. Prototype timing includes sort/prepare,
QR, secular roots and eigenvector recovery; diagnostic D2H is excluded from
device measurements. Neither number is an outer-method performance claim.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess

import numpy as np

spec=importlib.util.spec_from_file_location("gpu",Path(__file__).with_name("verify_structured_eig_gpu.py"))
gpu=importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("library");p.add_argument("dense_benchmark");p.add_argument("output")
    p.add_argument("--trials",type=int,default=7)
    p.add_argument("--n",type=int,default=351)
    args=p.parse_args()
    results=[]
    for count,extended in ((9,9),(16,16),(48,16)):
        dense=[]
        for lanes in (8,extended):
            text=subprocess.check_output([args.dense_benchmark,str(args.n),str(count),str(lanes),str(args.trials)],text=True)
            record=dict(x.split("=",1) for x in text.split() if "=" in x)
            assert record["eigenvalue_bit_mismatch_count"]=="0"
            assert record["eigenvector_bit_mismatch_count"]=="0"
            dense.append(record)
        row=np.arange(1,args.n+1,dtype=float)
        x=np.array([np.sin(.017*(c+1)*row)+np.cos(.0031*(c+3)*row*row) for c in range(count)])
        ctx=gpu.GpuComponents(args.library,args.n,35,count)
        try:
            baseline,_,_,_=ctx.run(x)
            runs=[]
            for _ in range(args.trials):
                parts,stages,diag,wall=ctx.run(x)
                assert all(all(np.array_equal(a,b) for a,b in zip(p,q)) for p,q in zip(parts,baseline))
                runs.append(dict(device_ms=float(sum(stages[:4])),stages_ms=stages[:4].tolist(),host_with_shadows_sec=wall))
        finally:ctx.close()
        median=statistics.median(r["device_ms"] for r in runs)
        incumbent=float(dense[0]["concurrent_device_median_ms"])
        record=dict(n=args.n,components=count,trials=args.trials,dense=dense,structured=runs,
                    structured_device_median_ms=median,incumbent_eig_only_device_median_ms=incumbent,
                    structured_speedup=incumbent/median,
                    workspace_bytes=int(stages[4]),max_rotation_count=int(diag[:,2].max()))
        results.append(record)
        print(json.dumps(record),flush=True)
    with open(args.output,"x") as f: json.dump(dict(schema="structured-eig-isolated-performance-v1",results=results),f,indent=2)


if __name__=="__main__":main()
