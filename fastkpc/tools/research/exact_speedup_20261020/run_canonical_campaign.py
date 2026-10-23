"""Sequential, fresh-process canonical trials; never overlap GPU workloads.

Freeze the R entry script before launching it: R --file may read it in chunks
around long-running expressions. Each run saves a .pending receipt before
offline gates, so a harness error cannot discard a completed GPU result.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    p=argparse.ArgumentParser()
    p.add_argument("directory")
    args=p.parse_args()
    directory=Path(args.directory).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    snapshot=directory/"canonical-runner-frozen.R"
    if snapshot.exists(): raise RuntimeError("campaign snapshot already exists")
    shutil.copyfile(Path(__file__).with_name("run_pinned_optimizer_ab.R"),snapshot)
    digest=hashlib.sha256(snapshot.read_bytes()).hexdigest()
    native=Path("fastkpc/build/fastkpc_cuda.so")
    native_digest=hashlib.sha256(native.read_bytes()).hexdigest()
    runs=[("canonical-default", "0", None, None),
          ("canonical-queues32", "0", "32", "canonical-default"),
          ("canonical-pinned-queues32", "1", "32", "canonical-queues32")]
    for name,pinned,connections,reference in runs:
        if hashlib.sha256(snapshot.read_bytes()).hexdigest()!=digest:
            raise RuntimeError("frozen campaign script changed")
        if hashlib.sha256(native.read_bytes()).hexdigest()!=native_digest:
            raise RuntimeError("campaign native changed between trials")
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES="0",OPENBLAS_NUM_THREADS="1",
                   OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
        env.pop("CUDA_DEVICE_MAX_CONNECTIONS",None)
        env.pop("CUDA_DEVICE_MAX_COPY_CONNECTIONS",None)
        if connections: env["CUDA_DEVICE_MAX_CONNECTIONS"]=connections
        output=directory/(name+".rds")
        command=["Rscript",str(snapshot),pinned,"canonical",str(output)]
        if reference: command.append(str(directory/(reference+".rds")))
        print(json.dumps(dict(stage="start",name=name,pinned=pinned,
                              connections=connections,snapshot_sha256=digest,
                              native_sha256=native_digest)),flush=True)
        with (directory/(name+".log")).open("x") as log:
            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        if hashlib.sha256(native.read_bytes()).hexdigest()!=native_digest:
            raise RuntimeError("campaign native changed during a trial")
        print(json.dumps(dict(stage="passed",name=name)),flush=True)


if __name__=="__main__":main()
