"""Full build of the frozen, unmodified Git checkout; never install into main."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "fastkpc/artifacts/full_cuda_ci/clean_general_inputs_20261106"
CHECKOUT = BASE / "checkout"
COMMIT = "2be6a816d3039a5f9340ba7018166d737ab11d9a"

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()

def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2)
        f.write("\n")

def git(*args):
    return subprocess.check_output(["git", "-C", str(CHECKOUT), *args], text=True).strip()

def sources():
    paths = git("ls-files", "fastkpc/src", "fastkpc/R", "kpcalg/R",
                "fastkpc/tools/build_cuda_native.sh", "fastkpc/tools/generate_recorded_dbdsqr.py")
    return {p: sha(CHECKOUT / p) for p in paths.splitlines()}

def main():
    assert git("rev-parse", "HEAD") == COMMIT
    assert not git("status", "--porcelain")
    assert not (CHECKOUT / "fastkpc/build").exists(), "full build must start without objects"
    out = BASE / "build"
    out.mkdir()
    before = sources()
    assert before["kpcalg/R/dcovgamma.R"] == "d0620dd9b06f006170b5e142b892c202619296797b2bf9ca842a7c2b5b3b715f"
    env = {k: v for k, v in os.environ.items() if not k.startswith("FASTKPC_")}
    assert not any(env.get(k) for k in ("LD_PRELOAD", "CUDA_INJECTION64_PATH", "CUDA_LAUNCH_BLOCKING"))
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    available = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, available[20:40] or available)
    command = ["sh", "-x", "fastkpc/tools/build_cuda_native.sh"]
    write(out / "protocol.json", dict(commit=COMMIT, checkout=str(CHECKOUT),
        command=command, source_sha256=before, incremental_flags={},
        parent_objects_reused=False, tools_sha256={str(Path(__file__).relative_to(ROOT)): sha(__file__)},
        affinity=sorted(os.sched_getaffinity(0))))
    started = time.time()
    print("START clean full build", COMMIT, flush=True)
    with (out / "build.log").open("x") as log:
        result = subprocess.run(command, cwd=CHECKOUT, env=env, stdout=log, stderr=subprocess.STDOUT)
    report = dict(exit_code=result.returncode, elapsed_sec=time.time()-started,
                  commit=COMMIT, clean_checkout=not git("status", "--porcelain"),
                  sources_unchanged=before == sources())
    write(out / "process.json", report)
    assert result.returncode == 0 and report["clean_checkout"] and report["sources_unchanged"]
    native = CHECKOUT / "fastkpc/build/fastkpc_cuda.so"
    objects = {p.name: sha(p) for p in sorted(native.parent.glob("*.o"))}
    assert len(objects) == 44
    commands = (out / "build.log").read_text().splitlines()
    compiles = [line for line in commands if line.startswith("+ ") and " -c " in line]
    assert len(compiles) == 44, len(compiles)
    method = [line for line in compiles if "full_cuda_ci_method_batch.cu" in line]
    assert len(method) == 1 and "--fmad=false" in method[0]
    write(out / "build.json", dict(status="PASS_CLEAN_FULL_BUILD", **report,
        native_path=str(native), native_sha256=sha(native), objects=objects,
        source_sha256=before, compile_count=len(compiles),
        generated_sha256={p: sha(native.parent/p) for p in
            ("recorded_dbdsqr.inc", "recorded_dbdsqr_fixed_vt.inc")},
        build_log_sha256=sha(out / "build.log")))
    print("PASS full build", sha(native), flush=True)

if __name__ == "__main__":
    main()
