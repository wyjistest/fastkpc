"""Build a separate candidate from authenticated objects; never replace native."""
import concurrent.futures
import importlib.util
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

TOOL = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("build_identity", TOOL.parent / "build_consistency_20261029/build.py")
identity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(identity)
ROOT = identity.ROOT
BASE = ROOT / "fastkpc/artifacts/full_cuda_ci/general_shapes_20261102"
PARENT = ROOT / "fastkpc/artifacts/full_cuda_ci/structured_rows_20261029/candidate"
FULL = ROOT / "fastkpc/artifacts/full_cuda_ci/build_consistency_20261029/full"

def main():
    tag = sys.argv[1]
    assert Path(tag).name == tag
    out = BASE / tag
    out.mkdir(parents=True)
    parent_dir = BASE / sys.argv[2] if len(sys.argv) > 2 else PARENT
    parent = identity.read(parent_dir / "build.json")
    assert identity.sha(parent_dir / "fastkpc_cuda.so") == parent["native_sha256"]
    sources = identity.source_map()
    snapshot = out / "source"
    for p in sources:
        dest = snapshot / p
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / p, dest)
    objects = out / "objects"
    objects.mkdir()
    for name, digest in parent["objects"].items():
        assert identity.sha(parent_dir / "objects" / name) == digest
        shutil.copyfile(parent_dir / "objects" / name, objects / name)
    generated = snapshot / "fastkpc/build"
    generated.mkdir()
    for name in ("recorded_dbdsqr.inc", "recorded_dbdsqr_fixed_vt.inc"):
        origin = parent_dir / name
        if not origin.exists(): origin = parent_dir / "source/fastkpc/build" / name
        shutil.copyfile(origin, generated / name)
    rebuild = {"full_cuda_ci_one_call.o", "mgcv_multi_penalty_gcv_capacity.o",
        "r_api_multi_penalty_capacity.o", "full_cuda_ci_vertical.o",
        "full_cuda_ci_structured_eig.o", "mgcv_multi_penalty_gcv_small64.o",
        "mgcv_multi_penalty_gcv_postback.o"}
    changed = {p for p, h in sources.items() if parent["source_sha256"].get(p) != h}
    if changed <= {"fastkpc/src/cuda/full_cuda_ci_vertical.cu",
        "fastkpc/src/cuda/full_cuda_ci_structured_launch.hpp",
        "fastkpc/src/cuda/full_cuda_ci_structured_eig.cu"}:
        rebuild = {"full_cuda_ci_one_call.o", "full_cuda_ci_vertical.o", "full_cuda_ci_structured_eig.o"}
    elif changed == {"fastkpc/src/full_cuda_ci_one_call.cpp"}:
        rebuild = {"full_cuda_ci_one_call.o"}
    elif changed <= {"fastkpc/src/full_cuda_ci_one_call.cpp",
                     "fastkpc/src/cuda/mgcv_multi_penalty_gcv_capacity.hpp"}:
        rebuild = {"full_cuda_ci_one_call.o", "mgcv_multi_penalty_gcv_capacity.o",
                   "r_api_multi_penalty_capacity.o"}
    commands = {}
    link = None
    old_source = str(FULL.parent / "source")
    for line in (FULL / "build.log").read_text().splitlines():
        if not line.startswith("+ "):
            continue
        args = shlex.split(line[2:])
        if "-o" not in args:
            continue
        name = Path(args[args.index("-o") + 1]).name
        if name in rebuild and "-c" in args:
            args = [s.replace(old_source, str(snapshot)) for s in args]
            args[args.index("-o") + 1] = str(objects / name)
            commands[name] = args
        elif "-shared" in args and "-Wl,-x" in args:
            link = args
    assert set(commands) == rebuild and link
    link[link.index("-o") + 1] = str(out / "fastkpc_cuda.so")
    link = [str(objects / Path(s).name) if s.endswith(".o") else s for s in link]
    identity.write(out / "protocol.json", dict(source_sha256=sources,
        parent_native_sha256=parent["native_sha256"], commands=commands, link=link,
        changed_sources=[p for p, h in sources.items() if parent["source_sha256"].get(p) != h],
        numerical_flags="reuse exact per-TU parent command", installed_native_unchanged=True))

    def compile_one(item):
        name, cmd = item
        print("BUILD", name, flush=True)
        with (out / (name + ".log")).open("x") as log:
            subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        return name

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for name in pool.map(compile_one, commands.items()):
            print("DONE", name, flush=True)
    with (out / "link.log").open("x") as log:
        subprocess.run(link, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    assert all(identity.sha(ROOT / p) == h for p, h in sources.items())
    identity.write(out / "build.json", dict(status="PASS", source_sha256=sources,
        native_path=str(out / "fastkpc_cuda.so"), native_sha256=identity.sha(out / "fastkpc_cuda.so"),
        parent_native_sha256=parent["native_sha256"], rebuilt=sorted(rebuild),
        objects={name: identity.sha(objects / name) for name in parent["objects"]}))
    print("PASS build", identity.sha(out / "fastkpc_cuda.so"), flush=True)

if __name__ == "__main__":
    main()
