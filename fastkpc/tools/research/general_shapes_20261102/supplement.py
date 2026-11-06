"""Sequential GPU adapter, dispatch-boundary and cross-method regressions."""
import importlib.util
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

TOOL = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("shape_jobs", TOOL.parent / "stage_b_work_pair_runtime_20261028/support.py")
jobs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jobs)
ROOT = jobs.ROOT
BASE = ROOT / "fastkpc/artifacts/full_cuda_ci/general_shapes_20261102"
build = jobs.read(BASE / sys.argv[1] / "build.json")
out = BASE / sys.argv[2]
out.mkdir()
reuse_slots = BASE / sys.argv[3] if len(sys.argv) > 3 else None
for key in list(os.environ):
    if key.startswith("FASTKPC_"):
        del os.environ[key]
env = jobs.environment()
native = Path(build["native_path"])
parent = ROOT / "fastkpc/artifacts/full_cuda_ci/build_consistency_20261029/full/qualification/receipt.rds"
baseline = ROOT / "fastkpc/artifacts/full_cuda_ci/structured_rows_20261029/canonical/on/on3/receipt.rds"
helpers = TOOL.parent / "build_consistency_20261029"
sources = [TOOL / n for n in ("slots.cpp", "slots.R", "run.R", "canonical.R", "supplement.py")]
sources += [helpers / "run_test.R", helpers / "methods.R", parent, baseline]
sources += [ROOT / f"fastkpc/tests/test_full_cuda_ci_l{level}_optimizer_slots.R" for level in range(3, 7)]
sources += [ROOT / "fastkpc/tests/test_general_shape_geometry.cpp"]
sources += [ROOT / f"fastkpc/tests/test_full_cuda_ci_{name}_component_prefill.R" for name in ("level", "l2")]
sources += [ROOT / "fastkpc/tests/test_full_cuda_ci_structured_prefill.R"]
dependencies = {str(p): jobs.sha(p) for p in sources + [native]}

def frozen():
    jobs.check_map(build["source_sha256"])
    jobs.check_map(dependencies)

reused_successes = []
def run(name, command, job_env):
    previous = reuse_slots / (name + "-process.json") if reuse_slots else None
    if previous and previous.exists():
        process = jobs.read(previous)
        old = jobs.read(reuse_slots / "protocol.json")
        if process["exit_code"] == 0 and process["isolated"] and old["native_sha256"] == build["native_sha256"]:
            for path in sources + [native]:
                if path != Path(__file__):
                    assert old["dependencies"][str(path)] == jobs.sha(path), str(path)
            frozen()
            for suffix in ("-process.json", "-gpu.csv", ".log"):
                shutil.copyfile(reuse_slots / (name + suffix), out / (name + suffix))
            if (reuse_slots / name).is_dir():
                shutil.copytree(reuse_slots / name, out / name, dirs_exist_ok=True)
            reused_successes.append(dict(test=name, original_directory=str(reuse_slots),
                process_sha256=jobs.sha(previous), rerun=False))
            print("REUSE PASS", name, flush=True)
            return
    jobs.job(out, name, command, job_env, frozen)

cppflags = shlex.split(subprocess.check_output(["R", "CMD", "config", "--cppflags"], text=True))
ldflags = shlex.split(subprocess.check_output(["R", "CMD", "config", "--ldflags"], text=True))
rcpp = subprocess.check_output(["Rscript", "--vanilla", "-e", 'cat(system.file("include", package="Rcpp"))'], text=True).strip()
adapter = out / "shape_slots.so"
command = ["g++", "-std=c++17", "-O2", "-shared", "-fPIC", "-I" + str(ROOT / "fastkpc/src"),
    "-I/usr/local/cuda/include", "-I" + rcpp, *cppflags, str(TOOL / "slots.cpp"), str(native),
    "-Wl,-rpath," + str(native.parent), *ldflags, "-o", str(adapter)]
jobs.write(out / "protocol.json", dict(native_sha256=build["native_sha256"],
    dependencies=dependencies, adapter_command=command,
    reused_slots=str(reuse_slots) if reuse_slots else None,
    order=["slots", "configuration_l3_l4_l5_l6", "other_methods", "cache_capacity_skip", "canonical"],
    numerical_tolerance=1e-10, performance_claim=False))
frozen()
host_test = out / "test_general_shape_geometry"
with (out / "host-geometry.log").open("x") as log:
    subprocess.run(["g++", "-std=c++17", "-I" + str(ROOT / "fastkpc/src"),
        str(ROOT / "fastkpc/tests/test_general_shape_geometry.cpp"), "-o", str(host_test)],
        stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run([str(host_test)], stdout=log, stderr=subprocess.STDOUT, check=True)
if reuse_slots:
    old_protocol = jobs.read(reuse_slots / "protocol.json")
    for path in (TOOL / "slots.cpp", TOOL / "slots.R"):
        assert old_protocol["dependencies"][str(path)] == jobs.sha(path)
    old_report = jobs.read(reuse_slots / "slots/report.json")
    old_process = jobs.read(reuse_slots / "slots-process.json")
    assert old_report["status"] == "PASS"
    if old_report["native_sha256"] != build["native_sha256"]:
        # The direct slot adapter never invokes the changed one-call scheduler.
        # Preserve its actual native SHA; authenticate every reused linked object.
        parents = [jobs.read(p) for p in BASE.glob("candidate*/build.json")]
        slot_parent = next(p for p in parents if p["native_sha256"] == old_report["native_sha256"])
        assert jobs.sha(slot_parent["native_path"]) == slot_parent["native_sha256"]
        changed_sources = [p for p, h in build["source_sha256"].items() if slot_parent["source_sha256"].get(p) != h]
        assert changed_sources == ["fastkpc/src/full_cuda_ci_one_call.cpp"]
        unchanged_objects = {p: h for p, h in build["objects"].items() if p != "full_cuda_ci_one_call.o"}
        assert all(slot_parent["objects"].get(p) == h for p, h in unchanged_objects.items())
        jobs.write(out / "slot-native-bridge.json", dict(reference_native_sha256=slot_parent["native_sha256"],
            current_native_sha256=build["native_sha256"], unchanged_objects=unchanged_objects,
            changed_sources=changed_sources, scope="direct slot/setup/optimizer APIs only; current scheduler qualified separately"))
    assert old_process["exit_code"] == 0 and old_process["isolated"]
    shutil.copytree(reuse_slots / "slots", out / "slots")
    for name in ("slots-process.json", "slots-gpu.csv", "slots.log"):
        shutil.copyfile(reuse_slots / name, out / name)
    jobs.write(out / "slots-reuse.json", dict(original_directory=str(reuse_slots),
        report_sha256=jobs.sha(reuse_slots / "slots/report.json"),
        process_sha256=jobs.sha(reuse_slots / "slots-process.json"), rerun=False))
else:
    with (out / "adapter-build.log").open("x") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    dependencies[str(adapter)] = jobs.sha(adapter)
    jobs.write(out / "adapter.json", dict(sha256=jobs.sha(adapter), command=command))
    jobs.job(out, "slots", ["Rscript", "--vanilla", str(TOOL / "slots.R"), str(native),
        str(adapter), str(out / "slots")], env, frozen)
for level in range(3, 7):
    run(f"config-l{level}", ["Rscript", "--vanilla", str(helpers / "run_test.R"),
        str(native), str(ROOT / f"fastkpc/tests/test_full_cuda_ci_l{level}_optimizer_slots.R")], env)
for name in ("level", "l2"):
    run("prefill-" + name, ["Rscript", "--vanilla", str(helpers / "run_test.R"),
        str(native), str(ROOT / f"fastkpc/tests/test_full_cuda_ci_{name}_component_prefill.R")], env)
run("prefill-structured", ["Rscript", "--vanilla", str(helpers / "run_test.R"),
    str(native), str(ROOT / "fastkpc/tests/test_full_cuda_ci_structured_prefill.R")], env)
(out / "methods").mkdir()
run("methods", ["Rscript", "--vanilla", str(helpers / "methods.R"),
    str(out / "methods"), str(native), str(parent)], env)
small_cache = env.copy()
small_cache["FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES"] = str(1024 * 1024)
run("capacity-skip", ["Rscript", "--vanilla", str(TOOL / "run.R"),
    "real300", "on", str(native), str(out / "capacity-skip"), "-"], small_cache)
skip = jobs.read(out / "capacity-skip/report.json")
assert skip["pass"] and skip["execution"]["level_component_prefill_capacity_skips"] > 0
run("canonical", ["Rscript", "--vanilla", str(TOOL / "canonical.R"),
    str(native), str(out / "canonical"), str(baseline)], env)
jobs.write(out / "reused-successes.json", reused_successes)
jobs.write(out / "closure.json", dict(status="PASS", native_sha256=build["native_sha256"],
    dependencies=dependencies, tests=["slots", "config-l3", "config-l4", "config-l5", "config-l6",
        "prefill-level", "prefill-l2", "prefill-structured", "methods", "capacity-skip", "canonical"]))
print("PASS supplemental regressions", flush=True)
