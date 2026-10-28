"""Unchanged-native L6 replacement probe; preserve all attempted artifacts."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

TOOL = Path(__file__).resolve().parent
ROOT = TOOL.parents[3]
sys.path.insert(0, str(TOOL.parent / 'optimizer_l6_slots_20261025'))
from common import CAPTURE, NATIVE, check_identity, load, sha, write_new

out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=False)
os.chdir(ROOT)
qualified = ROOT / 'fastkpc/artifacts/full_cuda_ci/optimizer_l6_slots_20261025/qualified'
identity = check_identity(qualified)
assert load(qualified / 'closure.json')['status'] == 'GO_CAPTURED_L6_SLOTS'
assert load(qualified / 'device-code.json')['status'] == 'PASS'
slot = qualified / 'slot_backend.so'
audit = qualified / 'audit.so'
symbols = [s.split()[-1] for s in subprocess.check_output(
    ['nm', '-D', '--defined-only', str(NATIVE)], text=True).splitlines()]
demangled = subprocess.check_output(['c++filt'], input='\n'.join(symbols), text=True).splitlines()
names = dict(plan='make_layer_plan',
             create='fastkpc::create_multi_penalty_gcv_capacity_prepared',
             build='fastkpc::build_full_cuda_ci_level_components')
lines = []
for alias, name in names.items():
    matches = [s for s, d in zip(symbols, demangled) if d.startswith(name + '(')]
    assert len(matches) == 1, (alias, matches)
    lines.append('constexpr const char* SYM_' + alias + ' = "' + matches[0] + '";')
(out / 'symbols.inc').write_text('\n'.join(lines) + '\n')
rcpp = subprocess.check_output(['Rscript', '--vanilla', '-e',
    'cat(system.file("include",package="Rcpp"))'], text=True)
command = ['g++', '-O2', '-std=c++17', '-fPIC', '-shared', '-pthread',
    '-I' + str(out), '-I' + str(ROOT / 'fastkpc/src'), '-I/usr/local/cuda/include',
    '-I/usr/local/lib/R/include', '-I' + rcpp, str(TOOL / 'coexistence.cpp'),
    str(slot), str(audit), str(NATIVE), '-Wl,-rpath,' + str(qualified),
    '-Wl,-rpath,' + str(NATIVE.parent), '-L/usr/local/cuda/lib64',
    '-Wl,-rpath,/usr/local/cuda/lib64', '-lcudart', '-lcublas', '-ldl',
    '-L/usr/local/lib/R/lib', '-Wl,-rpath,/usr/local/lib/R/lib', '-lR',
    '-o', str(out / 'coexistence.so')]
with (out / 'build.log').open('x') as log:
    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
dependencies = [TOOL / n for n in ['coexistence.py', 'coexistence.cpp', 'coexistence.R', 'REPLACEMENT_CAPACITY.md']]
dependencies += [TOOL.parent / 'optimizer_l4_slot_runtime_20261024/run.R',
    ROOT / 'fastkpc/tools/research/exact_speedup_20261020/pinned_optimizer_gates.R',
    ROOT / 'fastkpc/tools/run_phase10_pvalue_contract_audit.R',
    ROOT / 'fastkpc/tools/with_phase10_exact_development_env.sh',
    ROOT / 'fastkpc/R/fast_kpc.R', ROOT / 'fastkpc/R/full_cuda_ci_phase35_vertical.R',
    ROOT / 'fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds']
build = dict(status='PASS', native_sha256=sha(NATIVE), source_sha256=identity['source_sha256'],
    probe_sha256=sha(out / 'coexistence.so'), slot_dso_sha256=sha(slot),
    audit_dso_sha256=sha(audit), command=command,
    qualified_build_sha256=sha(qualified / 'build.json'),
    captured_closure_sha256=sha(CAPTURE / 'capture-closure.json'),
    dependencies={str(p.relative_to(ROOT)): sha(p) for p in dependencies})
write_new(out / 'build.json', build)
write_new(out / 'protocol.json', dict(schema='l6-real-cache-replacement-v1',
    declaration_epoch=time.time(), minimum_free_bytes=2 * 1024**3,
    maximum_warm_increment_bytes=1024**3, slots=32, target_capacity=5,
    diagnostic_output_bytes=6952288, original_window_handles_retained=False,
    weak_reference_expiry_required=245, interval='end of real L6 Stage A before Stage B',
    baseline='canonical S2, properties ON, L4/L5 slots ON, retained full-eig stack',
    performance_evidence=False, declaration_sha256=sha(TOOL / 'REPLACEMENT_CAPACITY.md')))
os.sched_setaffinity(0, range(20))
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
    '--format=csv,noheader'], text=True).strip(), 'both GPUs must be idle'
env = os.environ.copy()
assert not env.get('LD_PRELOAD') and not env.get('CUDA_LAUNCH_BLOCKING')
env.update(CUDA_VISIBLE_DEVICES='1', CUDA_DEVICE_MAX_CONNECTIONS='32',
    OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
    FASTKPC_CUDA_DEVICE_PROPERTIES_CACHE='1',
    FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H='1',
    FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES='11811160064',
    FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS='1',
    FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES='8',
    FASTKPC_WINDOW_NATIVE=str(NATIVE), FASTKPC_L6_PROBE_FIXTURES=str(CAPTURE / 'capture-raw'),
    FASTKPC_L6_PROBE_OUT=str(out), LD_PRELOAD=str(out / 'coexistence.so') + ':' + str(audit),
    LD_LIBRARY_PATH='/usr/local/lib/R/lib:/usr/local/cuda/lib64')
reference = ROOT / 'fastkpc/artifacts/full_cuda_ci/device_properties_cache_20261025/on1/receipt.rds'
command = ['sh', 'fastkpc/tools/with_phase10_exact_development_env.sh',
    'Rscript', '--vanilla', str(TOOL / 'coexistence.R'), str(out / 'canonical'), '1', str(reference), '-']
start = time.time()
samples, unexpected = [], set()
print('START L6 replacement coexistence with actual production cache/arena', flush=True)
with (out / 'run.log').open('x') as log, (out / 'gpu.csv').open('x') as gpu:
    monitor = subprocess.Popen(['nvidia-smi',
        '--query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,clocks.current.sm,temperature.gpu',
        '--format=csv,noheader,nounits', '-lms', '200'], stdout=gpu, stderr=subprocess.DEVNULL)
    try:
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        while True:
            raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                '--format=csv,noheader'], text=True)
            active = {int(r.split(',')[0]): r.split(',')[1].strip() for r in raw.splitlines() if r.strip()}
            samples.append(dict(epoch=time.time(), processes=active))
            unexpected.update(set(active) - {proc.pid})
            if unexpected:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait()
                break
            try:
                proc.wait(timeout=5)
                break
            except subprocess.TimeoutExpired:
                pass
    finally:
        monitor.terminate(); monitor.wait(timeout=10)
write_new(out / 'process.json', dict(command=command, exit_code=proc.returncode,
    start_epoch=start, end_epoch=time.time(), physical_gpu=1,
    affinity=sorted(os.sched_getaffinity(0)), unexpected_gpu_pids=sorted(unexpected)))
write_new(out / 'gpu-processes.json', samples)
assert proc.returncode == 0 and not unexpected, 'preserve failed attempt'
assert sha(NATIVE) == build['native_sha256']
check_identity(qualified)
assert all(sha(ROOT / p) == h for p, h in build['dependencies'].items())
report = load(out / 'canonical/report.json')
assert report['gates'] and all(report['gates'].values())
capacity = load(out / 'coexistence-summary.json')
assert capacity['status'] == 'PASS', 'replacement capacity STOP; no runtime integration'
assert capacity['resources_restored'] and capacity['worker_resource_mutations'] == 0
write_new(out / 'closure.json', dict(status='PASS_REAL_CACHE_ARENA_REPLACEMENT_COEXISTENCE',
    native_sha256=sha(NATIVE), capacity=capacity, canonical_hard_gates=report['gates'],
    oracle_errors=report['oracle_errors'], evidence_class='isolated diagnostic; no performance claim',
    reference_sha256=sha(reference), diagnostic_shadow_d2h_bytes=6952288))
write_new(out / 'manifest.json', dict(schema='l6-replacement-coexistence-manifest-v1',
    files={str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()},
    dependencies=build['dependencies'], qualified_manifest_sha256=sha(qualified / 'manifest.json')))
print('PASS', capacity, flush=True)
