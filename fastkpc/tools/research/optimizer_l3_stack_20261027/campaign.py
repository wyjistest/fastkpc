"""Separate, immutable diagnostic wrapper around the frozen coexistence probe."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

TOOL = Path(__file__).resolve().parent
ROOT = TOOL.parents[3]
sys.path.insert(0, str(TOOL.parent / 'optimizer_l3_slots_20261027'))
from common import NATIVE, PARENT, sha, load, write_new, check_identity, environment, source_identity

OLD = ROOT / 'fastkpc/artifacts/full_cuda_ci/optimizer_l3_coexistence_20261027/attempt2'
CAPTURE = ROOT / 'fastkpc/artifacts/full_cuda_ci/optimizer_l3_slots_20261027'


def check_old():
    check_identity(CAPTURE)
    b = load(OLD / 'build.json')
    for p, h in {**b['tools'], **b['dependencies'], **b['source_sha256']}.items():
        assert sha(ROOT / p) == h, p
    for p, h in b['outputs'].items():
        assert sha(OLD / p) == h, p
    manifest = load(OLD / 'manifest.json')
    for group, base in [('files', OLD), ('tools', ROOT), ('earlier_incomplete_builds', OLD.parent)]:
        for p, h in manifest[group].items():
            assert sha(base / p) == h, p
    return b


def validate(out):
    check_old()
    b = load(out / 'build.json')
    for p, h in b['inputs'].items():
        assert sha(ROOT / p) == h, p
    for p, h in b['outputs'].items():
        assert sha(out / p) == h, p
    assert sha(out / 'protocol.json') == b['protocol_sha256']
    return b


def prepare(out):
    original = check_old()
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / 'protocol.json', dict(
        schema='l3-first-use-stack-attribution-v1', declared_epoch=time.time(),
        declaration_sha256=sha(TOOL / 'README.md'),
        original_protocol_sha256=sha(OLD / 'protocol.json'),
        original_manifest_sha256=sha(OLD / 'manifest.json'),
        diagnostic_overlay_sha256=sha(OLD / 'probe.so'), native_sha256=sha(NATIVE),
        slots=32, capacity=33, maximum_raw_pool_increment_bytes=1536*1024**2,
        minimum_observed_free_bytes=2*1024**3,
        baseline='after real coefficient arena reserve, before pool construction',
        instrumented_l3_launches=64, warmup_added=False, stack_limit_modified=False,
        added_gpu_synchronization=False, model_is_acceptance_gate=False,
        selected_gpu=1, selected_gpu_exclusive=True, external_gpu0_allowed=True,
        evidence_class='GPU1-exclusive shared-host memory and semantic diagnostic',
        performance_evidence=False,
        expected_S='same raw capacity rejection after optimizer; no Stage A',
        unresolved_bytes_remain_in_capacity=True))
    symbols = subprocess.check_output(['nm', '-D', '--defined-only', str(NATIVE)], text=True).splitlines()
    symbols += subprocess.check_output(['nm', '-D', '--defined-only', str(OLD / 'probe.so')], text=True).splitlines()
    names = [r.split()[-1] for r in symbols if r.strip()]
    decoded = subprocess.check_output(['c++filt'], input='\n'.join(names), text=True).splitlines()
    definitions = []
    for key, prefix in [('batch', 'fastkpc::multi_penalty_gcv_cuda_optimize_batch_small64('),
                        ('optimizer', 'l3_probe::optimizer(')]:
        matches = [s for s, d in zip(names, decoded) if d.startswith(prefix)]
        assert len(matches) == 1, (key, matches)
        definitions.append('#define STACK_' + key + ' "' + matches[0] + '"\n')
    with (out / 'symbols.h').open('x') as stream:
        stream.write(''.join(definitions))
    rcpp = subprocess.check_output(['Rscript', '--vanilla', '-e',
                                   'cat(system.file("include",package="Rcpp"))'], text=True)
    cmd = ['g++', '-O2', '-std=c++17', '-shared', '-fPIC', '-pthread', '-I'+str(out),
           '-I'+str(ROOT/'fastkpc/src'), '-I/usr/local/cuda/include',
           '-I/usr/local/lib/R/include', '-I'+rcpp, str(TOOL/'tap.cpp'),
           str(OLD/'probe.so'), str(CAPTURE/'audit.so'),
           '-Wl,-rpath,'+str(OLD), '-Wl,-rpath,'+str(CAPTURE),
           '-L/usr/local/cuda/lib64', '-Wl,-rpath,/usr/local/cuda/lib64', '-lcudart', '-ldl',
           '-L/usr/local/lib/R/lib', '-Wl,-rpath,/usr/local/lib/R/lib', '-lR', '-o', str(out/'tap.so')]
    with (out/'build.log').open('x') as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    inputs = list(TOOL.glob('*')) + [OLD/'probe.so', OLD/'run.R', OLD/'build.json', OLD/'manifest.json',
        CAPTURE/'audit.so', TOOL.parent/'optimizer_l3_slots_20261027/common.py',
        TOOL.parent/'optimizer_l3_coexistence_20261027/probe.hpp',
        TOOL.parent/'optimizer_l3_coexistence_20261027/geometry.hpp',
        TOOL.parent/'optimizer_layer_refill_20261023/audit.hpp']
    write_new(out/'build.json', dict(status='PASS', command=cmd,
        native_sha256=sha(NATIVE), source_sha256=original['source_sha256'],
        inputs={str(p.relative_to(ROOT)):sha(p) for p in inputs if p.is_file()},
        outputs={p:sha(out/p) for p in ['tap.so','symbols.h','build.log']},
        protocol_sha256=sha(out/'protocol.json'), diagnostic_overlay_sha256=sha(OLD/'probe.so')))
    print('PREPARED', out, sha(out/'tap.so'), flush=True)


def query_processes():
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                   '--format=csv,noheader'], text=True)
    return {int(r.split(',')[0]):r.split(',')[1].strip() for r in raw.splitlines() if r.strip()}


def monitored(out, mode, cmd, env):
    os.chdir(ROOT)
    os.sched_setaffinity(0, range(20))
    uuid = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=uuid',
                                   '--format=csv,noheader'], text=True).strip()
    initial = query_processes()
    assert uuid not in initial.values(), 'GPU1 has an external compute task'
    started = time.time()
    samples, unexpected, external_other = [], set(), set(initial)
    p = None
    print('START', mode, time.strftime('%H:%M:%S'), 'GPU1 exclusive; host shared', flush=True)
    with (out/(mode+'.log')).open('x') as log, (out/(mode+'-gpu.csv')).open('x') as gpu:
        monitor = subprocess.Popen(['nvidia-smi',
            '--query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,clocks.current.sm,temperature.gpu',
            '--format=csv,noheader,nounits', '-lms', '200'], stdout=gpu, stderr=subprocess.DEVNULL)
        try:
            p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while True:
                active = query_processes()
                known = {p.pid}
                edges = [tuple(map(int, r.split())) for r in subprocess.check_output(
                    ['ps','-eo','pid=,ppid='], text=True).splitlines()]
                for _ in range(8):
                    known.update(pid for pid, parent in edges if parent in known)
                unexpected.update(pid for pid, device in active.items() if device==uuid and pid not in known)
                external_other.update(pid for pid, device in active.items() if device!=uuid and pid not in known)
                samples.append(dict(epoch=time.time(), processes=active, own_pids=sorted(known)))
                if unexpected:
                    break
                try:
                    p.wait(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if p is not None and p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
            monitor.terminate()
            monitor.wait(timeout=10)
    write_new(out/(mode+'-gpu-processes.json'), samples)
    process = dict(command=cmd, start_epoch=started, end_epoch=time.time(), exit_code=p.returncode,
        physical_gpu=1, gpu_uuid=uuid, gpu1_exclusive=not unexpected,
        unexpected_gpu1_pids=sorted(unexpected), external_other_gpu_pids=sorted(external_other),
        both_gpus_isolated=not unexpected and not external_other,
        evidence_class='shared-host diagnosis only; no performance evidence',
        affinity=sorted(os.sched_getaffinity(0)), native_sha256=sha(NATIVE),
        logging_tap_sha256=sha(out/'tap.so'), protocol_sha256=sha(out/'protocol.json'),
        environment={k:v for k,v in env.items() if k.startswith(('CUDA_','FASTKPC_')) or k in
                     ('LD_PRELOAD','LD_LIBRARY_PATH','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')})
    write_new(out/(mode+'-process.json'), process)
    assert not unexpected, 'GPU1 interference: incomplete attempt preserved'
    source_identity()
    return process


def run(out, mode):
    validate(out)
    assert mode in ('W','S')
    assert not (out/mode).exists() and not (out/(mode+'-process.json')).exists()
    reference = out/'W/stage-a.bin'
    if mode=='S':
        assert load(out/'W-closure.json')['status']=='PASS'
        assert sha(reference)==sha(OLD/'W/stage-a.bin')
    env=environment()
    env.update(LD_PRELOAD=':'.join(map(str,[out/'tap.so',OLD/'probe.so',CAPTURE/'audit.so'])),
        FASTKPC_L3_STACK_LIBRARY=str(out/'tap.so'), FASTKPC_L3_STACK_LOG=str(out/(mode+'-observations.jsonl')),
        FASTKPC_L3_ORIGINAL_RUNNER=str(OLD/'run.R'), FASTKPC_L3_PROBE_LIBRARY=str(OLD/'probe.so'),
        FASTKPC_L3_PROBE_OUT=str(out/mode), FASTKPC_DIAG_L3_REPLACEMENT='1' if mode=='S' else '0',
        FASTKPC_L3_PROBE_FIXTURES=str(CAPTURE/'capture-raw'), FASTKPC_L3_STAGE_A_REFERENCE=str(reference),
        FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_BYTES='11811160064',
        FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_CACHE_RESIDUAL_BYPASS='1',
        FASTKPC_DCC_GAMMA_LEGACY_COMPONENT_EIG_LANES='8', FASTKPC_DIAG_SHARED_HOST_QUALIFICATION='1')
    ref=PARENT/'on3/receipt.rds'
    process=monitored(out, mode, ['Rscript','--vanilla',str(TOOL/'run.R'),str(out/mode),'1',str(ref),str(ref)],env)
    closure=load(out/mode/'stack-closure.json')
    assert closure['resources_closed'] and closure['observation_errors']==0
    assert closure['l3_launches']==closure['l3_requests']==closure['optimizer_setups_bitwise_exact']==4065
    assert closure['optimizer_targets_bitwise_exact']==54878
    if mode=='W':
        assert process['exit_code']==0 and all(load(out/mode/'report.json')['gates'].values())
        assert load(out/mode/'probe.json')['status']=='PASS'
        assert sha(reference)==sha(OLD/'W/stage-a.bin')
        status='PASS'
    else:
        assert process['exit_code']==1
        assert closure['error']=='optimizer slots exceeded the qualified memory budget'
        assert not (out/mode/'receipt.rds').exists()
        status='CONFIRMED_ORIGINAL_CAPACITY_STOP'
    validate(out)
    write_new(out/(mode+'-closure.json'), dict(status=status, observation_closure=closure,
        process_sha256=sha(out/(mode+'-process.json')), observations_sha256=sha(out/(mode+'-observations.jsonl')),
        stage_a_reference_sha256=sha(reference), native_sha256=sha(NATIVE)))
    print(status, mode, closure, flush=True)


if __name__=='__main__':
    action=sys.argv[1]
    destination=Path(sys.argv[2]).resolve()
    if action=='prepare':
        prepare(destination)
    elif action=='run':
        run(destination,sys.argv[3])
    else:
        raise ValueError(action)
