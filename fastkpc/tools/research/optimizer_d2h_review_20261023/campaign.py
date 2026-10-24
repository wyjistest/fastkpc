"""Run Codex's replay (built by build_replay.py) on idle GPU0 with Codex's environment.
Usage: campaign.py OUT campaign|profile PAIR [PAIR...]   PAIR = L3 | L5"""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path
here = Path(__file__).resolve().parent; root = here.parents[3]; os.chdir(root)
out = Path(sys.argv[1]).resolve(); mode = sys.argv[2]; pairs = sys.argv[3:]
capture = root / 'fastkpc/artifacts/full_cuda_ci/optimizer_windows_20261022'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
build = json.loads((out / 'build.json').read_text())
assert sha(build['native']) == build['native_sha256']
assert all(sha(out / p) == v for p, v in build['outputs'].items())
assert all(sha(root / p) == v for p, v in build['inputs'].items())
os.sched_setaffinity(0, range(20))
env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES='0', CUDA_DEVICE_MAX_CONNECTIONS='32',
    OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
    LD_LIBRARY_PATH='/usr/local/lib/R/lib:/usr/local/cuda/lib64')
env['FASTKPC_WINDOW_NATIVE'] = build['native']  # observer resolves symbols in the loaded native
assert not env.get('LD_PRELOAD') and not env.get('CUDA_LAUNCH_BLOCKING')
assert mode in ('campaign', 'profile')
spec = dict(L3=(3, 31, 32), L5=(5, 7, 8))
for pair in pairs:
    level, a, b = spec[pair]
    name = f'{mode}-L{level}-W{a}+W{b}'; prefix = out / name
    assert not prefix.with_suffix('.log').exists(), 'preserve existing evidence'
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    args = [str(out / 'replay'), str(capture / 'raw' / f'L{level}-W{a}.bin'),
            str(capture / 'raw' / f'L{level}-W{b}.bin'), str(prefix), mode, '6']
    if mode == 'profile':
        args = ['nsys', 'profile', '--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none',
                '--capture-range=cudaProfilerApi', '--capture-range-end=stop', '--force-overwrite=false',
                '-o', str(prefix)] + args
    start = time.time(); print('START', name, flush=True)
    with (out / (name + '.log')).open('w') as log, (out / (name + '-gpu.csv')).open('w') as gpu:
        monitor = subprocess.Popen(['nvidia-smi',
            '--query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,clocks.current.sm,temperature.gpu',
            '--format=csv,noheader,nounits', '-lms', '200'], stdout=gpu, stderr=subprocess.DEVNULL)
        try:
            result = subprocess.run(args, env=env, stdout=log, stderr=subprocess.STDOUT)
        finally:
            monitor.terminate(); monitor.wait(timeout=10)
    (out / (name + '-process.json')).write_text(json.dumps(dict(command=args, start_epoch=start,
        end_epoch=time.time(), exit_code=result.returncode, native_sha256=build['native_sha256'],
        affinity=sorted(os.sched_getaffinity(0)),
        environment={k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_MAX_CONNECTIONS',
                     'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'FASTKPC_WINDOW_NATIVE')}), indent=2) + '\n')
    assert result.returncode == 0, f'{name} failed; see log'
    if mode == 'profile':
        subprocess.run(['nsys', 'export', '--type=sqlite', '--output', str(prefix) + '.sqlite',
                        str(prefix) + '.nsys-rep'], check=True)
    print('PASS', name, round(time.time() - start, 1), 's', flush=True)
