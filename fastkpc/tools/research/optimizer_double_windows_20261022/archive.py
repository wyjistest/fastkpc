"""Validate and bind all diagnostic artifacts without upgrading evidence status."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

here=Path(__file__).resolve().parent;root=here.parents[3]
out=Path(sys.argv[1]).resolve();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
build=json.loads((out/'build.json').read_text())
analysis=json.loads((out/'analysis.json').read_text())
assert sha(root/'fastkpc/build/fastkpc_cuda.so')==build['native_sha256']
for field in ('inputs','sources','source_closure'):
    assert all(sha(root/p)==v for p,v in build[field].items()),field
assert all(sha(out/p)==v for p,v in build['outputs'].items())
capture=root/'fastkpc/artifacts/full_cuda_ci/optimizer_windows_20261022'
assert sha(capture/'manifest.json')==build['capture_manifest_sha256']
assert len(analysis['pairs'])==len(analysis['profiles'])==2
for pair in analysis['pairs'].values():
    assert all(pair['paired_cycles'][v]['wins']==0 for v in 'BC')
    assert all(pair['timings_ms'][v]['min']>pair['timings_ms']['A']['max'] for v in 'BC')
full_pairs=full_targets=reduced_pairs=reduced_targets=0
for name in ('L3-W31+W32','L5-W7+W8'):
    for mode,expected_runs,expected_setups in (('campaign',24,128),('profile',6,128),('smoke',6,4),('memcheck',6,4)):
        prefix=out/f'{mode}-{name}'
        summary=json.loads(Path(str(prefix)+'-summary.json').read_text())
        process=json.loads(Path(str(prefix)+'-process.json').read_text())
        assert process['exit_code']==0 and process['affinity']==list(range(20))
        with Path(str(prefix)+'-runs.csv').open() as f:
            runs=list(csv.DictReader(f))
        assert len(runs)==summary['runs']==expected_runs
        assert summary['setups']==expected_setups and summary['status']=='PASS'
        assert summary['output_leases_closed'] and summary['observation_events_closed']
        assert all(r['parity']=='1' for r in runs)
        assert sum(int(r['targets']) for r in runs)==summary['compared_target_occurrences']
        if expected_setups==128:
            full_pairs+=len(runs);full_targets+=summary['compared_target_occurrences']
        else:
            reduced_pairs+=len(runs);reduced_targets+=summary['compared_target_occurrences']
    log=(out/f'memcheck-{name}.log').read_text()
    assert 'ERROR SUMMARY: 0 errors' in log and '0 bytes leaked' in log
    assert (out/f'profile-{name}.sqlite').is_file()
    variants=analysis['profiles']['profile-'+name]['variants']
    assert variants['A']['cross_window_optimizer_overlap_ms']==0
    assert all(not v['timed_allocation_calls'] for v in variants.values())
telemetry={}
for path in sorted(out.glob('*-gpu.csv')):
    with path.open() as f:
        data=list(csv.reader(f))
    g0=[r for r in data if len(r)==7 and int(r[1])==0]
    g1=[r for r in data if len(r)==7 and int(r[1])==1]
    assert g0 and g1
    telemetry[path.stem]=dict(gpu0_sampled_peak_mib=max(float(r[2]) for r in g0),
        gpu1_sampled_peak_mib=max(float(r[2]) for r in g1),
        gpu1_utilization_max=max(float(r[3]) for r in g1),
        gpu1_nonzero_utilization_samples=sum(float(r[3])>0 for r in g1),
        gpu1_sample_count=len(g1),
        gpu0_clock_range_mhz=[min(float(r[5]) for r in g0),max(float(r[5]) for r in g0)])
    # Primary timing isolation is a gate. Preserve incidental desktop activity
    # in separate profiler runs instead of claiming that every sample was zero.
    if path.name.startswith('campaign-'):
        assert telemetry[path.stem]['gpu1_utilization_max']==0
        assert telemetry[path.stem]['gpu1_sampled_peak_mib']==425
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
closure=dict(schema='optimizer-double-window-closure-v1',status='PASS_DIAGNOSTIC',
    native_sha256=build['native_sha256'],capture_manifest_sha256=build['capture_manifest_sha256'],
    analysis_sha256=sha(out/'analysis.json'),full_pair_runs=full_pairs,
    full_compared_target_occurrences=full_targets,reduced_pair_runs=reduced_pairs,
    reduced_compared_target_occurrences=reduced_targets,all_outputs_bitwise=True,
    target_identity_order_exact=True,gpu_exact_primary_authority=True,output_leases_closed=True,
    reduced_memcheck_errors=0,reduced_memcheck_leaked_bytes=0,
    profile_timed_cuda_alloc_free_calls=0,telemetry=telemetry,
    primary_gpu1_utilization_zero=True,
    profile_gpu1_activity_note='L5 profile has one 1% sample at 210 MHz; profiler peaks at 427 MiB versus 425 MiB desktop baseline. Raw telemetry retained; profile timings are not primary performance evidence.',
    production_sources_unchanged=True,production_defaults_unchanged=True,
    scope='Two adjacent captured windows; A two 64-request calls, B/C 128 requests with 64/32 workers',
    decisions=dict(direct_128_request_concatenation='STOP_FOR_CAPTURED_PAIRS',
                   combined_32_workers='STOP_FOR_CAPTURED_PAIRS'),
    runtime_rolling_integration='NOT_YET_GO',phase10='ACTIVE',candidate='NOT_FROZEN',
    holdout='SEALED',recommended=False,
    evidence_class='isolated captured-cohort development diagnostic; no full-method or promotion claim')
(out/'closure.json').write_text(json.dumps(closure,indent=2)+'\n')
environment=dict(cpu_affinity=list(range(20)),cuda_visible_devices='0',cuda_device_max_connections='32',
    blas_omp_mkl_threads=1,
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total,power.limit,memory.used','--format=csv'],text=True),
    gpu_processes_at_closure=subprocess.check_output(['nvidia-smi'],text=True),
    nsys=subprocess.check_output(['nsys','--version'],text=True),
    nvcc=subprocess.check_output(['/usr/local/cuda/bin/nvcc','--version'],text=True))
(out/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
manifest=dict(schema='optimizer-double-window-manifest-v1',native_sha256=build['native_sha256'],
    capture_manifest_sha256=build['capture_manifest_sha256'],inputs=build['inputs'],
    source_closure=build['source_closure'],
    tools={str(p.relative_to(root)):sha(p) for p in sorted(here.iterdir()) if p.is_file()},
    reused_tools={p:v for p,v in build['sources'].items() if 'optimizer_windows_20261022/' in p},
    files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'})
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('PASS closure; manifest SHA256',sha(out/'manifest.json'))
