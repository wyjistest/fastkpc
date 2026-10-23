"""Audit and index the completed canonical source-closure development campaign."""
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

root=Path(__file__).resolve().parents[4]
os.chdir(root)
out=Path(sys.argv[1]).resolve()
tool=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
build=read(out/'build.json')
checks=read(out/'checks.json')
assert build['status']=='PASS' and checks['status']=='PASS'
native=root/'fastkpc/build/fastkpc_cuda.so'
assert sha(native)==build['native_sha256']
assert all(sha(root/path)==value for path,value in build['source_sha256'].items())
names=('off','dense1','structured1','structured2','dense2')
reports={name:read(out/name/'report.json') for name in names}
for name,report in reports.items():
    assert report['gates'] and all(report['gates'].values()), name
    assert report['native_sha256']==build['native_sha256']
    assert report['runner_sha256']==sha(tool/'run.R')
    assert report['receipt_sha256']==sha(out/name/'receipt.rds')
    assert report['source_sha256']==build['source_sha256']
    pending=out/name/'pending.rds'
    if pending.exists(): pending.unlink()

memory={}
for name in names:
    with (out/(name+'-gpu.csv')).open() as f:
        rows=list(csv.reader(f))
    samples=[r for r in rows if len(r)==7 and r[1].strip()=='0']
    other=[r for r in rows if len(r)==7 and r[1].strip()=='1']
    assert samples and other
    memory[name]=dict(sample_interval_ms=200,
        gpu0_peak_mib=max(float(r[2]) for r in samples),
        gpu1_peak_mib=max(float(r[2]) for r in other),
        gpu1_peak_utilization_percent=max(float(r[3]) for r in other),
        gpu0_min_sm_clock_mhz=min(float(r[5]) for r in samples),
        gpu0_max_sm_clock_mhz=max(float(r[5]) for r in samples),
        gpu0_peak_temperature_c=max(float(r[6]) for r in samples))

numeric_stages=('elapsed_sec','optimizer_sec','stage_b_sec','consumer_dcov_sec',
                'prepared_build_sec','native_setup_sec','consumer_eig_sec')
medians={mode:{field:statistics.median(reports[name]['stages'][field]
                                    for name in (mode+'1',mode+'2'))
               for field in numeric_stages} for mode in ('dense','structured')}
delta={field:medians['structured'][field]-medians['dense'][field] for field in numeric_stages}
with (out/'timing.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=['run']+list(numeric_stages)+['sampled_peak_mib'])
    writer.writeheader()
    for name in names:
        writer.writerow(dict(run=name,**{field:reports[name]['stages'][field] for field in numeric_stages},
                             sampled_peak_mib=memory[name]['gpu0_peak_mib']))

structured={}
for name in ('structured1','structured2'):
    rows=reports[name]['level_component_prefill']
    structured[name]=dict(produced=sum(r['produced'] for r in rows),
        resolved=sum(r['resolved'] for r in rows),pending_unused=sum(r['pending_at_end'] for r in rows),
        accepted=sum(r['structured']['accepted'] for r in rows),
        dense_fallback=sum(r['structured']['dense_fallback'] for r in rows),
        fallback_info=sum(r['structured']['fallback_info'] for r in rows),
        fallback_rotation=sum(r['structured']['fallback_rotation'] for r in rows),
        fallback_gap=sum(r['structured']['fallback_gap'] for r in rows),
        minimum_gap=min(r['structured']['minimum_gap'] for r in rows),
        max_rotations=max(r['structured']['rotations_max'] for r in rows),
        max_pool_device_bytes=max(r['structured']['pool_device_bytes'] for r in rows),
        max_inflight=max(r['structured']['peak_inflight'] for r in rows),
        waves=sum(r['structured']['waves'] for r in rows))
closure=dict(schema='structured-prefill-canonical-closure-v1',status='PASS_CANONICAL_OPT_IN',
    evidence_class='full source-closure build; off once plus two dense and two structured development runs; not promotion',
    native_sha256=build['native_sha256'],source_sha256=build['source_sha256'],
    all_ci_count=240498,configuration=dict(n=351,p=48,numCol=35,default_inf=True,
        cache_bytes=11811160064,eig_lanes=8,cuda_connections=32,structured_pool_contexts=10,
        structured_width=64,physical_gpu=0),
    run_order=list(names),elapsed_sec={name:r['elapsed_sec'] for name,r in reports.items()},
    two_run_medians=medians,structured_minus_dense=delta,
    outer_fraction_reduction=1-medians['structured']['elapsed_sec']/medians['dense']['elapsed_sec'],
    oracle_errors={name:r['oracle_errors'] for name,r in reports.items()},
    structured_comparisons={name:reports[name]['comparisons'] for name in structured},
    structured_ledger=structured,memory=memory,checks=checks,
    production_default=False,numerical_production='GPU only; archived kpcalg authority comparison-only',
    final_pdag_campaign='NOT_RUN',real_data_campaign='NOT_RUN',phase10='ACTIVE',
    candidate='NOT_FROZEN',holdout='SEALED',recommended=False)
(out/'closure.json').write_text(json.dumps(closure,indent=2)+'\n')
with native.open('rb') as source,(out/'fastkpc_cuda.so.gz').open('wb') as dest:
    with gzip.GzipFile(fileobj=dest,mode='wb',mtime=0) as compressed:
        shutil.copyfileobj(source,compressed)
for path in tool.iterdir():
    if path.is_file():
        dest=out/'tools'/path.relative_to(root)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest)
for name in ('fastkpc/tests/test_full_cuda_ci_structured_prefill.R',
             'fastkpc/tests/test_full_cuda_ci_level_component_prefill.R',
             'fastkpc/tests/test_full_cuda_ci_legacy_component_cache.R',
             'fastkpc/tools/with_phase10_exact_development_env.sh',
             'fastkpc/tools/research/exact_speedup_20261020/pinned_optimizer_gates.R',
             'fastkpc/tools/run_phase10_pvalue_contract_audit.R'):
    dest=out/'tools'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,dest)
with (out/'runtime.patch').open('w') as f:
    subprocess.run(['git','diff','--','fastkpc/src','fastkpc/tools/build_cuda_native.sh'],stdout=f,check=True)
inputs=['fastkpc/artifacts/full_cuda_ci/structured_prefill_20261022/manifest.json',
        'fastkpc/artifacts/full_cuda_ci/level_component_prefill_canonical_20261021/manifest.json',
        'fastkpc/artifacts/full_cuda_ci/level_component_prefill_canonical_20261021/canonical-off/receipt.rds.gz',
        'fastkpc/artifacts/full_cuda_ci/oracle_351x48_default_inf_v2/logical_ci_trace.rds',
        'fastkpc/artifacts/full_cuda_ci/oracle_351x48_default_inf_v2/pmax.rds',
        'fastkpc/artifacts/full_cuda_ci/oracle_351x48_default_inf_v2/adjacency.rds',
        'fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds']
files={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p))
       for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'}
(out/'manifest.json').write_text(json.dumps(dict(schema='structured-prefill-canonical-manifest-v1',
    status='PASS',native_sha256=build['native_sha256'],inputs={p:sha(p) for p in inputs},files=files),indent=2)+'\n')
print(json.dumps({key:closure[key] for key in ('status','native_sha256','elapsed_sec','two_run_medians',
    'outer_fraction_reduction','oracle_errors','structured_ledger','memory')},indent=2),flush=True)
print('PASS manifest',sha(out/'manifest.json'),flush=True)
