"""Bind the completed diagnostic to immutable capture and production identities."""
import csv
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CAPTURE, NATIVE, ROOT, TOOL, check_identity, load, sha, write_new

out = Path(sys.argv[1]).resolve()
build = check_identity(out)
analysis = load(out/'analysis.json')
assert analysis['analysis_sha256'] == sha(TOOL/'analyze.py')
assert analysis['status'] in ['GO_L4_BOUNDED_REFILL_EXPERIMENT_ONLY',
    'CONDITIONAL_MODEL_NOT_VALIDATED','STOP_L4_REFILL_OPPORTUNITY_TOO_SMALL']
with (out/'analysis-tests.log').open('x') as log:
    subprocess.run([sys.executable,'-I',str(TOOL/'test_analysis.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
processes, summaries, telemetry = [], [], []
for run in (1,2):
    name='profile-'+str(run)
    p=load(out/(name+'-process.json'));s=load(out/(name+'-summary.json'))
    assert p['exit_code']==0 and not p['unexpected_gpu_pids']
    assert s['all_outputs_bitwise_exact'] and s['resources_closed'] and s['output_leases_closed']
    assert s['forbidden_calls']==0 and s['minimum_free_after_create']>=2*1024**3
    assert s['setups_per_pass']==2154 and s['targets_per_pass']==17767 and s['windows_per_pass']==34 and s['passes']==2
    rows=list(csv.reader((out/(name+'-gpu.csv')).open()))
    gpu1=[r for r in rows if int(r[1])==1]
    assert gpu1
    telemetry.append(dict(run=run,sample_count=len(gpu1),
        gpu1_sampled_peak_mib=max(int(r[2]) for r in gpu1),
        gpu1_sm_clock_mhz_min=min(int(r[5]) for r in gpu1),gpu1_sm_clock_mhz_max=max(int(r[5]) for r in gpu1),
        gpu1_temperature_c_max=max(int(r[6]) for r in gpu1),
        other_gpu_pids=p['other_gpu_pids']))
    processes.append(p);summaries.append(s)
closure=dict(schema='l4-tail-closure-v1',status=analysis['status'],
    evidence_class='GPU1 captured diagnostic on committed runtime source closure; measured profiles and counterfactual service-time models, not canonical outer performance',
    producer_head=build['producer_head'],runtime_commit=build['runtime_commit'],native_sha256=sha(NATIVE),
    original_capture_manifest_sha256=sha(CAPTURE/'manifest.json'),
    independent_processes=2,full_layer_passes=4,
    compared_target_occurrences=sum(s['compared_target_occurrences'] for s in summaries),
    all_outputs_bitwise_exact=True,resource_closure=True,forbidden_native_resource_calls=0,
    model_fit_pass=analysis['host_model_fit_pass'],modeled_opportunity_pass=analysis['modeled_opportunity_pass'],
    analysis_sha256=sha(out/'analysis.json'),telemetry=telemetry,
    production_source_and_native_unchanged=True,production_cache_present=False,
    new_slot_implementation=False,new_full_canonical_run=False,new_final_pdag_qualified=False,new_real_data_qualified=False,
    previous_canonical_development_median_sec=461.663,
    full_layer_capacity_status='STOP_FULL_LAYER_CAPACITY',runtime_integration='NOT_YET_GO',
    phase10='ACTIVE',candidate='NOT_FROZEN',holdout='SEALED',recommended=False)
assert closure['compared_target_occurrences']==71068
write_new(out/'closure.json',closure)
tools=[p for p in sorted(TOOL.iterdir()) if p.is_file()]
files=[p for p in sorted(out.iterdir()) if p.is_file() and p.name!='manifest.json']
write_new(out/'manifest.json',dict(schema='l4-tail-manifest-v1',native_sha256=sha(NATIVE),
    source_sha256=build['source_sha256'],dependencies=build['dependencies'],
    references={str((CAPTURE/p).relative_to(ROOT)):h for p,h in build['fixtures'].items()},
    tools={str(p.relative_to(ROOT)):sha(p) for p in tools},
    files={p.name:sha(p) for p in files}))
print(analysis['status'])
print('manifest SHA256',sha(out/'manifest.json'))
