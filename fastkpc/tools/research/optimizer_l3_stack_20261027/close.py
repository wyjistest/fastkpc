"""Seal observations without converting the unchanged capacity STOP into a GO."""
import csv
import json
from pathlib import Path
import sys
from campaign import ROOT, TOOL, OLD, NATIVE, sha, load, write_new, validate

out=Path(sys.argv[1]).resolve()
validate(out)
assert load(out/'W-closure.json')['status']=='PASS'
assert load(out/'S-closure.json')['status']=='CONFIRMED_ORIGINAL_CAPACITY_STOP'
summary={}
for mode in ['W','S']:
    rows=[json.loads(line) for line in (out/(mode+'-observations.jsonl')).read_text().splitlines()]
    assert all(r['stack_status']==r['memory_status']==0 for r in rows)
    launches=[r for r in rows if r['phase']=='l3_launch_enter']
    returns=[r for r in rows if r['phase']=='l3_launch_return']
    assert len(launches)==len(returns)==64
    assert set(r['launch'] for r in launches)==set(range(1,65))
    assert all(r['request_key'] and r['target_count'] in range(2,34) for r in launches)
    by_phase={}
    for r in rows:
        by_phase.setdefault(r['phase'],[]).append(r)
    assert all(r['api_status']==0 for r in returns)
    closed=by_phase['after_native_cleanup'][0]
    assert all(closed[k]==0 for k in ['app_bytes','allocations','streams','events','blas'])
    selected=['single_launch_enter','single_launch_return','before_l3_begin','level_begin',
        'pool_before','pool_created','workers_start','optimizer_workers_joined','level_end','after_l3_finish','after_native_cleanup']
    summary[mode]=dict(checkpoints={p:by_phase[p] for p in selected if p in by_phase},
        observed_stack_values=sorted(set(r['stack_bytes'] for r in rows)),
        first_l3_entry=min(launches,key=lambda r:r['begin_ns']),
        first_l3_return=min(returns,key=lambda r:r['begin_ns']),
        l3_entry_stack_values=sorted(set(r['stack_bytes'] for r in launches)),
        l3_return_stack_values=sorted(set(r['stack_bytes'] for r in returns)),
        observations_with_allocation_change=sum(r['app_bytes_before']!=r['app_bytes'] or
            r['allocation_calls_before']!=r['allocation_calls'] for r in rows),
        process=load(out/(mode+'-process.json')), closure=load(out/(mode+'-closure.json')))
    if mode=='W':
        windows=by_phase['stage_a_window_completed'];assert len(windows)==64
        start=by_phase['level_begin'][0]
        growth=[start['free_bytes']-r['free_bytes']-(r['app_bytes']-start['app_bytes']) for r in windows]
        summary[mode]['after_window_non_application_growth_bytes']=growth
        summary[mode]['canonical_gates']=load(out/'W/report.json')['gates']
        summary[mode]['probe']=load(out/'W/probe.json')
    else:
        before=by_phase['pool_before'][0];created=by_phase['pool_created'][0];worked=by_phase['optimizer_workers_joined'][0]
        assert before['free_bytes']-worked['free_bytes']>1536*1024**2
        assert worked['free_bytes']>=2*1024**3 and worked['worker_mutations']==0
        assert created['app_bytes']==worked['app_bytes']
        summary[mode]['accounting']=dict(
            create_increment_bytes=before['free_bytes']-created['free_bytes'],
            first_execution_growth_bytes=created['free_bytes']-worked['free_bytes'],
            total_raw_increment_bytes=before['free_bytes']-worked['free_bytes'],
            pool_application_bytes=created['app_bytes']-before['app_bytes'],
            stack_limit_change_bytes=worked['stack_bytes']-created['stack_bytes'],
            stack_limit_delta_times_196608_bytes=(worked['stack_bytes']-created['stack_bytes'])*196608,
            first_execution_growth_minus_stack_delta_model_bytes=created['free_bytes']-worked['free_bytes']-
                (worked['stack_bytes']-created['stack_bytes'])*196608)
write_new(out/'closure.json',dict(schema='l3-first-use-stack-attribution-closure-v1',
    status='DIAGNOSTIC_COMPLETE_ORIGINAL_CAPACITY_STOP_RETAINED', variants=summary,
    native_sha256=sha(NATIVE), original_overlay_sha256=sha(OLD/'probe.so'),
    protocol_sha256=sha(out/'protocol.json'), build_sha256=sha(out/'build.json'),
    performance_evidence=False, production_sources_changed=False, production_native_changed=False,
    new_capacity_contract_qualified=False, S_stage_a_qualified=False, runtime_integration='NOT_YET_GO',
    captured_cohort_status='GO_CAPTURED_L3_SLOTS', accepted_canonical_median_sec=368.851,
    phase10='ACTIVE', candidate='NOT_FROZEN', holdout='SEALED', recommended=False))
write_new(out/'manifest.json',dict(schema='l3-stack-attribution-manifest-v1',
    files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()},
    tools={str(p.relative_to(ROOT)):sha(p) for p in sorted(TOOL.iterdir()) if p.is_file()}))
print('DIAGNOSTIC_COMPLETE_ORIGINAL_CAPACITY_STOP_RETAINED')
print(json.dumps(summary['S']['accounting'],indent=2))
