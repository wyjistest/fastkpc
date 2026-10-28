"""Seal a capacity STOP; never reinterpret an aborted S run as qualification."""
import csv
import subprocess
from pathlib import Path
import sys
TOOL=Path(__file__).resolve().parent;ROOT=TOOL.parents[3]
sys.path.insert(0,str(TOOL.parent/'optimizer_l3_slots_20261027'))
from common import NATIVE,load,sha,write_new,source_identity,check_identity
out=Path(sys.argv[1]).resolve();source_identity()
check_identity(ROOT/'fastkpc/artifacts/full_cuda_ci/optimizer_l3_slots_20261027')
b=load(out/'build.json');protocol=load(out/'protocol.json')
for p,h in {**b['tools'],**b['dependencies'],**b['source_sha256']}.items():assert sha(ROOT/p)==h,p
for p,h in b['outputs'].items():assert sha(out/p)==h,p
w=load(out/'W-closure.json');assert w['status']=='PASS' and all(w['canonical_hard_gates'].values())
processes={name:load(out/(name+'-process.json')) for name in ['W','S','S-attribution']}
assert all(p['isolated'] for p in processes.values())
assert processes['W']['exit_code']==0 and all(processes[n]['exit_code']==1 for n in ['S','S-attribution'])
assert all('optimizer slots exceeded the qualified memory budget' in (out/(n+'.log')).read_text() for n in ['S','S-attribution'])
extra=out/'failure-attribution';f=load(extra/'closure.json');a=f['attribution']
assert f['status']=='CONFIRMED_PREDECLARED_CAPACITY_STOP' and not f['threshold_changed'] and not f['shape_changed']
assert a['work_increment_bytes']>protocol['maximum_pool_increment_bytes']
assert a['free_after_optimizer_bytes']>=protocol['minimum_observed_free_bytes']
assert a['optimizer_setups_bitwise_exact']==4065 and a['optimizer_targets_bitwise_exact']==54878
assert a['resources_closed_after_exception'] and a['worker_resource_mutations']==0
tap=load(extra/'build.json')
for p,h in tap['dependencies'].items():assert sha(ROOT/p)==h,p
assert sha(extra/'tap.so')==tap['tap_sha256']
assert sha(out/'W/stage-a.bin')==w['stage_a_reference_sha256']==f['original_stage_a_reference_sha256']
assert not (out/'S/receipt.rds').exists() and not (out/'S-attribution/receipt.rds').exists()
w_rows=list(csv.DictReader((out/'W/checkpoints.csv').open()));w_start=w_rows[0]
w_completed=[r for r in w_rows if r['checkpoint']=='stage_a_window_completed']
assert len(w_completed)==64
shared_growth=[]
for r in w_completed:
    shared_growth.append(dict(targets=int(r['stage_a_targets']),
        non_application_physical_growth_bytes=(int(w_start['free_bytes'])-int(r['free_bytes']))-
            (int(r['application_bytes'])-int(w_start['application_bytes'])),
        blas_handles=int(r['blas'])))
exports=subprocess.check_output(['nm','-D','--defined-only',str(out/'probe.so')],text=True)
assert 'multi_penalty_gcv_cuda_optimize_batch_small64' not in exports
assert 'multi_penalty_gcv_cuda_finalize_selected_coefficients_small64' not in exports
write_new(out/'closure.json',dict(schema='l3-real-replacement-coexistence-closure-v1',
    status='STOP_L3_REPLACEMENT_POOL_INCREMENT',evidence_class='isolated diagnostic capacity rejection; not performance',
    native_sha256=sha(NATIVE),diagnostic_overlay_sha256=sha(out/'probe.so'),
    logging_tap_sha256=sha(extra/'tap.so'),protocol_sha256=sha(out/'protocol.json'),
    build_sha256=sha(out/'build.json'),attribution=a,W_canonical=w,
    W_non_application_physical_growth_after_window_release=shared_growth,
    accounting_note='W also retains approximately 1.13 GiB beyond application bytes after original handles release; first-use overhead must be separated in any future protocol. Allocator attribution is not established.',
    overlay_does_not_export_production_small64_math_entrypoints=True,
    failed_attempts={k:processes[k] for k in ['S','S-attribution']},
    S_optimizer_outputs_bitwise_exact=True,S_stage_a_qualified=False,S_full_canonical_qualified=False,
    S_result_published=False,threshold_changed=False,shape_changed=False,
    original_window_prepared_retained=False,production_sources_changed=False,production_native_changed=False,
    runtime_integration='NOT_YET_GO',accepted_canonical_median_sec=368.851,
    captured_cohort_status='GO_CAPTURED_L3_SLOTS',phase10='ACTIVE',candidate='NOT_FROZEN',holdout='SEALED',recommended=False))
write_new(out/'manifest.json',dict(schema='l3-replacement-coexistence-manifest-v1',
    files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()},
    tools={str(p.relative_to(ROOT)):sha(p) for p in sorted(TOOL.iterdir()) if p.is_file()},
    earlier_incomplete_builds={str(p.relative_to(out.parent)):sha(p) for p in sorted((out.parent/'attempt1').rglob('*')) if p.is_file()}))
print('STOP_L3_REPLACEMENT_POOL_INCREMENT',a,flush=True)
