"""Recompute the gates, optionally install the qualified default-off binary, then seal evidence."""
import fcntl
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from support import ROOT,TOOL,BASE,sha,read,write,processes

attempt,qualification,action=sys.argv[1:]
assert action in ('audit','install','seal')
out=BASE/attempt;qa=out/qualification;canonical=out/'canonical'
build=read(out/'build.json');baseline=read(BASE/'baseline/identity.json')
assert sha(out/'fastkpc_cuda.so')==build['native_sha256']
for p,h in build['source_sha256'].items():assert sha(ROOT/p)==h,p
for p,h in build['objects'].items():assert sha(p)==h,p
for p,h in read(out/'device-build.json')['source_sha256'].items():assert sha(ROOT/'fastkpc/src'/p)==h,p
assert read(qa/'closure.json')['status']=='PASS_RUNTIME_QUALIFICATION'
assert read(qa/'static-gates.json')['status']=='PASS'
outer=read(canonical/'closure.json');protocol=read(canonical/'protocol.json')
assert len(outer['runs'])==6 and [(r['variant'],r['cycle']) for r in outer['runs']]==[tuple(x) for x in protocol['sequence']]
for path in [qa/'run-protocol.json',canonical/'protocol.json']:
    for p,h in read(path)['dependencies'].items():assert sha(p)==h,p
for directory in [qa,canonical]:
    for path in directory.glob('*-process.json'):
        p=read(path);assert p['isolated'] and p['exit_code']==0,str(path)
reports={}
for row in outer['runs']:
    path=Path(row['report']);r=read(path)
    assert all(r['gates'].values()) and r['native_sha256']==build['native_sha256']
    assert r['elapsed_sec']==row['elapsed_sec'] and sha(path.parent/'receipt.rds')==r['receipt_sha256']
    assert r['comparisons']['changed_ci_count']==r['comparisons']['changed_pmax_count']==0
    assert sum(x['count'] for x in r['comparisons']['by_level'])==240498
    reports[(row['variant'],row['cycle'])]=r
medians={v:statistics.median(reports[(v,i)]['elapsed_sec'] for i in [1,2,3]) for v in ['off','on']}
ratio=medians['on']/medians['off']
wins=sum(reports[('on',i)]['elapsed_sec']<reports[('off',i)]['elapsed_sec'] for i in [1,2,3])
assert medians==outer['medians'] and ratio==outer['ratio'] and wins==outer['paired_wins']
assert ratio<=protocol['performance_gate']['on_median_ratio_max'] and wins==protocol['performance_gate']['paired_wins']
assert outer['status']=='GO_DEFAULT_OFF_RUNTIME_DEVELOPMENT'
assert not processes(),'GPU task active; closure/install deferred'

if action=='audit':
    derived={}
    for v in ['off','on']:
        rows=[reports[(v,i)] for i in [1,2,3]]
        derived[v]=dict(outer_median_sec=medians[v],
          l45_optimizer_sum_median_sec=statistics.median(sum(x['host_sec'] for x in r['stages']['optimizer_by_level']
            if x['level'] in [4,5]) for r in rows),
          optimizer_by_level_median_sec={str(level):statistics.median(next(x['host_sec'] for x in r['stages']['optimizer_by_level']
            if x['level']==level) for r in rows) for level in range(1,9)},
          stage_medians_sec={field:statistics.median(r['stages'][field] for r in rows)
            for field in ['optimizer_sec','stage_a_sec','stage_b_sec','native_setup_sec','consumer_dcov_sec']},
          gate_counts=[len(r['gates']) for r in rows],
          maximum_authority_error=max(max(r['oracle_errors']) for r in rows),
          minimum_capacity_checkpoint_free_bytes=min(r['stages'][pool]['minimum_free_bytes']
            for r in rows for pool in ['l3_pool','l4_pool','l6_pool']),
          minimum_l5_post_creation_free_bytes=min(r['stages']['l5_slots']['free_after_create'] for r in rows),
          vector_postback_execution=rows[0]['vector_postback_execution'])
    pairs=[]
    for i in [1,2,3]:
        off,on=reports[('off',i)],reports[('on',i)]
        def level_time(report,levels):
            return sum(x['host_sec'] for x in report['stages']['optimizer_by_level'] if x['level'] in levels)
        saving=off['elapsed_sec']-on['elapsed_sec']
        l45=level_time(off,[4,5])-level_time(on,[4,5])
        l7=level_time(off,[7])-level_time(on,[7])
        l8=level_time(off,[8])-level_time(on,[8])
        pairs.append(dict(cycle=i,outer_saving_sec=saving,l45_optimizer_saving_sec=l45,
          l7_optimizer_difference_sec=l7,l8_optimizer_difference_sec=l8,
          remaining_difference_sec=saving-l45-l7-l8))
    write(out/'analysis.json',dict(status='PASS_RECOMPUTED_AUDIT',variants=derived,ratio=ratio,paired_wins=wins,pairs=pairs,
      paired_savings_sec=[reports[('off',i)]['elapsed_sec']-reports[('on',i)]['elapsed_sec'] for i in [1,2,3]],
      median_saving_sec=medians['off']-medians['on'],
      capacity_scope='original raw checkpoints; not continuous memory sampling or larger-data qualification',
      compilation_scope='two new host objects plus independent ON CUDA object; 41 authenticated baseline objects reused'))
    print('PASS recomputed audit',medians,'ratio',ratio,flush=True)
elif action=='install':
    assert read(out/'analysis.json')['status']=='PASS_RECOMPUTED_AUDIT'
    dest=ROOT/'fastkpc/build'
    with (dest/'fastkpc_cuda.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert sha(dest/'fastkpc_cuda.so')==baseline['native_sha256'],'installed native changed externally'
        for name,h in baseline['objects'].items():assert sha(dest/name)==h,name
        mapping={'mgcv_multi_penalty_gcv_capacity.o':'mgcv_multi_penalty_gcv_capacity.o',
          'full_cuda_ci_one_call.o':'full_cuda_ci_one_call.o',
          'mgcv_multi_penalty_gcv_postback.o':'postback.o','fastkpc_cuda.so':'fastkpc_cuda.so'}
        for target,source in mapping.items():
            pending=dest/(target+'.postback-qualified.tmp');assert not pending.exists()
            shutil.copy2(out/source,pending);assert sha(pending)==sha(out/source)
            os.replace(pending,dest/target)
        installed={p.name:sha(p) for p in dest.glob('*.o')}
        write(out/'installation.json',dict(status='PASS_INSTALLED_DEFAULT_OFF',epoch=time.time(),
          previous_native_sha256=baseline['native_sha256'],native_sha256=sha(dest/'fastkpc_cuda.so'),
          objects=installed,environment_defaults_changed=False,source_closure_sha256=sha(out/'build.json')))
    print('PASS installed qualified default-off runtime',build['native_sha256'],flush=True)
else:
    installed=read(out/'installation.json')
    assert sha(ROOT/'fastkpc/build/fastkpc_cuda.so')==installed['native_sha256']==build['native_sha256']
    for name,h in installed['objects'].items():assert sha(ROOT/'fastkpc/build'/name)==h,name
    snapshot=out/'tools';snapshot.mkdir()
    tools={}
    for p in TOOL.iterdir():
        if p.is_file():shutil.copyfile(p,snapshot/p.name);tools[str(p.relative_to(ROOT))]=sha(p)
    # Earlier source identities remain bound to their pre-edit snapshot.
    for p,h in baseline['production_source_sha256'].items():assert sha(BASE/'baseline'/p)==h,p
    write(out/'closure.json',dict(status=outer['status'],machine_audit='PASS',native_sha256=build['native_sha256'],
      accepted_opt_in_development_median_sec=medians['on'],same_native_off_median_sec=medians['off'],
      paired_wins=wins,ratio=ratio,baseline_identity_sha256=sha(BASE/'baseline/identity.json'),
      build_sha256=sha(out/'build.json'),qualification_sha256=sha(qa/'closure.json'),
      canonical_sha256=sha(canonical/'closure.json'),installation_sha256=sha(out/'installation.json'),
      production_default=False,phase10='ACTIVE',candidate='NOT_FROZEN',holdout='SEALED',recommended=False,
      evidence_class='incremental runtime source closure; same-native repeated canonical development',
      limitations=['no clean-checkout full rebuild or promotion','no new real-data or final-PDAG campaign',
        'capacity checks are discrete existing checkpoints; qualification covers canonical and captured shapes'],
      preserved_failed_harness_build='qualification/build-3.log: diagnostic cudaMalloc pointer cast; no GPU execution',
      head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}
    write(out/'manifest.json',dict(schema='postback-runtime-source-closure-manifest-v1',
      native_sha256=build['native_sha256'],production_source_sha256=build['source_sha256'],tools=tools,
      frozen_baseline_identity_sha256=sha(BASE/'baseline/identity.json'),files=files))
    print('PASS sealed closure',sha(out/'closure.json'),'manifest',sha(out/'manifest.json'),flush=True)
