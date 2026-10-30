"""Audit the declared gates, install only a passing capacity result, seal identities."""
import fcntl
import importlib.util
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build import ROOT,PARENT,BASE,TOOL,sha,read,write
support=TOOL.parent/'stage_b_work_pair_runtime_20261028/support.py'
spec=importlib.util.spec_from_file_location('jobs',support)
jobs=importlib.util.module_from_spec(spec);spec.loader.exec_module(jobs)
action=sys.argv[1];assert action in ['audit','install','seal']
build=read(BASE/'candidate/build.json');qa=read(BASE/'qualification/closure.json')
outer=read(BASE/'canonical/closure.json');protocol=read(BASE/'canonical/protocol.json')
assert read(PARENT/'closure.json')['status']=='PASS_BUILD_FIX_PARENT'
assert read(PARENT/'object-payload-comparison.json')['status']=='PASS'
assert qa['status']=='PASS_ROWS_PAYLOAD_FALLBACK_RESOURCES'
for mode in ['parent','full','incremental']:
    folder=PARENT/mode/'qualification'
    assert read(folder/'closure.json')['status']=='PASS'
    report=read(folder/'report.json')
    assert report['status']=='PASS' and all(all(r['gates'].values()) for r in report['reports'].values())
    for path,h in read(folder/'protocol.json')['dependencies'].items():
        if sha(path)!=h:
            archived=folder/'tools'/Path(path).name
            assert archived.is_file() and sha(archived)==h,path
for p in (PARENT/'full/regressions2').glob('*-process.json'):
    value=read(p);assert value['isolated'] and value['exit_code']==0,str(p)
for path,h in read(PARENT/'full/regressions2/protocol.json')['dependencies'].items():
    assert sha(path)==h,path
for p,h in build['source_sha256'].items():assert sha(ROOT/p)==h,p
for p,h in protocol['dependencies'].items():assert sha(p)==h,p
for p,h in qa['dependencies'].items():assert sha(p)==h,p
for name,h in build['objects'].items():assert sha(BASE/'candidate/objects'/name)==h,name
assert sha(build['native_path'])==build['native_sha256']
for directory in [BASE/'qualification',BASE/'canonical',PARENT/'canonical']:
    for p in directory.glob('*-process.json'):
        value=read(p)
        isolation_ok=value['isolated'] or (directory in [BASE/'qualification',PARENT/'canonical'] and
            value.get('numerical_only') and value.get('gpu1_exclusive') and not value['unexpected'])
        assert isolation_ok and value['exit_code']==0,str(p)
assert [(r['variant'],r['cycle']) for r in outer['runs']]==[tuple(x) for x in protocol['sequence']]
reports={}
for row in outer['runs']:
    path=Path(row['report']);r=read(path)
    assert all(r['gates'].values()) and r['elapsed_sec']==row['elapsed_sec']
    assert r['native_sha256']==outer['native_sha256'][row['variant']]
    assert sha(path.parent/'receipt.rds')==r['receipt_sha256']
    assert r['comparisons']['changed_ci_count']==r['comparisons']['changed_pmax_count']==0
    reports[(row['variant'],row['cycle'])]=r
medians={v:statistics.median(reports[(v,i)]['elapsed_sec'] for i in [1,2,3]) for v in ['off','on']}
ratio=medians['on']/medians['off'];assert medians==outer['medians'] and ratio==outer['ratio']
gate=protocol['gate'];assert gate==outer['gate']
assert ratio<=gate['median_nonregression_ratio_max']
for cycle in [1,2,3]:
    a,b=reports[('off',cycle)],reports[('on',cycle)]
    assert b['elapsed_sec']/a['elapsed_sec']<=gate['each_pair_ratio_max']
    assert len(a['level_component_prefill'])==len(b['level_component_prefill'])==7
    for x,y in zip(a['level_component_prefill'],b['level_component_prefill']):
        assert x['level']==y['level']
        x,y=x['structured'],y['structured']
        assert x['solver_workspace_bytes']==y['solver_workspace_bytes']
        assert x['pool_device_bytes']-y['pool_device_bytes']==gate['measured_pool_bytes_reduction']
    for pool in ['l3_pool','l4_pool','l6_pool']:
        assert b['stages'][pool]['minimum_free_bytes']-a['stages'][pool]['minimum_free_bytes']>=gate['minimum_free_improvement_bytes']
assert outer['status']=='GO_CAPACITY'
assert not jobs.processes(),'external GPU task; installation/sealing deferred'

if action=='audit':
    write(BASE/'audit.json',dict(status='PASS_RECOMPUTED',medians=medians,ratio=ratio,
        numerical_behavior='PASS',capacity='PASS',outer_nonregression='PASS',speed=outer['speed'],
        maximum_authority_error=max(max(r['oracle_errors']) for r in reports.values()),
        candidate_minimum_free_bytes=min(reports[('on',i)]['stages'][p]['minimum_free_bytes']
            for i in [1,2,3] for p in ['l3_pool','l4_pool','l6_pool']),
        scope='full rebuild bridge; one-TU capacity change; six canonical development runs',
        no_new_full_real_dataset_or_final_pdag_qualification=True))
    print('PASS recomputed audit',medians,flush=True)
elif action=='install':
    assert read(BASE/'audit.json')['status']=='PASS_RECOMPUTED'
    original=read(PARENT/'parent/identity.json');dest=ROOT/'fastkpc/build'
    with (dest/'fastkpc_cuda.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert sha(dest/'fastkpc_cuda.so')==original['native_sha256']
        for name,h in original['objects'].items():assert sha(dest/name)==h,name
        mapping={name:BASE/'candidate/objects'/name for name in build['objects']}
        for name in ['recorded_dbdsqr.inc','recorded_dbdsqr_fixed_vt.inc','fastkpc_cuda.so']:
            mapping[name]=BASE/'candidate'/name
        for name,source in mapping.items():
            pending=dest/(name+'.rows-qualified.tmp');assert not pending.exists()
            shutil.copy2(source,pending);assert sha(pending)==sha(source)
            os.replace(pending,dest/name)
        write(BASE/'installation.json',dict(status='PASS_INSTALLED',epoch=time.time(),
            previous_native_sha256=original['native_sha256'],native_sha256=sha(dest/'fastkpc_cuda.so'),
            linked_objects=build['objects'],directory_object_inventory={p.name:sha(p) for p in dest.glob('*.o')},
            defaults_changed=False))
    print('PASS installed capacity optimization',build['native_sha256'],flush=True)
else:
    installation=read(BASE/'installation.json')
    assert sha(ROOT/'fastkpc/build/fastkpc_cuda.so')==installation['native_sha256']==build['native_sha256']
    for name,h in build['objects'].items():assert sha(ROOT/'fastkpc/build'/name)==h,name
    tools={}
    for folder in [TOOL,TOOL.parent/'build_consistency_20261029']:
        for p in folder.iterdir():
            if p.is_file():
                target=BASE/'tools'/folder.name/p.name;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(p,target);tools[str(p.relative_to(ROOT))]=sha(p)
    write(BASE/'closure.json',dict(status='GO_CAPACITY',machine_audit='PASS',
        numerical_behavior='PASS',capacity='PASS',outer_nonregression='PASS',speed=outer['speed'],
        native_sha256=build['native_sha256'],parent_native_sha256=build['previous_native_sha256'],
        medians=medians,ratio=ratio,pool_bytes_saved=gate['measured_pool_bytes_reduction'],
        parent_closure_sha256=sha(PARENT/'closure.json'),build_sha256=sha(BASE/'candidate/build.json'),
        qualification_sha256=sha(BASE/'qualification/closure.json'),canonical_sha256=sha(BASE/'canonical/closure.json'),
        audit_sha256=sha(BASE/'audit.json'),installation_sha256=sha(BASE/'installation.json'),
        evidence_class='full-build parent plus one rebuilt TU; repeated source-closure development',
        production_default=False,phase10='ACTIVE',candidate='NOT_FROZEN',holdout='SEALED',recommended=False,
        limitations=['not clean committed producer or promotion','no new full real-input or final-PDAG campaign',
            'capacity is checkpoint/sample evidence, not continuous peak proof'],
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    files={str(p.relative_to(BASE)):sha(p) for p in BASE.rglob('*') if p.is_file() and p.name!='manifest.json'}
    parent_files={str(p.relative_to(PARENT)):sha(p) for p in PARENT.rglob('*') if p.is_file()}
    write(BASE/'manifest.json',dict(schema='structured-rows-capacity-closure-v1',native_sha256=build['native_sha256'],
        production_source_sha256=build['source_sha256'],tools=tools,files=files,parent_files=parent_files))
    print('PASS sealed closure',sha(BASE/'closure.json'),'manifest',sha(BASE/'manifest.json'),flush=True)
