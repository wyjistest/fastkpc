"""Close the original-window status-wait qualification without promotion."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

root=Path(__file__).resolve().parents[4]
out=Path(sys.argv[1]).resolve()
tool=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
build=json.loads((out/'build.json').read_text())
checks=json.loads((out/'checks.json').read_text())
assert checks['status']=='PASS' and checks['native_sha256']==build['native_sha256']
assert sha(root/'fastkpc/build/fastkpc_cuda.so')==build['native_sha256']
assert all(sha(root/p)==h for p,h in build['source_sha256'].items())
names=['off1','on1','on2','off2']
reports={name:json.loads((out/name/'report.json').read_text()) for name in names}
processes={name:json.loads((out/(name+'-process.json')).read_text()) for name in names}
for name,r in reports.items():
    assert r['gates'] and all(r['gates'].values()),name
    assert r['native_sha256']==build['native_sha256']
    assert r['comparisons']['changed_ci_count']==r['comparisons']['changed_pmax_count']==0
    process=processes[name]
    assert process['exit_code']==0 and process['selected_gpu_isolated'] and not process['unexpected_gpu_pids']
    assert process['isolated'] or process['shared_host_allowed']
assert len({p['physical_gpu'] for p in processes.values()})==1
device=processes['off1']['physical_gpu']
isolated=all(p['isolated'] for p in processes.values())
medians={mode:statistics.median(reports[n]['elapsed_sec'] for n in names if n.startswith(mode))
         for mode in ['off','on']}
ratio=medians['on']/medians['off']
neutral=ratio<=1.03
memory={}
for name in names:
    with (out/(name+'-gpu.csv')).open() as stream:
        rows=list(csv.reader(stream))
    g0=[r for r in rows if len(r)==7 and int(r[1])==device]
    g1=[r for r in rows if len(r)==7 and int(r[1])!=device]
    assert g0 and g1
    memory[name]=dict(selected_gpu_peak_mib=max(int(r[2]) for r in g0),
                      other_gpu_peak_utilization=max(int(r[3]) for r in g1))
closure=dict(schema='optimizer-status-wait-closure-v1',
    status=('PASS_ORIGINAL_WINDOW_OPT_IN' if isolated else 'PASS_SHARED_HOST_ORIGINAL_WINDOW_OPT_IN')
           if neutral else 'STOP_ORIGINAL_WINDOW_REGRESSION',
    evidence_class=('uncommitted full-source-closure development; two runs per mode; not promotion; '+
                    ('isolated' if isolated else 'shared host; other GPU has external compute')),
    physical_gpu=device,isolated=isolated,processes=processes,
    native_sha256=build['native_sha256'],source_sha256=build['source_sha256'],
    hard_gates='PASS',all_ci_count=240498,ci_and_pmax_bitwise_exact=True,
    outer_medians_sec=medians,enabled_over_disabled=ratio,
    no_material_regression_limit=1.03,no_material_regression_pass=neutral,
    elapsed_sec={n:reports[n]['elapsed_sec'] for n in names},
    max_authority_error=max(max(r['oracle_errors'].values()) if isinstance(r['oracle_errors'],dict)
                            else max(r['oracle_errors']) for r in reports.values()),
    optimizer_medians_sec={mode:statistics.median(reports[n]['stages']['optimizer_sec']
        for n in names if n.startswith(mode)) for mode in ['off','on']},
    memory=memory,configuration_default=False,scheduler_changed=False,
    runtime_rolling='NOT_YET_GO',phase10='ACTIVE',candidate='NOT_FROZEN',
    holdout='SEALED',recommended=False,new_real_data_campaign=False,final_pdag_campaign=False)
(out/'closure.json').write_text(json.dumps(closure,indent=2)+'\n')
manifest=dict(schema='optimizer-status-wait-manifest-v1',
    native_sha256=build['native_sha256'],source_sha256=build['source_sha256'],
    tools={str(p.relative_to(root)):sha(p) for p in sorted(tool.iterdir()) if p.is_file()},
    files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()
           and p.name!='manifest.json' and not any(x in p.relative_to(out).parts for x in ['build-source','baseline'])},
    references={str(p.relative_to(root)):sha(p) for p in [
        root/'fastkpc/artifacts/full_cuda_ci/optimizer_d2h_review_20261023/manifest.json',
        root/'fastkpc/artifacts/full_cuda_ci/l2_prefill_runtime_20261022/manifest.json']})
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(closure,indent=2))
print('manifest SHA256',sha(out/'manifest.json'))
