"""Audit the predeclared L8 timing campaign separately from capture and traces."""
import hashlib,json,statistics,sys
from pathlib import Path
root=Path(__file__).resolve().parents[4];out=Path(sys.argv[1]).resolve()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
names=['off1','on1','on2','off2','off3','on3'];rows={};protocols={};processes={}
for name in names:
    r=read(out/name/'report.json');h=read(out/name/'hook.json');p=read(out/(name+'-process.json'));protocol=read(out/(name+'-protocol.json'))
    assert all(r['gates'].values()) and all(h['gates'].values()) and p['exit_code']==0 and not p['unexpected_gpu_pids']
    assert protocol['kind']=='timing' and not h['capture'] and h['preload']==name.startswith('on')
    assert sha(out/name/'receipt.rds')==r['receipt_sha256']==h['receipt_sha256']
    assert h['after']['capture_diagnostic_d2h_bytes']==0
    for path,digest in protocol['tools'].items():assert sha(root/path)==digest,path
    rows[name]=dict(elapsed_sec=r['elapsed_sec'],levels_sec={str(l['level']):l['elapsed_ms']/1000 for l in r['level_trace']},
      optimizer_by_level=r['stages']['optimizer_by_level'],stages=r['stages'],hook=h['after'],oracle_errors=r['oracle_errors'])
    protocols[name]=protocol;processes[name]=p
for a,b in zip(names,names[1:]):assert processes[a]['end_epoch']<processes[b]['start_epoch']
for key in ['native_sha256','observer_sha256','reference_sha256','tools']:
    assert all(protocols[n][key]==protocols[names[0]][key] for n in names)
env0={k:v for k,v in protocols[names[0]]['environment'].items() if k!='FASTKPC_LATE_L8_PRELOAD'}
assert all({k:v for k,v in protocols[n]['environment'].items() if k!='FASTKPC_LATE_L8_PRELOAD'}==env0 for n in names)
summaries={}
for mode in ['off','on']:
    rs=[rows[mode+str(i)] for i in [1,2,3]];times=[r['elapsed_sec'] for r in rs]
    summaries[mode]=dict(median_sec=statistics.median(times),range_sec=max(times)-min(times),elapsed_sec=times,
       level_medians_sec={str(l):statistics.median(r['levels_sec'][str(l)] for r in rs) for l in range(9)},
       preload_median_ms=statistics.median(r['hook']['preload_ms'] for r in rs))
off,on=summaries['off'],summaries['on'];saved=off['median_sec']-on['median_sec']
threshold=max(2,.005*off['median_sec'],off['range_sec'],on['range_sec'])
l8_saved=off['level_medians_sec']['8']-on['level_medians_sec']['8']
pairs=[dict(pair=i,saved_sec=rows['off'+str(i)]['elapsed_sec']-rows['on'+str(i)]['elapsed_sec'],
            l8_saved_sec=rows['off'+str(i)]['levels_sec']['8']-rows['on'+str(i)]['levels_sec']['8']) for i in [1,2,3]]
wins=sum(p['saved_sec']>0 for p in pairs)
go=saved>=threshold and wins==3 and l8_saved>=1
status='GO_L8_PRELOAD_CAPTURED_DEVELOPMENT' if go else 'CONDITIONAL_L8_PRELOAD' if saved>0 else 'STOP_L8_PRELOAD_PERFORMANCE'
trace=read(out/'trace-audit.json');capture=read(out/'capture-closure.json')
assert trace['status']=='PASS_UNCHANGED_L8_DEVICE_WORK' and capture['status']=='PASS'
for path,digest in trace['references'].items():assert sha(out/path)==digest,path
trace_completeness={}
for name in ['trace-off','trace-on']:
    t=out/(name+'-trace')
    counts={k:int(v) for k,v in (line.split() for line in (t/'ktrace-summary.txt').read_text().splitlines())}
    assert counts['dropped']==0 and counts['kernels']==569524
    assert (t/'kernels.bin').stat().st_size==counts['kernels']*54
    assert (t/'runtime.bin').stat().st_size==counts['runtime']*28
    trace_completeness[name]=counts
closure=dict(schema='l8-preload-closure-v1',status=status,hard_gates='PASS',
  evidence_class='same native, research host interposer, GPU1 exclusive; shared-host status explicit; not runtime integration or promotion',
  native_sha256=protocols[names[0]]['native_sha256'],observer_sha256=protocols[names[0]]['observer_sha256'],
  summaries=summaries,pairs=pairs,paired_wins=wins,saved_median_sec=saved,saved_percent=100*saved/off['median_sec'],
  required_saving_sec=threshold,l8_saved_median_sec=l8_saved,
  timing_all_runs_isolated=all(p['isolated'] for p in processes.values()),processes=processes,runs=rows,
  capture_sha256=sha(out/'capture-closure.json'),trace_sha256=sha(out/'trace-audit.json'),trace_completeness=trace_completeness,
  production_native_changed=False,runtime_integration='NOT_YET_GO',accepted_canonical_median_sec=407.531,
  phase10='ACTIVE',candidate='NOT_FROZEN',holdout='SEALED',recommended=False)
with (out/'closure.json').open('x') as stream:json.dump(closure,stream,indent=2);stream.write('\n')
tool_dirs=[root/'fastkpc/tools/research/late_optimizer_20261025',Path(__file__).resolve().parent]
manifest=dict(schema='late-optimizer-manifest-v1',native_sha256=closure['native_sha256'],
  tools={str(p.relative_to(root)):sha(p) for d in tool_dirs for p in sorted(d.iterdir()) if p.is_file()},
  files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'})
with (out/'manifest.json').open('x') as stream:json.dump(manifest,stream,indent=2);stream.write('\n')
print(json.dumps({k:closure[k] for k in ['status','summaries','paired_wins','saved_median_sec','required_saving_sec','l8_saved_median_sec','timing_all_runs_isolated']},indent=2))
print('manifest',sha(out/'manifest.json'))
