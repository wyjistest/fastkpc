"""Resume only runs invalidated by external GPU interference, preserving all attempts.

Execute the frozen campaign's setup and run loop verbatim. Its numerical runner,
configuration, checks, sequence and performance gates are unchanged. Never retry
a completed numerical failure or discard a completed valid timing.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

tool=Path(__file__).resolve().parent
root=tool.parents[3]
out=Path(sys.argv[1]).resolve()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
protocol=json.loads((out/'protocol.json').read_text())
original=tool/'campaign.py'
assert sha(original)==protocol['frozen_tools'][str(original.relative_to(root))]
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
    '--format=csv,noheader'],text=True).strip(),'wait until both GPUs are idle'
source=original.read_text()
prefix=source[:source.index("with (out/'protocol.json').open('x') as protocol:")]
loop=source[source.index('for name, mode, reference, repeated in runs:'):]
loop=loop[:loop.index("print('PASS complete device-properties cache qualification',flush=True)")]
scope={'__file__':str(original),'__name__':'device_properties_cache_continuation'}
exec(compile(prefix,str(original),'exec'),scope)
assert all(sha(root/p)==h for p,h in protocol['frozen_tools'].items())
assert all(sha(root/p)==h for p,h in protocol['frozen_dependencies'].items())
remaining=[];invalidated=[];completed=[]
for run in scope['runs']:
    name=run[0]
    report=out/name/'report.json'
    process_path=out/(name+'-process.json')
    if report.exists():
        r=json.loads(report.read_text());p=json.loads(process_path.read_text())
        assert r['gates'] and all(r['gates'].values()),'never retry a numerical failure'
        assert p['exit_code']==0 and p['isolated'],'completed timing is not valid'
        assert sha(out/name/'receipt.rds')==r['receipt_sha256']
        completed.append(name)
        continue
    if (out/name).exists() or process_path.exists():
        assert process_path.exists(),'incomplete run lacks an auditable invalidation'
        p=json.loads(process_path.read_text())
        assert p['exit_code']!=0 and p['unexpected_gpu_pids'], 'only interference is retryable'
        assert not (out/name/'receipt.rds').exists() and not (out/name/'pending.rds').exists(),\
            'never discard a finished timing'
        base=out/'invalidated';base.mkdir(exist_ok=True)
        index=1
        while (base/(name+'-attempt'+str(index))).exists():index+=1
        dest=base/(name+'-attempt'+str(index));dest.mkdir()
        paths=[out/name,*[out/(name+suffix) for suffix in
               ['.log','-gpu.csv','-gpu-processes.json','-process.json']]]
        for path in paths:
            assert path.exists(),str(path)
            path.rename(dest/path.name)
        invalidated.append(dict(name=name,path=str(dest.relative_to(out)),
            process_sha256=sha(dest/process_path.name),reason='external GPU interference',
            completed_timing=False))
    remaining.append(run)
assert remaining,'campaign already complete'
index=1
while (out/('continuation-'+str(index)+'.json')).exists():index+=1
artifact_tool=out/('continuation-'+str(index)+'-resume.py')
shutil.copyfile(__file__,artifact_tool)
record=dict(schema='device-properties-cache-continuation-v1',
    reason='resume interrupted isolation campaign; no change to gates or configuration',
    original_protocol_sha256=sha(out/'protocol.json'),campaign_sha256=sha(original),
    resume_tool_sha256=sha(__file__),archived_tool=str(artifact_tool.relative_to(out)),
    setup_code_sha256=hashlib.sha256(prefix.encode()).hexdigest(),
    run_loop_sha256=hashlib.sha256(loop.encode()).hexdigest(),
    valid_completed_runs=completed,remaining_sequence=[r[0] for r in remaining],
    invalidated=invalidated,native_sha256=scope['build']['native_sha256'])
(out/('continuation-'+str(index)+'.json')).write_text(json.dumps(record,indent=2)+'\n')
with (out/('continuation-'+str(index)+'-environment.log')).open('x') as log:
    subprocess.run(['nvidia-smi',
        '--query-gpu=index,uuid,name,driver_version,memory.total,power.limit,clocks.current.sm,temperature.gpu',
        '--format=csv'],stdout=log,stderr=subprocess.STDOUT,check=True)
scope.update(runs=remaining,frozen_tools=protocol['frozen_tools'],
             frozen_dependencies=protocol['frozen_dependencies'])
exec(compile(loop,str(original),'exec'),scope)
print('PASS continuation',','.join(r[0] for r in remaining),flush=True)
