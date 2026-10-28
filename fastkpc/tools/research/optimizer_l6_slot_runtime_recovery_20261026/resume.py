"""Resume the frozen L6 loop; never retry or remove a completed timing."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

root = Path(__file__).resolve().parents[4]
out = Path(sys.argv[1]).resolve()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
plan, protocol = read(out/'recovery-plan.json'), read(out/'protocol.json')
assert sha(out/'protocol.json') == plan['original_protocol_sha256']
assert all(sha(root/plan['original_root']/p) == h for p,h in plan['original_files'].items())
original = root/'fastkpc/tools/research/optimizer_l6_slot_runtime_20261026/campaign.py'
assert sha(original) == protocol['frozen_tools'][str(original.relative_to(root))]
for mapping in [protocol['frozen_tools'],protocol['frozen_dependencies']]:
    assert all(sha(root/p) == h for p,h in mapping.items())
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
    '--format=csv,noheader'],text=True).strip(), 'wait until both GPUs are idle'
source = original.read_text()
prefix = source[:source.index("with (out/(prefix+'protocol.json')).open('x') as protocol:")]
loop = source[source.index('for name, mode, reference, repeated in runs:'):]
loop = loop[:loop.index("print('PASS shared-host correctness/resource qualification; no performance acceptance'")]
scope = {'__file__':str(original),'__name__':'optimizer_l6_isolated_continuation'}
exec(compile(prefix,str(original),'exec'),scope)
assert not scope['qualification_only'] and not scope['shared_host']
remaining, invalidated, completed = [], [], []
for run in scope['runs']:
    name = run[0]
    report, process_path = out/name/'report.json', out/(name+'-process.json')
    if report.exists():
        r, p = read(report), read(process_path)
        assert r['gates'] and all(r['gates'].values()), 'never retry a numerical failure'
        assert p['exit_code'] == 0 and p['isolated'], 'completed timing is not valid'
        assert sha(out/name/'receipt.rds') == r['receipt_sha256']
        completed.append(name)
        continue
    if (out/name).exists() or process_path.exists():
        assert process_path.exists(), 'incomplete run lacks an auditable invalidation'
        p = read(process_path)
        assert p['exit_code'] != 0 and p['unexpected_gpu_pids'], 'only interference is retryable'
        assert not (out/name/'receipt.rds').exists() and not (out/name/'pending.rds').exists(), 'never discard a finished timing'
        index = 1
        while (out/'invalidated'/(name+'-attempt'+str(index))).exists(): index += 1
        dest = out/'invalidated'/(name+'-attempt'+str(index))
        dest.mkdir(parents=True)
        for path in [out/name,*[out/(name+suffix) for suffix in
                     ['.log','-gpu.csv','-gpu-processes.json','-process.json']]]:
            assert path.exists(), str(path)
            path.rename(dest/path.name)
        invalidated.append(dict(name=name,path=str(dest.relative_to(out)),
            process_sha256=sha(dest/process_path.name),completed_timing=False,reason='external GPU interference'))
    remaining.append(run)
assert remaining, 'campaign already complete'
assert completed == protocol['sequence'][:len(completed)], 'completed runs must be the original prefix'
assert [r[0] for r in remaining] == protocol['sequence'][len(completed):]
index = 1
while (out/('continuation-'+str(index)+'.json')).exists(): index += 1
artifact_tool = out/('continuation-'+str(index)+'-resume.py')
shutil.copyfile(__file__,artifact_tool)
record = dict(schema='optimizer-l6-isolated-continuation-v1',
    original_protocol_sha256=sha(out/'protocol.json'),campaign_sha256=sha(original),
    resume_tool_sha256=sha(__file__),archived_tool=str(artifact_tool.relative_to(out)),
    setup_code_sha256=hashlib.sha256(prefix.encode()).hexdigest(),
    run_loop_sha256=hashlib.sha256(loop.encode()).hexdigest(),
    valid_completed_runs=completed,remaining_sequence=[r[0] for r in remaining],invalidated=invalidated,
    native_sha256=scope['build']['native_sha256'],completed_timings_discarded=0)
with (out/('continuation-'+str(index)+'.json')).open('x') as stream:
    json.dump(record,stream,indent=2);stream.write('\n')
with (out/('continuation-'+str(index)+'-environment.log')).open('x') as log:
    subprocess.run(['nvidia-smi',
        '--query-gpu=index,uuid,name,driver_version,memory.total,power.limit,clocks.current.sm,temperature.gpu',
        '--format=csv'],stdout=log,stderr=subprocess.STDOUT,check=True)
scope.update(runs=remaining,frozen_tools=protocol['frozen_tools'],
             frozen_dependencies=protocol['frozen_dependencies'])
exec(compile(loop,str(original),'exec'),scope)
print('PASS continuation',','.join(r[0] for r in remaining),flush=True)
