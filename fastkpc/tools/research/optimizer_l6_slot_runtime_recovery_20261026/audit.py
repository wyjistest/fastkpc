"""Audit preserved L6 attempts, frozen inputs and the exact planned run sequence."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

root = Path(__file__).resolve().parents[4]
out = Path(sys.argv[1]).resolve()
assert len(sys.argv) == 2 or (len(sys.argv) == 3 and sys.argv[2] == '--pending')
pending = len(sys.argv) == 3
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
plan, protocol = read(out/'recovery-plan.json'), read(out/'protocol.json')
assert sha(out/'protocol.json') == plan['original_protocol_sha256']
assert sha(out/'recovery-prepare.py') == plan['prepare_tool_sha256']
assert all(sha(root/plan['original_root']/p) == h for p,h in plan['original_files'].items())
build = read(out/'build.json')
assert sha(root/'fastkpc/build/fastkpc_cuda.so') == build['native_sha256'] == protocol['native_sha256']
for mapping in [build['source_sha256'],protocol['frozen_tools'],protocol['frozen_dependencies']]:
    assert all(sha(root/p) == h for p,h in mapping.items())
invalidated = list(plan['invalidated'])
records = sorted(out.glob('continuation-[0-9]*.json'))
campaign = root/'fastkpc/tools/research/optimizer_l6_slot_runtime_20261026/campaign.py'
source = campaign.read_text()
prefix = source[:source.index("with (out/(prefix+'protocol.json')).open('x') as protocol:")]
loop = source[source.index('for name, mode, reference, repeated in runs:'):]
loop = loop[:loop.index("print('PASS shared-host correctness/resource qualification; no performance acceptance'")]
for path in records:
    record = read(path)
    assert record['original_protocol_sha256'] == sha(out/'protocol.json')
    assert record['campaign_sha256'] == sha(campaign)
    assert record['resume_tool_sha256'] == sha(out/record['archived_tool'])
    assert record['native_sha256'] == protocol['native_sha256']
    assert record['setup_code_sha256'] == hashlib.sha256(prefix.encode()).hexdigest()
    assert record['run_loop_sha256'] == hashlib.sha256(loop.encode()).hexdigest()
    assert record['completed_timings_discarded'] == 0
    invalidated.extend(record['invalidated'])
for attempt in invalidated:
    directory = out/attempt['path']; name = attempt['name']
    assert sha(directory/(name+'-process.json')) == attempt['process_sha256']
    process = read(directory/(name+'-process.json'))
    assert process['exit_code'] != 0 and process['unexpected_gpu_pids'] and not process['isolated']
    assert attempt['completed_timing'] is False
    assert all(not (directory/name/p).exists() for p in ['report.json','receipt.rds','pending.rds'])
completed, elapsed, references = [], {}, {}
for name in protocol['sequence']:
    if not (out/name/'report.json').exists():
        assert pending, 'full recovery is not complete'
        continue
    r, p = read(out/name/'report.json'), read(out/(name+'-process.json'))
    assert p['exit_code'] == 0 and p['isolated'] and not p['unexpected_gpu_pids']
    assert r['gates'] and all(r['gates'].values())
    assert r['native_sha256'] == p['native_sha256'] == protocol['native_sha256']
    assert r['source_sha256'] == build['source_sha256']
    assert sha(out/name/'receipt.rds') == r['receipt_sha256']
    ref = root/p['command'][-2]
    assert sha(ref) == r['reference_sha256']
    references[str(ref.relative_to(root))] = sha(ref)
    if name in plan['valid_completed_runs']:
        old = plan['valid_completed_runs'][name]
        assert r['receipt_sha256'] == old['receipt_sha256']
        assert sha(out/name/'report.json') == old['report_sha256']
        assert r['elapsed_sec'] == old['elapsed_sec']
    completed.append(name); elapsed[name] = r['elapsed_sec']
assert completed == protocol['sequence'][:len(completed)]
assert pending or len(completed) == 6
index = 1
while pending and (out/('pending-audit-'+str(index)+'.json')).exists(): index += 1
name = 'pending-audit-'+str(index) if pending else 'recovery-audit'
assert not (out/(name+'.json')).exists()
shutil.copyfile(__file__,out/(name+'.py'))
receipt = dict(schema='optimizer-l6-isolated-recovery-audit-v1',
    status='PASS_PRESERVED_PENDING_ISOLATION' if pending else 'PASS',
    original_protocol_sha256=sha(out/'protocol.json'),native_sha256=protocol['native_sha256'],
    original_tree_unchanged=True,original_tools_unchanged=True,performance_gates_unchanged=True,
    valid_full_runs=len(completed),valid_run_sequence=completed,elapsed_sec_by_run=elapsed,
    remaining_sequence=protocol['sequence'][len(completed):],
    invalidated_incomplete_attempts=invalidated,completed_timings_discarded=0,
    continuations={p.name:sha(p) for p in records},audit_tool_sha256=sha(__file__),references=references,
    files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
with (out/(name+'.json')).open('x') as stream:
    json.dump(receipt,stream,indent=2);stream.write('\n')
print(receipt['status'],len(completed),'complete; remaining',receipt['remaining_sequence'])
print(name+' SHA256',sha(out/(name+'.json')))
