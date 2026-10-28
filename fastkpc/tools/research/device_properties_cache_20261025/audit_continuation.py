"""Bind interruption and recovery separately from the immutable original protocol."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

tool=Path(__file__).resolve().parent
root=tool.parents[3]
out=Path(sys.argv[1]).resolve()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
protocol=read(out/'protocol.json')
records=sorted(out.glob('continuation-[0-9]*.json'))
assert records,'no continuation to audit'
interrupted=[]
for path in records:
    record=read(path)
    assert record['original_protocol_sha256']==sha(out/'protocol.json')
    assert record['campaign_sha256']==sha(tool/'campaign.py')
    assert record['resume_tool_sha256']==sha(out/record['archived_tool'])
    assert record['native_sha256']==protocol['native_sha256']
    for attempt in record['invalidated']:
        directory=out/attempt['path']
        p=directory/(attempt['name']+'-process.json')
        assert sha(p)==attempt['process_sha256']
        process=read(p)
        assert process['unexpected_gpu_pids'] and not process['isolated']
        assert process['exit_code']!=0 and not attempt['completed_timing']
        assert not (directory/attempt['name']/'receipt.rds').exists()
        interrupted.append(attempt)
for name in protocol['sequence']:
    report=read(out/name/'report.json');process=read(out/(name+'-process.json'))
    assert process['isolated'] and process['exit_code']==0
    assert report['gates'] and all(report['gates'].values())
    assert sha(out/name/'receipt.rds')==report['receipt_sha256']
assert all(sha(root/p)==h for p,h in protocol['frozen_tools'].items())
shutil.copyfile(__file__,out/'audit-continuation.py')
receipt=dict(schema='device-properties-cache-recovery-audit-v1',status='PASS',
    original_protocol_sha256=sha(out/'protocol.json'),native_sha256=protocol['native_sha256'],
    original_tools_unchanged=True,performance_gates_unchanged=True,
    valid_full_runs=len(protocol['sequence']),valid_run_sequence=protocol['sequence'],
    invalidated_incomplete_attempts=interrupted,completed_timings_discarded=0,
    continuations={p.name:sha(p) for p in records},audit_tool_sha256=sha(__file__))
with (out/'recovery-audit.json').open('x') as stream:
    json.dump(receipt,stream,indent=2);stream.write('\n')
print('PASS recovery audit: six isolated completed runs; incomplete interference attempts preserved')
