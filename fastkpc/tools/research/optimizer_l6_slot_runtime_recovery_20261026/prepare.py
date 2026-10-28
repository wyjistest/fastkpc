"""Prepare a fresh recovery tree without changing the interrupted evidence."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

root = Path(__file__).resolve().parents[4]
original, out = (Path(p).resolve() for p in sys.argv[1:])
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
assert not out.exists(), 'use a new continuation directory'
assert not out.is_relative_to(original), 'continuation must be outside the original tree'
protocol = read(original/'protocol.json')
assert protocol['sequence'] == ['off1','on1','on2','off2','off3','on3']
assert protocol['performance_eligible'] is True
assert protocol['outer_ratio_limit'] == .98 and protocol['l6_optimizer_ratio_limit'] == .90
assert protocol['required_pair_wins'] == 3
build = read(original/'build.json')
assert sha(root/'fastkpc/build/fastkpc_cuda.so') == build['native_sha256'] == protocol['native_sha256']
for mapping in [build['source_sha256'],protocol['frozen_tools'],protocol['frozen_dependencies']]:
    assert all(sha(root/p) == h for p,h in mapping.items())
completed = {}
for name in protocol['sequence'][:4]:
    r, p = read(original/name/'report.json'), read(original/(name+'-process.json'))
    assert p['exit_code'] == 0 and p['isolated'] and not p['unexpected_gpu_pids']
    assert r['gates'] and all(r['gates'].values())
    assert sha(original/name/'receipt.rds') == r['receipt_sha256']
    completed[name] = dict(elapsed_sec=r['elapsed_sec'],receipt_sha256=r['receipt_sha256'],
                           report_sha256=sha(original/name/'report.json'))
process = read(original/'off3-process.json')
assert process['exit_code'] != 0 and process['unexpected_gpu_pids'] and not process['isolated']
assert all(not (original/'off3'/p).exists() for p in ['receipt.rds','pending.rds','report.json'])
assert not (original/'on3').exists()
inventory = {str(p.relative_to(original)):sha(p) for p in sorted(original.rglob('*')) if p.is_file()}
shutil.copytree(original, out, copy_function=shutil.copy2)
assert all(sha(out/p) == h for p,h in inventory.items())
dest = out/'invalidated/off3-attempt1'
dest.mkdir(parents=True)
for name in ['off3','off3.log','off3-gpu.csv','off3-gpu-processes.json','off3-process.json']:
    (out/name).rename(dest/name)
assert all(sha(original/p) == h for p,h in inventory.items())
plan = dict(schema='optimizer-l6-isolated-recovery-plan-v1',status='READY_PENDING_IDLE_GPUS',
    original_root=str(original.relative_to(root)),original_files=inventory,
    original_protocol_sha256=sha(out/'protocol.json'),native_sha256=protocol['native_sha256'],
    valid_completed_runs=completed,remaining_sequence=['off3','on3'],completed_timings_discarded=0,
    invalidated=[dict(name='off3',path=str(dest.relative_to(out)),
        process_sha256=sha(dest/'off3-process.json'),completed_timing=False,reason='external GPU0 interference')],
    prepare_tool_sha256=sha(__file__))
shutil.copyfile(__file__,out/'recovery-prepare.py')
with (out/'recovery-plan.json').open('x') as stream:
    json.dump(plan,stream,indent=2);stream.write('\n')
print('PASS preparation: original unchanged; four completed receipts retained; only OFF3/ON3 remain')
print('recovery plan SHA256',sha(out/'recovery-plan.json'))
