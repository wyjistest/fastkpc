"""Complete the pending pair on exclusive GPU1 with shared-host evidence labels.

Usage: python3 campaign.py ORIGINAL_ARTIFACT NEW_SUPPLEMENT_ARTIFACT
The user authorized continuing while GPU0 is occupied. The original protocol,
four isolated runs and interrupted attempt remain immutable. Reuse its exact
run loop and hard gates, allowing foreign compute only on the other GPU.
Never pool these timings into the original isolated performance gate.
"""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

assert len(sys.argv) == 3, __doc__
supplement_tool = Path(__file__).resolve().parent
original_tool = supplement_tool.parent
root = original_tool.parents[3]
parent = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
original = original_tool / 'campaign.py'
protocol = read(parent / 'protocol.json')
assert sha(original) == protocol['frozen_tools'][str(original.relative_to(root))]
assert all(sha(root / p) == h for p, h in protocol['frozen_tools'].items())
assert all(sha(root / p) == h for p, h in protocol['frozen_dependencies'].items())
assert not out.exists(), 'preserve every previous attempt'

# Use the frozen setup, references and numerical execution loop verbatim.
source = original.read_text()
prefix = source[:source.index("with (out/'protocol.json').open('x') as protocol:")]
loop = source[source.index('for name, mode, reference, repeated in runs:'):]
loop = loop[:loop.index("print('PASS complete device-properties cache qualification',flush=True)")]
scope = {'__file__': str(original), '__name__': 'device_properties_shared_host'}
exec(compile(prefix, str(original), 'exec'), scope)
assert scope['out'] == parent
runs = [r for r in scope['runs'] if r[0] in ('off3', 'on3')]
assert [r[0] for r in runs] == ['off3', 'on3']
assert not (parent / 'off3/receipt.rds').exists()
assert not (parent / 'on3/receipt.rds').exists()
for name in ('off1', 'on1', 'on2', 'off2'):
    report = read(parent / name / 'report.json')
    process = read(parent / (name + '-process.json'))
    assert report['gates'] and all(report['gates'].values())
    assert process['exit_code'] == 0 and process['isolated']
    assert sha(parent / name / 'receipt.rds') == report['receipt_sha256']
assert not subprocess.check_output([
    'nvidia-smi', '-i', '1', '--query-compute-apps=pid',
    '--format=csv,noheader'], text=True).strip(), 'GPU1 must be exclusive'
parent_files = {str(p.relative_to(parent)): sha(p)
                for p in sorted(parent.rglob('*')) if p.is_file()}
supplement_files = {str(p.relative_to(root)): sha(p)
                    for p in sorted(supplement_tool.iterdir()) if p.is_file()}
out.mkdir(parents=True)
for name in ('build.json', 'checks.json', 'stats.so'):
    shutil.copyfile(parent / name, out / name)
for name in ('campaign.py', 'run.R'):
    shutil.copyfile(supplement_tool / name, out / ('supplement-' + name))
evidence_class = ('GPU1 exclusive, shared-host supplemental development; not '
                  'isolated performance qualification, clean producer or promotion')
record = dict(
    schema='device-properties-cache-shared-host-protocol-v1',
    authorization='User: 那就继续测试, after GPU0 occupation and GPU1 availability were explained',
    evidence_class=evidence_class, parent_artifact=str(parent.relative_to(root)),
    original_protocol_sha256=sha(parent / 'protocol.json'),
    native_sha256=scope['build']['native_sha256'], stats_sha256=sha(out / 'stats.so'),
    sequence=[r[0] for r in runs], selected_gpu=1, shared_host_allowed=True,
    original_isolated_performance_gate_eligible=False,
    numerical_configuration_and_gates_unchanged=True,
    parent_files=parent_files, frozen_tools=protocol['frozen_tools'],
    frozen_dependencies=protocol['frozen_dependencies'],
    frozen_supplement_tools=supplement_files,
    setup_code_sha256=hashlib.sha256(prefix.encode()).hexdigest(),
    run_loop_sha256=hashlib.sha256(loop.encode()).hexdigest(),
    runner_change='Only evidence_class string; original computation and all hard gates unchanged')
with (out / 'protocol.json').open('x') as stream:
    json.dump(record, stream, indent=2); stream.write('\n')
protocol_sha = sha(out / 'protocol.json')
with (out / 'environment.log').open('x') as stream:
    subprocess.run(['nvidia-smi',
        '--query-gpu=index,uuid,name,driver_version,memory.total,power.limit,clocks.current.sm,temperature.gpu',
        '--format=csv'], stdout=stream, stderr=subprocess.STDOUT, check=True)
scope.update(out=out, tool=supplement_tool, shared_host=True, runs=runs,
             frozen_tools={**protocol['frozen_tools'], **supplement_files},
             frozen_dependencies=protocol['frozen_dependencies'])
exec(compile(loop, str(original), 'exec'), scope)

# Audit and report this pair alone. The isolated six-run gate remains pending.
assert sha(out / 'protocol.json') == protocol_sha
assert all(sha(parent / p) == h for p, h in parent_files.items())
assert all(sha(root / p) == h for p, h in scope['frozen_tools'].items())
assert all(sha(root / p) == h for p, h in protocol['frozen_dependencies'].items())
assert sha(scope['native']) == record['native_sha256']
assert sha(out / 'stats.so') == record['stats_sha256']
assert all(sha(root / p) == h for p, h in scope['build']['source_sha256'].items())
reports, processes, peaks, references = {}, {}, {}, {}
for name, mode, reference, repeated in runs:
    r = read(out / name / 'report.json')
    p = read(out / (name + '-process.json'))
    assert r['gates'] and all(r['gates'].values())
    assert r['evidence_class'] == evidence_class
    assert r['receipt_sha256'] == sha(out / name / 'receipt.rds')
    assert r['runner_sha256'] == sha(original_tool / 'run.R')
    assert r['native_sha256'] == p['native_sha256'] == record['native_sha256']
    assert r['source_sha256'] == scope['build']['source_sha256']
    assert p['exit_code'] == 0 and p['physical_gpu'] == 1
    assert p['shared_host_allowed'] and p['selected_gpu_isolated']
    assert not p['unexpected_gpu_pids']
    assert r['properties_mode'] == p['properties_cache'] == mode
    assert r['comparisons']['changed_ci_count'] == r['comparisons']['changed_pmax_count'] == 0
    assert max(r['oracle_errors']) <= 1e-10
    ref = root / reference
    assert sha(ref) == r['reference_sha256']
    references[str(ref.relative_to(root))] = sha(ref)
    repeat_ref = root / repeated
    references[str(repeat_ref.relative_to(root))] = sha(repeat_ref)
    for field, filename in [('logical_ci_trace', 'logical_ci_trace.rds'),
                            ('pmax', 'pmax.rds'), ('adjacency', 'adjacency.rds')]:
        path = root / 'fastkpc/artifacts/full_cuda_ci/oracle_351x48_default_inf_v2' / filename
        assert sha(path) == r['authority_sha256'][field]
        references[str(path.relative_to(root))] = sha(path)
    frozen = root / 'fastkpc/artifacts/full_cuda_ci/l2_prefill_20261022/qualification.rds'
    assert sha(frozen) == r['comparisons']['frozen_l2']['receipt_sha256']
    references[str(frozen.relative_to(root))] = sha(frozen)
    with (out / (name + '-gpu.csv')).open() as stream:
        peaks[name] = max(int(row[2]) for row in csv.reader(stream)
                          if len(row) == 7 and int(row[1]) == 1)
    reports[name], processes[name] = r, p
saving = reports['off3']['elapsed_sec'] - reports['on3']['elapsed_sec']
closure = dict(
    schema='device-properties-cache-shared-host-closure-v1',
    status='PASS_SHARED_HOST_SUPPLEMENT', hard_gates='PASS',
    evidence_class=evidence_class, native_sha256=record['native_sha256'],
    protocol_sha256=protocol_sha, original_evidence_unchanged=True,
    original_isolated_performance_gate='PENDING_OFF3_ON3',
    new_accepted_performance_median=False, full_canonical_runs=2,
    logical_ci_per_run=240498, ci_and_pmax_bits_exact=True,
    elapsed_sec_by_run={n: r['elapsed_sec'] for n, r in reports.items()},
    paired_saving_seconds=saving,
    paired_saving_fraction=saving / reports['off3']['elapsed_sec'],
    performance_interpretation='One shared-host pair; do not pool with isolated runs or claim performance GO',
    cache_stats_by_run={n: r['properties_cache'] for n, r in reports.items()},
    stages_by_run={n: r['stages'] for n, r in reports.items()},
    gpu1_sampled_total_peak_mib=peaks, processes=processes,
    authority_max_ci_error=max(r['oracle_errors'][0] for r in reports.values()),
    authority_max_pmax_error=max(r['oracle_errors'][1] for r in reports.values()),
    default_enabled=False, phase10='ACTIVE', candidate='NOT_FROZEN',
    holdout='SEALED', recommended=False)
with (out / 'closure.json').open('x') as stream:
    json.dump(closure, stream, indent=2); stream.write('\n')
manifest = dict(
    schema='device-properties-cache-shared-host-manifest-v1',
    native_sha256=record['native_sha256'], protocol_sha256=protocol_sha,
    sources=scope['build']['source_sha256'], tools=scope['frozen_tools'],
    dependencies=protocol['frozen_dependencies'], references=references,
    parent_artifact=record['parent_artifact'], parent_files=parent_files,
    files={str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
with (out / 'manifest.json').open('x') as stream:
    json.dump(manifest, stream, indent=2); stream.write('\n')
print(json.dumps({k: v for k, v in closure.items()
                  if k not in ('stages_by_run', 'processes')}, indent=2), flush=True)
print('manifest SHA256', sha(out / 'manifest.json'), flush=True)
