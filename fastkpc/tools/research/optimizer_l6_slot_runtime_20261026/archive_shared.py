"""Close only the shared-host hard gates; never accept a performance median."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[4]
out = Path(sys.argv[1]).resolve()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads((out/p).read_text())
build, checks, protocol = (read(p) for p in
    ['build.json', 'checks.json', 'shared-qualification-protocol.json'])
assert build['status'] == checks['status'] == 'PASS'
assert protocol['performance_eligible'] is False
assert protocol['sequence'] == ['shared-off', 'shared-on']
assert sha(root/'fastkpc/build/fastkpc_cuda.so') == build['native_sha256'] == checks['native_sha256']
for mapping in [build['source_sha256'], protocol['frozen_tools'], protocol['frozen_dependencies']]:
    assert all(sha(root/p) == h for p, h in mapping.items())
reports, references = {}, {}
for name in protocol['sequence']:
    r, p = read(name+'/report.json'), read(name+'-process.json')
    assert p['exit_code'] == 0 and p['physical_gpu'] == 1 and p['shared_host_allowed']
    assert p['selected_gpu_isolated'] and not p['unexpected_gpu_pids']
    assert r['gates'] and all(r['gates'].values())
    assert 'shared-host' in r['evidence_class']
    assert r['native_sha256'] == p['native_sha256'] == build['native_sha256']
    assert r['source_sha256'] == build['source_sha256']
    assert sha(out/name/'receipt.rds') == r['receipt_sha256']
    assert r['runner_sha256'] == sha(Path(__file__).with_name('run.R'))
    ref = root/p['command'][-2]
    assert r['reference_sha256'] == sha(ref)
    references[str(ref.relative_to(root))] = sha(ref)
    assert p['l4_slots'] == p['l5_slots'] == p['properties_cache'] == '1'
    assert r['l6_slots'] == p['l6_slots'] == ('1' if name == 'shared-on' else '0')
    assert r['comparisons']['changed_ci_count'] == r['comparisons']['changed_pmax_count'] == 0
    assert max(r['oracle_errors']) <= 1e-10
    for field, filename in [('logical_ci_trace','logical_ci_trace.rds'),
                            ('pmax','pmax.rds'),('adjacency','adjacency.rds')]:
        path = root/'fastkpc/artifacts/full_cuda_ci/oracle_351x48_default_inf_v2'/filename
        assert sha(path) == r['authority_sha256'][field]
        references[str(path.relative_to(root))] = sha(path)
    reports[name] = r
pool = reports['shared-on']['stages']['l6_pool']
closure = dict(status='PASS_L6_RUNTIME_SHARED_HOST_HARD_GATES', performance_status='PENDING_ISOLATED_CAMPAIGN',
    native_sha256=build['native_sha256'], logical_ci_per_run=240498, full_canonical_runs=2,
    ci_and_pmax_bits_exact=True, authority_max_error=max(max(r['oracle_errors']) for r in reports.values()),
    minimum_l6_pool_free_bytes=pool['minimum_free_bytes'],
    maximum_l6_pool_increment_bytes=pool['maximum_pool_increment_bytes'],
    observed_outer_sec={k:r['elapsed_sec'] for k,r in reports.items()},
    timing_use='diagnostic only; not pooled with isolated timings; no accepted median',
    build_sha256=sha(out/'build.json'), checks_sha256=sha(out/'checks.json'),
    protocol_sha256=sha(out/'shared-qualification-protocol.json'),
    source_sha256=build['source_sha256'], tools=protocol['frozen_tools'], references=references,
    phase10='ACTIVE', candidate='NOT_FROZEN', holdout='SEALED', recommended=False)
closure['files'] = {str(p.relative_to(out)):sha(p) for p in sorted(out.glob('shared-*'))
                    if p.is_file()}
for name in protocol['sequence']:
    closure['files'].update({str(p.relative_to(out)):sha(p) for p in sorted((out/name).rglob('*')) if p.is_file()})
with (out/'shared-qualification-closure.json').open('x') as stream:
    json.dump(closure, stream, indent=2); stream.write('\n')
print(json.dumps({k:v for k,v in closure.items() if k not in ['source_sha256','tools','references','files']}, indent=2))
print('closure SHA256', sha(out/'shared-qualification-closure.json'))
