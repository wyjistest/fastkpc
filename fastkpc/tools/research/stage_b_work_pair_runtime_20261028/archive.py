"""Close qualified default-off integration, preserving source-closure provenance."""
import sys
from support import *

out = Path(sys.argv[1]).resolve()
deep = Path(sys.argv[2]).resolve()
b = read(out/'build.json')
p = read(out/'protocol.json')
c = read(out/'campaign.json')
q = read(out/'checks.json')
d = read(deep/'closure.json')
assert c['status'] == 'PASS_COMPLETE_FULL_METHOD_CAMPAIGN'
assert c['decision'] == 'GO_DEFAULT_OFF_WORK_PAIR_RUNTIME'
assert all(c['performance_gates'].values())
assert q['status'] == 'PASS_PRODUCTION_MODE_LIFETIME_AND_GPU_PARITY'
assert d['status'] == 'PASS_DEEP_WORK_PAIR_GPU_BITWISE'
check_map(read(deep/'manifest.json')['files'], deep)
check_map(b['source_sha256'])
check_map(b['source_sha256'], out/'build-source')
check_map(b['objects'], out/'build-source/fastkpc/build')
check_map(p['tools'])
check_map(p['dependencies'])
assert sha(NATIVE) == b['native_sha256']
assert sha(out/'build-source/fastkpc/build/fastkpc_cuda.so') == sha(NATIVE)
assert not processes(), 'do not close with GPU work still active'
reports = []
for r in c['runs']:
    report = read(out/r['name']/'report.json')
    assert all(report['gates'].values()) and report['native_sha256'] == b['native_sha256']
    assert sha(out/r['name']/'report.json') == r['report_sha256']
    assert sha(out/r['name']/'receipt.rds') == r['receipt_sha256']
    proc = read(out/(r['name']+'-process.json'))
    assert proc['isolated'] and proc['exit_code'] == 0
    reports.append(report)
checks_process = read(out/'checks-process.json')
assert checks_process['isolated'] and checks_process['exit_code'] == 0
write(out/'closure.json', dict(status='PASS_WORK_PAIR_RUNTIME_SOURCE_CLOSURE', decision=c['decision'],
    native_sha256=sha(NATIVE), production_inputs=len(b['source_sha256']),
    build_sha256=sha(out/'build.json'), protocol_sha256=sha(out/'protocol.json'),
    checks_sha256=sha(out/'checks.json'), campaign_sha256=sha(out/'campaign.json'),
    deep_closure_sha256=sha(deep/'closure.json'), deep_manifest_sha256=sha(deep/'manifest.json'),
    prior_isolated_closure_sha256=sha(SEALED/'closure.json'),
    captured_intermediate_targets_total=130670, captured_intermediate_targets_new=25873,
    captured_rotations_total=10515312078+d['qualification']['captured']['rotations_compared'],
    full_canonical_runs=6, logical_ci_per_run=240498,
    all_gates=True, maximum_authority_error=max(max(r['oracle_errors']) for r in reports),
    medians=c['medians'], level_medians=c['level_medians'], outer_fraction_reduction=c['outer_fraction_reduction'],
    accepted_opt_in_development_median=c['medians']['1']['elapsed_sec'], default_enabled=False,
    phase10='ACTIVE', candidate='NOT_FROZEN', holdout='SEALED', recommended=False,
    l8_optimizer='STOP_UNCHANGED', actual_sm_occupancy='NOT_MEASURED',
    evidence_class='isolated same-native incremental source-closure development; not clean producer or promotion',
    capacity_coverage='existing raw checkpoints and external sampled memory, not a continuous peak proof'))
write(out/'manifest.json', dict(schema='work-pair-runtime-manifest-v1',
    native_sha256=sha(NATIVE), source_sha256=b['source_sha256'], dependencies=p['dependencies'], tools=p['tools'],
    files={str(x.relative_to(out)): sha(x) for x in sorted(out.rglob('*')) if x.is_file()}))
print(json.dumps(dict(decision=c['decision'], closure_sha256=sha(out/'closure.json'),
                     manifest_sha256=sha(out/'manifest.json')), indent=2))
