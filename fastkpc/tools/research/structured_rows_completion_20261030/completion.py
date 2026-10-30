"""Descriptive stage deltas and explicit historical-native identity verification."""
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys

TOOL = Path(__file__).resolve().parent
ROOT = TOOL.parents[3]
BASE = ROOT/'fastkpc/artifacts/full_cuda_ci/structured_rows_20261029'
PARENT = ROOT/'fastkpc/artifacts/full_cuda_ci/build_consistency_20261029'
SHARED = ROOT/'fastkpc/artifacts/full_cuda_ci/structured_rows_shared_host_20261029'
OUT = BASE/'completion_20261030'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def analyze():
    outer = read(BASE/'canonical/closure.json')
    protocol = read(BASE/'canonical/protocol.json')
    pending = read(BASE/'pending-outer-protocol.json')
    assert outer['gate'] == protocol['gate'] == pending['gate']
    assert protocol['sequence'] == pending['sequence']
    assert sha(ROOT/'fastkpc/tools/research/structured_rows_20261029/campaign.py') == pending['campaign_tool_sha256']
    reports = {}
    for variant, cycle in protocol['sequence']:
        folder = BASE/'canonical'/variant/(variant+str(cycle))
        report = read(folder/'report.json')
        assert all(report['gates'].values())
        assert sha(folder/'receipt.rds') == report['receipt_sha256']
        process = read(BASE/'canonical'/(variant+str(cycle)+'-process.json'))
        assert process['isolated'] and not process['unexpected'] and process['exit_code'] == 0
        reports[variant,cycle] = report
    medians = {v:statistics.median(reports[v,i]['elapsed_sec'] for i in [1,2,3]) for v in ['off','on']}
    assert medians == outer['medians']
    rows = []
    for cycle in [1,2,3]:
        phases = {}
        for variant in ['off','on']:
            report = reports[variant,cycle]
            levels = {row['level']:row['host_sec'] for row in report['stages']['optimizer_by_level']}
            phases[variant] = dict(outer=report['elapsed_sec'], l7_optimizer=levels[7],
                l8_optimizer=levels[8], l4_optimizer=levels[4], l5_optimizer=levels[5],
                stage_b=report['stages']['stage_b_sec'])
        delta = {key:phases['on'][key]-phases['off'][key] for key in phases['off']}
        late = delta['l7_optimizer']+delta['l8_optimizer']
        middle = delta['l4_optimizer']+delta['l5_optimizer']
        remainder = delta['outer']-late-middle-delta['stage_b']
        rows.append(dict(cycle=cycle,phases=phases,delta_on_minus_off=delta,
                         l78_delta=late,l45_delta=middle,remainder=remainder))
    OUT.mkdir()
    saved = OUT/'tools'
    saved.mkdir()
    for path in TOOL.iterdir():
        if path.is_file():
            shutil.copyfile(path,saved/path.name)
    write(OUT/'attribution.json',dict(status='PASS_DESCRIPTIVE_ACCOUNTING',pairs=rows,
        outer_status=outer['status'],outer_medians=medians,gate=outer['gate'],speed=outer['speed'],
        scope='paired host-time differences; no causal claim or adjusted acceptance metric',
        original_outer_used_for_acceptance=True,all_completed_runs_retained=True,
        canonical_closure_sha256=sha(BASE/'canonical/closure.json'),
        original_pending_protocol_sha256=sha(BASE/'pending-outer-protocol.json'),
        tool_sha256={path.name:sha(path) for path in saved.iterdir()}))
    print('PASS paired descriptive accounting',json.dumps(rows),flush=True)


def verify_history():
    assert OUT.is_dir()
    installation = read(BASE/'installation.json')
    live = ROOT/'fastkpc/build/fastkpc_cuda.so'
    archived = PARENT/'parent/fastkpc_cuda.so'
    original = read(PARENT/'parent/identity.json')
    assert installation['status'] == 'PASS_INSTALLED'
    assert sha(live) == installation['native_sha256'] == read(BASE/'candidate/build.json')['native_sha256']
    assert sha(archived) == installation['previous_native_sha256'] == original['native_sha256']
    assert sha(SHARED/'manifest.json') == 'ad13ec1733094e815d7b92c2121f367a4d762367b4198cf1827c29c6efeacd9d'
    manifest = read(SHARED/'manifest.json')
    checked = 0
    mapping = []
    for name, expected in manifest['files'].items():
        assert sha(SHARED/name) == expected,name
        checked += 1
    for name, expected in manifest['dependencies'].items():
        path = Path(name)
        if path == live:
            assert expected == original['native_sha256'] and sha(archived) == expected
            mapping.append(dict(original_path=name,archived_path=str(archived),sha256=expected))
        else:
            assert sha(path) == expected,name
        checked += 1
    for variant, sources in manifest['source_sha256'].items():
        snapshot = PARENT/'source' if variant == 'off' else BASE/'candidate/source'
        for name, expected in sources.items():
            assert sha(snapshot/name) == expected,(variant,name)
            if variant == 'on':
                assert sha(ROOT/name) == expected,name
            checked += 1
    assert len(mapping) == 1 and checked == 419
    write(OUT/'historical-verification.json',dict(status='PASS_PRESERVED_HISTORICAL_IDENTITY',
        checked_manifest_entries=checked,explicit_archived_native_mapping=mapping,
        current_installed_native_sha256=sha(live),installation_sha256=sha(BASE/'installation.json'),
        shared_manifest_sha256=sha(SHARED/'manifest.json'),
        original_shared_receipts_and_manifest_unchanged=True,
        pending_protocol_sha256=sha(BASE/'pending-outer-protocol.json'),
        pending_closure_sha256=sha(BASE/'pending-closure.json'),
        pending_manifest_sha256=sha(BASE/'pending-manifest.json')))
    print('PASS historical identity after authorized installation;',checked,'entries',flush=True)


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ['analyze','verify-history']
    if sys.argv[1] == 'analyze':
        analyze()
    else:
        verify_history()
