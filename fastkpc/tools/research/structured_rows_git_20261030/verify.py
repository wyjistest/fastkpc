"""Read-only verification of the measured source closure and its Git commits."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = ROOT / 'fastkpc/artifacts/full_cuda_ci'
ROWS = ARTIFACTS / 'structured_rows_20261029'
PARENT = ARTIFACTS / 'build_consistency_20261029'
SHARED = ARTIFACTS / 'structured_rows_shared_host_20261029'
TOOL_NAMES = (
    'build_consistency_20261029', 'structured_rows_20261029',
    'structured_rows_shared_host_20261029', 'structured_rows_completion_20261030',
)
TESTS = (
    'fastkpc/tests/test_full_cuda_ci_method_deferred_svd.R',
    'fastkpc/tests/test_full_cuda_ci_method_permutation_overlap.R',
)
PINNED = {
    ROWS / 'closure.json': 'cdc26f74cab09449dcfd47ef2d3f6b710246dd4682dc93122fca5855081bfaa3',
    ROWS / 'manifest.json': '196a0a84da7535a65927dbc4d8a49db5b192950fc7372848a5e970d1e4a8c87c',
    PARENT / 'closure.json': '4358903b70b35a529716b746a77cd755e598d5a2454084e081578b85e4634953',
    SHARED / 'manifest.json': 'ad13ec1733094e815d7b92c2121f367a4d762367b4198cf1827c29c6efeacd9d',
}


def require(condition, detail):
    if not condition:
        raise RuntimeError(detail)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def check_map(base, mapping):
    for name, expected in mapping.items():
        require(sha(base / name) == expected, 'hash mismatch: ' + str(base / name))
    return len(mapping)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def check_git(commit, mapping):
    for name, expected in mapping.items():
        require(hashlib.sha256(git('show', commit + ':' + name)).hexdigest() == expected,
                'Git blob mismatch: ' + commit + ':' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-commit')
    parser.add_argument('--runtime-commit')
    args = parser.parse_args()
    require(bool(args.build_commit) == bool(args.runtime_commit),
            'supply both commit arguments or neither')
    for path, expected in PINNED.items():
        require(sha(path) == expected, 'sealed identity changed: ' + str(path))

    manifest = read(ROWS / 'manifest.json')
    closure = read(ROWS / 'closure.json')
    build = read(ROWS / 'candidate/build.json')
    parent = read(PARENT / 'full/build.json')
    install = read(ROWS / 'installation.json')
    require(closure['status'] == 'GO_CAPACITY' and closure['speed'] == 'NOT_PROVEN',
            'unexpected acceptance result')
    counts = {
        'production_source_sha256': check_map(ROOT, manifest['production_source_sha256']),
        'tools': check_map(ROOT, manifest['tools']),
        'files': check_map(ROWS, manifest['files']),
        'parent_files': check_map(PARENT, manifest['parent_files']),
    }
    source = build['source_sha256']
    require(source == manifest['production_source_sha256'], 'source map differs')
    check_map(ROWS / 'candidate/source', source)
    check_map(PARENT / 'source', parent['source_sha256'])
    require(set(source) == set(parent['source_sha256']), 'parent source set differs')
    require([p for p in source if source[p] != parent['source_sha256'][p]] ==
            ['fastkpc/src/cuda/full_cuda_ci_vertical.cu'], 'unexpected runtime changes')

    live = ROOT / 'fastkpc/build/fastkpc_cuda.so'
    original = PARENT / 'parent/fastkpc_cuda.so'
    require(sha(live) == build['native_sha256'] == closure['native_sha256'] ==
            install['native_sha256'], 'installed native differs')
    check_map(ROOT / 'fastkpc/build', build['objects'])
    require(set(build['objects']) == set(parent['objects']), 'linked object set differs')
    require([n for n in build['objects'] if build['objects'][n] != parent['objects'][n]] ==
            ['full_cuda_ci_vertical.o'], 'unexpected rebuilt objects')
    generated = {name: sha(ROWS / 'candidate' / name) for name in
                 ('recorded_dbdsqr.inc', 'recorded_dbdsqr_fixed_vt.inc')}
    check_map(ROOT / 'fastkpc/build', generated)

    historical = read(SHARED / 'manifest.json')
    history_count = check_map(SHARED, historical['files'])
    mapping = []
    for name, expected in historical['dependencies'].items():
        path = Path(name)
        if path == live:
            require(expected == install['previous_native_sha256'], 'old native identity differs')
            mapping.append(dict(original_path=str(live.relative_to(ROOT)),
                                archived_path=str(original.relative_to(ROOT)), sha256=expected))
            path = original
        require(sha(path) == expected, 'historical dependency differs: ' + name)
        history_count += 1
    for variant, sources in historical['source_sha256'].items():
        require(variant in ('off', 'on'), 'unknown historical source variant')
        snapshot = PARENT / 'source' if variant == 'off' else ROWS / 'candidate/source'
        history_count += check_map(snapshot, sources)
    require(len(mapping) == 1 and history_count == 419, 'historical verification scope differs')

    tools = dict(manifest['tools'])
    for name, expected in historical['dependencies'].items():
        path = Path(name)
        if path.parent == ROOT / 'fastkpc/tools/research/structured_rows_shared_host_20261029':
            tools[str(path.relative_to(ROOT))] = expected
    for name, expected in read(ROWS / 'completion_20261030/attribution.json')['tool_sha256'].items():
        tools['fastkpc/tools/research/structured_rows_completion_20261030/' + name] = expected
    current_tools = {str(p.relative_to(ROOT)) for folder in TOOL_NAMES
                     for p in (ROOT / 'fastkpc/tools/research' / folder).iterdir() if p.is_file()}
    require(set(tools) == current_tools and len(tools) == 25, 'qualification tool set differs')
    check_map(ROOT, tools)
    regression_deps = read(PARENT / 'full/regressions2/protocol.json')['dependencies']
    tests = {name: regression_deps[str(ROOT / name)] for name in TESTS}
    check_map(ROOT, tests)
    verifier = {str(p.relative_to(ROOT)): sha(p) for p in Path(__file__).parent.iterdir() if p.is_file()}

    python_paths = [ROOT / p for p in {**tools, **verifier} if p.endswith('.py')]
    for path in python_paths:
        ast.parse(path.read_text(), filename=str(path))
    r_paths = [str(ROOT / p) for p in {**tools, **tests} if p.endswith('.R')]
    subprocess.run(['Rscript', '--vanilla', '-e',
                    'invisible(lapply(commandArgs(TRUE), function(p) parse(file=p)))',
                    *r_paths], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(['bash', '-n', 'fastkpc/tools/build_cuda_native.sh'], cwd=ROOT, check=True)

    result = dict(
        schema='phase10-source-closure-git-attestation-v1', date='2026-10-30',
        status='PASS_WORKTREE_SOURCE_CLOSURE', original_campaign_head=closure['head'],
        evidence_class=closure['evidence_class'],
        source_count=len(source), source_sha256=source,
        qualification_file_count=len(tools), qualification_sha256=tools,
        regression_sha256=tests, git_verification_tool_sha256=verifier,
        verified_manifest_entries=counts, historical_verified_entries=history_count,
        historical_native_mapping=mapping, native_sha256=build['native_sha256'],
        parent_native_sha256=parent['native_sha256'], linked_object_sha256=build['objects'],
        generated_include_sha256=generated,
        archive_sha256={str(p.relative_to(ROOT)): expected for p, expected in PINNED.items()},
        correctness=closure['numerical_behavior'], capacity=closure['capacity'],
        outer_nonregression=closure['outer_nonregression'], speed=closure['speed'],
        medians=closure['medians'], pool_bytes_saved=closure['pool_bytes_saved'],
        checks=dict(python_ast=len(python_paths), r_parse=len(r_paths), shell_syntax=1),
        clean_checkout_campaign=False, new_gpu_runs=False, native_rebuilt=False,
        phase10='ACTIVE', candidate='NOT_FROZEN', holdout='SEALED', recommended=False,
    )
    if args.runtime_commit:
        build_commit = git('rev-parse', '--verify', args.build_commit + '^{commit}').decode().strip()
        runtime_commit = git('rev-parse', '--verify', args.runtime_commit + '^{commit}').decode().strip()
        require(git('rev-parse', runtime_commit + '^').decode().strip() == build_commit,
                'runtime commit must directly follow build commit')
        check_git(build_commit, parent['source_sha256'])
        check_git(build_commit, tests)
        check_git(runtime_commit, {**source, **tools, **tests, **verifier})
        result.update(status='PASS_COMMITTED_SOURCE_CLOSURE', build_fix_commit=build_commit,
                      runtime_commit=runtime_commit, committed_source_hashes_match=True,
                      committed_qualification_hashes_match=True)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
