"""Build a deep-only observer and qualifier; retain the unchanged production native."""
import subprocess
import sys
from support import *

out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=False)
b = original_identity()
activity = ROOT / 'fastkpc/tools/research/stage_b_attribution_20261028'
wp = ROOT / 'fastkpc/tools/research/stage_b_work_pair_20261028'
runner = ROOT / 'fastkpc/tools/research/optimizer_l3_slot_runtime_20261028/run.R'
text = runner.read_text()
token = 'Sys.setenv(FASTKPC_MULTI_PENALTY_STATUS_WAIT_BEFORE_D2H="1",'
assert text.count(token) == 1
text = text.replace(token, 'evidence_class <- "deep component capture; diagnostic D2H; not performance evidence"\n' + token)
text = text.replace('optimizer-l3-slot-runtime-canonical-v1', 'work-pair-deep-capture-v1')
text = text.replace('sha("fastkpc/tools/research/optimizer_l3_slot_runtime_20261028/run.R")',
                    'sha(' + json.dumps(str(out / 'capture_runner.R')) + ')')
(out / 'capture_runner.R').write_text(text)
raw = subprocess.check_output(['nm', '-D', '--defined-only', str(NATIVE)], text=True)
symbols = [r.split()[-1] for r in raw.splitlines()]
names = subprocess.check_output(['c++filt'], input='\n'.join(symbols), text=True).splitlines()
with (out / 'symbols.inc').open('x') as f:
    for short, term in [('begin', 'begin_full_cuda_ci_level_component_prefill'),
                        ('store', 'store_full_cuda_ci_level_component_residual'),
                        ('build', 'build_full_cuda_ci_level_components'),
                        ('info', 'full_cuda_ci_legacy_eig_component_cache_info')]:
        found = [s for s, n in zip(symbols, names) if 'fastkpc::' + term + '(' in n]
        assert len(found) == 1
        f.write('constexpr const char* SYM_' + short + ' = "' + found[0] + '";\n')
commands = [
    ['g++', '-O2', '-std=c++17', '-pthread', '-fPIC', '-shared', '-I'+str(ROOT/'fastkpc/src'),
     '-I'+str(activity), '-I'+str(out), '-I/usr/local/cuda/include', str(TOOL/'observe.cpp'),
     '-L/usr/local/cuda/lib64', '-Wl,-rpath,/usr/local/cuda/lib64', '-lcudart', '-lcrypto', '-ldl', '-o', str(out/'observe.so')],
    ['/usr/local/cuda/bin/nvcc', '-O3', '-arch=sm_89', '-std=c++17', '--cudart', 'shared',
     '-I'+str(wp), str(TOOL/'qualify.cu'), '-ldl', '-o', str(out/'qualify')]]
for i, command in enumerate(commands):
    with (out / f'build-{i}.log').open('x') as f:
        subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True)
sealed = read(SEALED / 'manifest.json')
deps = [SEALED / name for name in ['baseline_diag.so', 'candidate_diag.so']]
for p in deps:
    assert sha(p) == sealed['files'][p.name]
deps += [SEALED/'manifest.json', SEALED/'closure.json', SEALED/'qualification.json',
         activity/'common.hpp', wp/'diagnostic.hpp', runner, PARENT/'build.json', PARENT/'on3/receipt.rds',
         ROOT/'fastkpc/tools/with_phase10_exact_development_env.sh',
         ROOT/'fastkpc/tools/research/exact_speedup_20261020/pinned_optimizer_gates.R',
         ROOT/'fastkpc/tools/run_phase10_pvalue_contract_audit.R']
original_identity()
write(out/'build.json', dict(status='PASS', native_sha256=b['native_sha256'], source_sha256=b['source_sha256'],
    commands=commands, production_kernels_recompiled=False,
    tools={str(p.relative_to(ROOT)): sha(p) for p in sorted(TOOL.iterdir()) if p.is_file()},
    dependencies={str(p): sha(p) for p in deps},
    outputs={p: sha(out/p) for p in ['capture_runner.R', 'symbols.inc', 'observe.so', 'qualify']}))
print('PASS deep diagnostic build; production unchanged', flush=True)
