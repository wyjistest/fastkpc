"""Same-native isolated full-method OFF/ON campaign; all other options fixed."""
import statistics
import sys
from support import *

out = Path(sys.argv[1]).resolve()
b = read(out/'build.json')
assert read(out/'checks.json')['status'] == 'PASS_PRODUCTION_MODE_LIFETIME_AND_GPU_PARITY'
tools = {str(p.relative_to(ROOT)): sha(p) for p in sorted(TOOL.iterdir()) if p.is_file()}
deps = [PARENT/'on3/receipt.rds', ROOT/'fastkpc/tools/with_phase10_exact_development_env.sh',
    ROOT/'fastkpc/tools/research/exact_speedup_20261020/pinned_optimizer_gates.R',
    ROOT/'fastkpc/tools/run_phase10_pvalue_contract_audit.R', ROOT/'fastkpc/R/fast_kpc.R',
    ROOT/'fastkpc/R/full_cuda_ci_phase35_vertical.R',
    ROOT/'fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds']
deps += [ROOT/'fastkpc/artifacts/full_cuda_ci/oracle_351x48_default_inf_v2'/name
         for name in ['logical_ci_trace.rds','pmax.rds','adjacency.rds']]
dependencies = {str(p): sha(p) for p in deps}
def frozen():
    assert sha(NATIVE) == b['native_sha256']
    check_map(b['source_sha256'])
    check_map(tools)
    check_map(dependencies)

env = environment()
frozen()
assert not processes(), 'external compute; no campaign started'
sequence = [('off1', '0'), ('on1', '1'), ('on2', '1'), ('off2', '0'), ('off3', '0'), ('on3', '1')]
write(out/'protocol.json', dict(schema='work-pair-full-method-protocol-v1', declared_epoch=time.time(),
    native_sha256=sha(NATIVE), build_sha256=sha(out/'build.json'), checks_sha256=sha(out/'checks.json'),
    sequence=[name for name, _ in sequence], physical_gpu=1, other_gpu_idle=True,
    affinity=sorted(os.sched_getaffinity(0)), outer_median_ratio_limit=.98,
    every_on_faster_than_every_off=True, stage_b_median_ratio_limit=.95,
    capacity_contract='unchanged L3 split raw and L4/L6 guards; no subtraction or warmup',
    diagnostic_interposition=False, extra_payload_d2h=False, tools=tools, dependencies=dependencies,
    evidence_class='isolated incremental source-closure development, not clean producer or promotion'))
with (out/'environment.log').open('x') as f:
    subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,memory.total,power.limit,clocks.sm,temperature.gpu',
                    '--format=csv'], stdout=f, check=True)
runs = []
for name, mode in sequence:
    assert not (out/name).exists(), 'preserve earlier attempt'
    reference = PARENT/'on3/receipt.rds' if name == 'off1' else out/'off1/receipt.rds'
    repeated = '-' if name in ('off1','on1') else str(out/('on1' if mode=='1' else 'off1')/'receipt.rds')
    env['FASTKPC_DCC_GAMMA_STRUCTURED_WORK_PAIR'] = mode
    print('START', name, time.strftime('%H:%M:%S'), flush=True)
    job(out, name, ['sh', 'fastkpc/tools/with_phase10_exact_development_env.sh', 'Rscript', '--vanilla',
        str(TOOL/'run.R'), str(out/name), mode, str(reference), repeated], env, frozen)
    r = read(out/name/'report.json')
    assert all(r['gates'].values())
    runs.append(dict(name=name, mode=mode, elapsed_sec=r['elapsed_sec'],
        stage_b_sec=r['stages']['stage_b_sec'],
        stage_b_levels={str(row['level']): row['stage_b_host_ms']/1000 for row in r['level_component_prefill']},
        oracle_errors=r['oracle_errors'], report_sha256=sha(out/name/'report.json'),
        receipt_sha256=sha(out/name/'receipt.rds')))
    print('RESULT', name, r['elapsed_sec'], 'Stage B', r['stages']['stage_b_sec'], flush=True)
medians = {mode: {key: statistics.median(r[key] for r in runs if r['mode']==mode)
    for key in ['elapsed_sec','stage_b_sec']} for mode in ['0','1']}
level_medians = {mode: {str(level): statistics.median(r['stage_b_levels'][str(level)] for r in runs if r['mode']==mode)
    for level in range(2,9)} for mode in ['0','1']}
gates = dict(outer_median=medians['1']['elapsed_sec'] <= .98*medians['0']['elapsed_sec'],
    separation=max(r['elapsed_sec'] for r in runs if r['mode']=='1') < min(r['elapsed_sec'] for r in runs if r['mode']=='0'),
    stage_b=medians['1']['stage_b_sec'] <= .95*medians['0']['stage_b_sec'])
write(out/'campaign.json', dict(status='PASS_COMPLETE_FULL_METHOD_CAMPAIGN',
    decision='GO_DEFAULT_OFF_WORK_PAIR_RUNTIME' if all(gates.values()) else 'STOP_FULL_METHOD_BENEFIT',
    runs=runs, medians=medians, level_medians=level_medians, performance_gates=gates,
    outer_fraction_reduction=1-medians['1']['elapsed_sec']/medians['0']['elapsed_sec'],
    stage_b_fraction_reduction=1-medians['1']['stage_b_sec']/medians['0']['stage_b_sec']))
print('COMPLETE', medians, gates, flush=True)
