"""Close actual replacement evidence without changing production status."""
import csv
from pathlib import Path
import sys
TOOL=Path(__file__).resolve().parent;ROOT=TOOL.parents[3]
sys.path.insert(0,str(TOOL.parent/'optimizer_l3_slots_20261027'))
from common import NATIVE,load,sha,write_new,source_identity
out=Path(sys.argv[1]).resolve();source_identity();b=load(out/'build.json')
assert not (out/'closure.json').exists()
rows={v:load(out/(v+'-closure.json')) for v in 'WS'}
for v,r in rows.items():
    assert r['status']=='PASS' and all(r['canonical_hard_gates'].values())
    assert load(out/(v+'-process.json'))['isolated']
    p=r['probe'];assert p['optimizer_setups']==4065 and p['optimizer_targets']==p['stage_a_targets']==54878
    assert p['minimum_free_bytes']>=2*1024**3
    assert r['diagnostic_overlay_sha256']==sha(out/'probe.so')
assert rows['W']['stage_a_reference_sha256']==rows['S']['stage_a_reference_sha256']==sha(out/'W/stage-a.bin')
s=rows['S']['probe'];assert s['pool_increment_bytes']<=1536*1024**2 and s['pool_resources_closed'] and s['worker_resource_mutations']==0
assert s['stage_a_batches']==rows['W']['probe']['stage_a_batches']
for p,h in {**b['tools'],**b['dependencies'],**b['source_sha256']}.items():assert sha(ROOT/p)==h,p
for p,h in b['outputs'].items():assert sha(out/p)==h,p
write_new(out/'closure.json',dict(schema='l3-real-replacement-coexistence-closure-v1',
    status='PASS_L3_REPLACEMENT_COEXISTENCE',native_sha256=sha(NATIVE),diagnostic_overlay_sha256=sha(out/'probe.so'),
    protocol_sha256=sha(out/'protocol.json'),build_sha256=sha(out/'build.json'),variants=rows,
    all_optimizer_outputs_bitwise_exact=True,all_stage_a_outputs_bitwise_exact=True,
    original_stage_a_cohort_order_exact=True,original_window_prepared_retained=False,
    diagnostic_only=True,production_sources_changed=False,production_native_changed=False,
    performance_evidence=False,accepted_canonical_median_sec=368.851,
    runtime_integration='NOT_YET_GO',phase10='ACTIVE',candidate='NOT_FROZEN',holdout='SEALED',recommended=False))
write_new(out/'manifest.json',dict(schema='l3-replacement-coexistence-manifest-v1',
    files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()},
    tools={str(p.relative_to(ROOT)):sha(p) for p in sorted(TOOL.iterdir()) if p.is_file()}))
print('PASS_L3_REPLACEMENT_COEXISTENCE',s,flush=True)
