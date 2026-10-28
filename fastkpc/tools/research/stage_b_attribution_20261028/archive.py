"""Close diagnostic evidence, including failures; no source/native/default mutation."""
import json
from pathlib import Path
import shutil
import sys
from build import ROOT,TOOL,PARENT,NATIVE,sha,write,identity

base=Path(sys.argv[1]).resolve();out=base/'attempt3'
b=json.loads((out/'build.json').read_text());identity()
assert all(sha(out/'execution-source'/Path(p).name)==h for p,h in b['tools'].items())
post_campaign_corrections={'README.md','analyze.py'}
assert all(sha(ROOT/p)==h for p,h in b['tools'].items() if Path(p).name not in post_campaign_corrections)
assert all(sha(out/p)==h for p,h in b['outputs'].items())
campaign=json.loads((out/'campaign.json').read_text())
assert campaign['status']=='PASS_AUTHENTICATED_ACTIVITY_DIAGNOSTIC'
analysis=json.loads((out/'analysis.json').read_text())
assert analysis['analysis_tool_sha256']==sha(TOOL/'analyze.py')
assert all(sha(out/p)==h for p,h in analysis['input_sha256'].items())
for name in ['capture','replay1','replay2','replay3','trace']:
    p=json.loads((out/(name+'-process.json')).read_text())
    assert p['exit_code']==0 and p['isolated'] and not p['unexpected']
    for level,count in [(2,49919),(3,54878)]:
        r=json.loads((out/name/f'l{level}-observation.json').read_text())
        assert r['components']==count and r['three_field_hashes']==count
        assert r['dense_fallback']==0 and r['contexts']==10 and r['width']==64
        assert sha(out/name/f'l{level}-payload-sha256.txt')==sha(out/'capture'/f'l{level}-payload-sha256.txt')
canonical=json.loads((out/'capture/report.json').read_text())
assert all(canonical['gates'].values()) and canonical['comparisons']['changed_ci_count']==0
for level,count in [(2,49919),(3,54878)]:
    assert (out/'capture'/f'l{level}-residuals.f64').stat().st_size==count*351*8
    rows=[s.split() for s in (out/'capture'/f'l{level}-targets.tsv').read_text().splitlines()]
    assert [int(r[1]) for r in rows]==list(range(count)) and len(set(r[2] for r in rows))==count
    assert all(int(r[3])==0 for r in rows)
    assert len(set(r[0] for r in rows))==(3356 if level==2 else 5573)
for p,h in json.loads((out/'capture-identity.json').read_text())['hashes'].items():assert sha(out/p)==h
# Preserve source text for every historical build, not just its binary/hash.
for evidence,snapshot in [(base,base/'failed-tools'),(base/'attempt2',base/'attempt2/tool-source')]:
    old=json.loads((evidence/'build.json').read_text())
    assert all(sha(snapshot/Path(p).name)==h for p,h in old['tools'].items())
snapshot=out/'tool-source';snapshot.mkdir()
for p in TOOL.iterdir():
    if p.is_file():shutil.copyfile(p,snapshot/p.name)
shutil.copyfile(NATIVE,out/'accepted-native.so')
closure=dict(status='AUTHENTICATED_STAGE_B_ACTIVITY_DIAGNOSTIC_COMPLETE',
    hardware_counter_status='UNAVAILABLE_PRIVILEGE_NOT_MEASURED',
    native_sha256=sha(NATIVE),production_source_unchanged=True,production_kernels_recompiled=False,
    canonical_gate_count=len(canonical['gates']),canonical_ci_count=240498,
    component_count=104797,validated_replays=4,payload_hash_comparisons=419188,
    unprofiled_replay_times={str(d['level']):d['unprofiled_sec'] for d in analysis['levels']},
    unprofiled_replay_medians={str(d['level']):d['unprofiled_median_sec'] for d in analysis['levels']},
    trace_dropped=campaign['trace']['dropped'],trace_invalid_timestamps=campaign['trace']['invalid_timestamps'],
    failure_history=['initial capture: observer cache-info symbol unresolved after L2 hashes; exit127',
      'attempt2 activity smoke: injection library symbols local; fixed explicit handle lookup'],
    declaration_correction='L2 has ceil(49919/640)=78 waves, not 79; native/count/order gates unchanged; see CORRECTIONS.md',
    accepted_development_median_unchanged=329.078,default_changed=False,performance_promotion=False,
    phase10='ACTIVE',candidate='NOT_FROZEN',holdout='SEALED',recommended=False,
    analysis_sha256=sha(out/'analysis.json'),protocol_sha256=sha(out/'protocol.json'),
    canonical_report_sha256=sha(out/'capture/report.json'))
write(base/'closure.json',closure)
files={str(p.relative_to(base)):sha(p) for p in sorted(base.rglob('*')) if p.is_file()}
write(base/'manifest.json',dict(schema='stage-b-attribution-evidence-v1',files=files,
    native_sha256=sha(NATIVE),source_sha256=b['source_sha256'],
    tools={str(p.relative_to(ROOT)):sha(p) for p in sorted(TOOL.iterdir()) if p.is_file()},
    references={str(p.relative_to(ROOT)):sha(p) for p in [PARENT/'build.json',PARENT/'manifest.json',PARENT/'on3/receipt.rds',
      ROOT/'fastkpc/artifacts/kpc_tprs_real_zhu/cancer_RD-causalDiscoveryInput.rds',
      ROOT/'fastkpc/tools/research/optimizer_l3_slot_runtime_20261028/run.R',
      ROOT/'fastkpc/tools/research/exact_speedup_20261020/pinned_optimizer_gates.R',
      ROOT/'fastkpc/tools/with_phase10_exact_development_env.sh',ROOT/'fastkpc/tools/run_phase10_pvalue_contract_audit.R']}))
print('PASS closure',sha(base/'closure.json'),'manifest',sha(base/'manifest.json'),flush=True)
