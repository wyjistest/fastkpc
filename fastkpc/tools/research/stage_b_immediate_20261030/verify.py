"""Read-only closure verification and compact reporting; does not run CUDA.

Added after registration for evidence review, not part of the experiment.
Missing local captures/binaries fail rather than yielding a successful SKIP.
"""
import csv
import json
from pathlib import Path
import statistics
import sys
from build import ROOT, TOOL, CAPTURE, identity, read, sha

out=Path(sys.argv[1]).resolve()
identity()
b=read(out/'build.json');closure=read(out/'closure.json')
for group in ['tools','dependencies']:
    for name,expected in b[group].items():assert sha(ROOT/name)==expected,name
for name,expected in closure['files'].items():assert sha(out/name)==expected,name
campaign=read(out/'campaign.json');timing=read(out/'timing.json');q=read(out/'qualify.json')
assert campaign['status']=='PASS_ISOLATED_IMMEDIATE_CAMPAIGN'
assert campaign['build_sha256']==sha(out/'build.json')
assert campaign['protocol_sha256']==sha(out/'protocol.json')
assert campaign['qualification_sha256']==sha(out/'qualify.json')
assert campaign['timing_sha256']==sha(out/'timing.json')
access=read(out/'candidate-access-audit.json')
assert access['tool_sha256']==sha(TOOL/'audit_candidate.py') and access['status']=='PASS_TARGET_ACCESS_REMOVAL'
assert q['captured']['targets']==104797 and q['captured']['mismatches']==q['synthetic']['mismatches']==0
assert q['captured']['checkpoint_stages'][1]==q['captured']['sweeps']
assert q['negative_controls']==2 and (q['captured']['coverage_mask']|q['synthetic']['coverage_mask'])==2047
rows=timing['runs']
assert [r['name'] for r in rows]==['A1','C1','C2','A2','A3','C3']
both=all(timing['level_medians']['candidate'][j]<timing['level_medians']['baseline'][j] for j in [0,1])
ratio=sum(timing['level_medians']['candidate'])/sum(timing['level_medians']['baseline'])
separated=max(r['total'] for r in rows if r['variant']=='candidate')<min(r['total'] for r in rows if r['variant']=='baseline')
decision='GO_ISOLATED_STAGE_B' if ratio<=.95 and both and separated else 'CONDITIONAL' if ratio<1 else 'STOP_NO_STAGE_GAIN'
assert timing['decision']==campaign['decision']==closure['decision']==decision
replays=['native_control','baseline_authentication',*[r['name'] for r in rows]]
telemetry={}
for name in ['resources','memcheck','qualify',*replays]:
    p=read(out/(name+'-process.json'));assert p['exit_code']==0 and p['isolated'] and not p['unexpected']
    values={0:[],1:[]}
    with (out/(name+'-gpu.csv')).open() as f:
        for row in csv.reader(f):
            assert len(row)==7
            values[int(row[1])].append([float(x.strip()) for x in row[2:]])
    assert values[0] and values[1]
    telemetry[name]=dict(samples=[len(values[0]),len(values[1])],
        gpu0_max_sampled_memory_mib=max(v[0] for v in values[0]),
        gpu0_max_sampled_utilization_percent=max(v[1] for v in values[0]),
        gpu1_max_sampled_memory_mib=max(v[0] for v in values[1]))
    # Both-idle condition is separately checked against the 200 ms telemetry;
    # this does not assert continuous sampling coverage between observations.
    assert telemetry[name]['gpu0_max_sampled_memory_mib']==50
    assert telemetry[name]['gpu0_max_sampled_utilization_percent']==0
    if name in replays:
        for level in [2,3]:
            assert sha(out/name/f'l{level}-payload-sha256.txt')==sha(CAPTURE/f'l{level}-payload-sha256.txt')
            row=read(out/name/f'l{level}-observation.json');assert row['dense_fallback']==0
        assert read(out/name/'mode.json')['pass']
        assert 'PASS native replay resource closure' in (out/(name+'.log')).read_text()
result=dict(status='PASS_READ_ONLY_CLOSURE',decision=decision,
    source_inputs=len(b['source_sha256']),frozen_tools=len(b['tools']),dependencies=len(b['dependencies']),
    closure_files=len(closure['files']),post_registration_analysis_tools={
        str((TOOL/n).relative_to(ROOT)):sha(TOOL/n) for n in ['audit_candidate.py','verify.py']},
    native_sha256=b['native_sha256'],build_sha256=sha(out/'build.json'),protocol_sha256=sha(out/'protocol.json'),
    closure_sha256=sha(out/'closure.json'),captured=q['captured'],synthetic=q['synthetic'],
    timing=dict(runs=[dict(name=r['name'],l2=r['levels'][0]['stage_b_sec'],l3=r['levels'][1]['stage_b_sec'],
               total=r['total']) for r in rows],level_medians=timing['level_medians'],
               sum_of_level_medians=timing['sum_of_level_medians'],median_of_run_totals=timing['median_of_run_totals'],
               ratio=ratio,fraction_reduction=1-ratio,both_levels_improve=both,separated=separated,
               slowest_candidate=max(r['total'] for r in rows if r['variant']=='candidate'),
               fastest_baseline=min(r['total'] for r in rows if r['variant']=='baseline')),
    telemetry=telemetry,full_replay_component_hash_comparisons=8*104797,
    raw_payload_hash_is_not_field_comparison=True,accepted_development_median_unchanged=283.328)
print(json.dumps(result,indent=2))
