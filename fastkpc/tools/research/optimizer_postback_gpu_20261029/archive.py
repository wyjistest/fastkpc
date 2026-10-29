"""Audit closed experiment and bind current tools, sources and all result files."""
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build import ROOT,TOOL,AUDIT,sha,write
out=Path(sys.argv[1]).resolve()
identity=json.loads((out/'source.json').read_text())
assert len(identity['production_source_sha256'])==104
assert sha(ROOT/'fastkpc/build/fastkpc_cuda.so')==identity['native_sha256']
for p,h in identity['production_source_sha256'].items():assert sha(ROOT/p)==h,p
prior=json.loads((AUDIT/'manifest.json').read_text())
assert sum(len(prior[s]) for s in ['production_source_sha256','inputs','tools','files'])==133
for section in ['production_source_sha256','inputs','tools','files']:
    for p,h in prior[section].items():assert sha((AUDIT if section=='files' else ROOT)/p)==h,p
for path in out.rglob('*-process.json'):
    result=json.loads(path.read_text());assert result['exit_code']==0 and result['isolated'],str(path)
for path in out.rglob('*protocol.json'):
    protocol=json.loads(path.read_text())
    for p,h in protocol.get('dependencies',{}).items():assert sha(p)==h,p
unit=json.loads((out/'unit-closure.json').read_text())
qualified=json.loads((out/'cohorts-qualification/closure.json').read_text())
profile=json.loads((out/'cohorts-profile/closure.json').read_text())
timing=json.loads((out/'cohorts-timing/closure.json').read_text())
assert unit['status']=='PASS_GPU_VECTOR_AND_BLOCKED_POSTBACK'
assert qualified['status']=='PASS_FULL_COHORT_BITWISE' and profile['status']=='PASS_PROFILE'
analysis={}
for run in profile['runs']:
    stats=run['layers'][-1]['profile_cumulative'];assert len(stats)==30
    levels={}
    for level,base in [(4,0),(5,15)]:
        sides={}
        for name,start in [('Q',base),('immediate_P',base+5),('deferred_P',base+10)]:
            calls,blocks,total,ft,fb=stats[start:start+5]
            assert ft+fb<=total
            assert blocks==calls*(19 if level==4 and name=='Q' else 18 if level==4 else 2)
            sides[name]=dict(calls=calls,blocks=blocks,total_cycles=total,dlarft_cycles=ft,dlarfb_cycles=fb)
        levels[str(level)]=sides
    analysis[run['variant']]=levels
for level in ['4','5']:
    for side in ['Q','immediate_P','deferred_P']:
        r=analysis['reference-profile'][level][side];c=analysis['candidate-profile'][level][side]
        assert r['calls']==c['calls'] and r['blocks']==c['blocks']
write(out/'profile-analysis.json',dict(status='PASS_COUNTS_AND_CYCLE_PARTITION',values=analysis,
    cycles_are_not_elapsed=True,profile_outputs_bitwise=True))
if timing['status']=='GO_ISOLATED_OUTER_QUALIFICATION':
    outer=json.loads((out/'canonical/closure.json').read_text())
    protocol=json.loads((out/'canonical/protocol.json').read_text())
    build=json.loads((out/'canonical/build.json').read_text())
    assert [(r['variant'],r['cycle']) for r in outer['runs']]==[tuple(r) for r in protocol['sequence']]
    assert len(outer['runs'])==6
    reports={}
    for run in outer['runs']:
        report=json.loads(Path(run['report']).read_text());assert all(report['gates'].values())
        assert report['elapsed_sec']==run['elapsed_sec']
        assert report['native_sha256']==build['native_hashes'][run['variant']]
        assert sha(Path(run['report']).parent/'receipt.rds')==report['receipt_sha256']
        assert report['comparisons']['changed_ci_count']==report['comparisons']['changed_pmax_count']==0
        assert sum(r['count'] for r in report['comparisons']['by_level'])==240498
        reports[(run['variant'],run['cycle'])]=report
    medians={v:statistics.median(reports[(v,i)]['elapsed_sec'] for i in [1,2,3])
             for v in ['reference','candidate']}
    wins=sum(reports[('candidate',i)]['elapsed_sec']<reports[('reference',i)]['elapsed_sec'] for i in [1,2,3])
    ratio=medians['candidate']/medians['reference']
    assert outer['medians']==medians and outer['ratio']==ratio and outer['paired_wins']==wins
    passed=ratio<=protocol['performance_gate']['candidate_median_ratio_max'] and wins==protocol['performance_gate']['paired_wins']
    assert outer['status']==('GO_ISOLATED_POSTBACK_DEVELOPMENT' if passed else 'STOP_OUTER_WALL_TIME')
    derived={}
    for variant in ['reference','candidate']:
        rows=[reports[(variant,i)] for i in [1,2,3]]
        by_level={str(level):statistics.median(next(x['host_sec'] for x in r['stages']['optimizer_by_level']
                      if x['level']==level) for r in rows) for level in range(1,9)}
        derived[variant]=dict(
            outer_median_sec=medians[variant],
            l45_optimizer_sum_median_sec=statistics.median(sum(x['host_sec'] for x in r['stages']['optimizer_by_level']
                if x['level'] in [4,5]) for r in rows),
            optimizer_by_level_median_sec=by_level,
            stage_medians_sec={field:statistics.median(r['stages'][field] for r in rows)
                for field in ['optimizer_sec','stage_a_sec','stage_b_sec','consumer_dcov_sec','native_setup_sec']},
            gate_counts=[len(r['gates']) for r in rows],
            maximum_authority_error=max(max(r['oracle_errors']) for r in rows),
            minimum_observed_capacity_free_bytes=min(r['stages'][pool]['minimum_free_bytes']
                for r in rows for pool in ['l3_pool','l4_pool','l6_pool']),
            minimum_l5_post_creation_free_bytes=min(r['stages']['l5_slots']['free_after_create'] for r in rows))
    write(out/'canonical-analysis.json',dict(status='PASS_RECOMPUTED_OUTER_AUDIT',variants=derived,
        paired_savings_sec=[reports[('reference',i)]['elapsed_sec']-reports[('candidate',i)]['elapsed_sec'] for i in [1,2,3]],
        outer_median_saving_sec=medians['reference']-medians['candidate'],
        ratio=ratio,paired_wins=wins,
        capacity_scope='existing raw pool checkpoints, not a continuous device-memory bound',
        stage_scope='medians of named host boundaries; overlapping stages are not additive savings'))
    status=outer['status']
else:
    assert timing['status']=='STOP_LAYER_WALL_TIME';outer=None;status=timing['status']
for variant in ['candidate','reference-profile','candidate-profile']:
    b=json.loads((out/variant/'build.json').read_text())
    assert sha(out/variant/'fastkpc_cuda.so')==b['native_sha256']
    assert sha(out/variant/'small64.o')==b['object_sha256']
    for p,h in json.loads((out/variant/'build-protocol.json').read_text())['source_overlay_sha256'].items():
        assert sha(out/'source'/p)==h,p
snapshot=out/'tools';snapshot.mkdir()
tools={}
for p in TOOL.iterdir():
    if p.is_file():shutil.copyfile(p,snapshot/p.name);tools[str(p.relative_to(ROOT))]=sha(p)
shared={}
shared_git={}
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
for directory,names in [
    ('optimizer_windows_20261022',['fixture.hpp']),
    ('optimizer_l4_diagnostic_20261023',['fixture.hpp']),
    ('optimizer_layer_refill_20261023',['layer_fixture.hpp','audit.cpp','audit.hpp']),
    ('stage_b_work_pair_runtime_20261028',['support.py','run.R'])]:
    for name in names:
        p=TOOL.parent/directory/name;relative=str(p.relative_to(ROOT));shared[relative]=sha(p)
        committed=subprocess.check_output(['git','cat-file','blob',head+':'+relative],cwd=ROOT)
        shared_git[relative]=hashlib.sha256(committed).hexdigest()
        assert shared_git[relative]==shared[relative],relative
write(out/'shared-source-attestation.json',dict(status='PASS_CURRENT_EQUALS_COMMITTED_BLOBS',head=head,
    evidence_class='post-run source attestation; does not replace pre-run frozen dependency maps',
    source_sha256=shared_git))
write(out/'closure.json',dict(status=status,machine_audit='PASS',native_sha256=identity['native_sha256'],
    production_inputs_unchanged=104,unit=unit,cohort_qualification=qualified['status'],
    profile=profile['status'],timing=timing['analysis'],outer=outer,
    runtime_integration='NOT_YET_GO',accepted_development_median_sec=311.440,
    phase10='ACTIVE',candidate='NOT_FROZEN',holdout='SEALED',recommended=False,
    limitations=['captured cohorts and canonical only; no new real-data or final-PDAG qualification',
      'research overlay; accepted production source and native unchanged',
      'profile cycle sums include stalls and are not a wall-time savings pool'],
    head=head))
files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}
write(out/'manifest.json',dict(schema='postback-gpu-isolated-manifest-v1',
    native_sha256=identity['native_sha256'],production_source_sha256=identity['production_source_sha256'],
    static_audit_manifest_sha256=sha(AUDIT/'manifest.json'),tools=tools,shared_tools=shared,files=files))
print('PASS',status,'closure',sha(out/'closure.json'),'manifest',sha(out/'manifest.json'),flush=True)
