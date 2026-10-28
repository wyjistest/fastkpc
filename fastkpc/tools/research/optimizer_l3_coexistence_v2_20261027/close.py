"""Audit raw samples and seal only the newly declared coexistence qualification."""
import csv
from pathlib import Path
import sys
from campaign import ROOT,TOOL,OLD,NATIVE,sha,load,write_new,validate

out=Path(sys.argv[1]).resolve()
build=validate(out);summary={}
for mode in ['W','S']:
    closure=load(out/(mode+'-closure.json'));assert closure['status']=='PASS'
    probe=closure['probe'];report=load(out/mode/'report.json')
    rows=list(csv.DictReader((out/mode/'memory-samples.csv').open()))
    checkpoints=list(csv.DictReader((out/mode/'checkpoints.csv').open()))
    minimum=min(int(r['free_bytes']) for r in rows)
    assert len(rows)==probe['observations'] and minimum==probe['minimum_free_bytes'] and minimum>=2*1024**3
    assert all(int(r['capacity_pass'])==1 for r in checkpoints)
    assert len([r for r in checkpoints if r['checkpoint']=='stage_a_cohort'])==5573
    assert report['comparisons']['changed_ci_count']==report['comparisons']['changed_pmax_count']==0
    assert report['comparisons']['alpha_flips']==0
    samples=[r for r in rows if r['kind']=='sample']
    times=sorted(int(r['monotonic_ns']) for r in samples)
    gaps=[(b-a)/1e6 for a,b in zip(times,times[1:])]
    process=load(out/(mode+'-process.json'));assert process['gpu1_exclusive']
    if mode=='S':
        by_name={r['checkpoint']:r for r in checkpoints}
        b=int(by_name['pool_before']['free_bytes']);c=int(by_name['pool_created']['free_bytes'])
        o=int(by_name['optimizer_done']['free_bytes'])
        assert b-c==probe['creation_increment_bytes']<=1536*1024**2
        assert c-o==probe['optimizer_growth_bytes']<=1280*1024**2
        assert b-o==probe['optimizer_total_increment_bytes']<=2816*1024**2
        stage=[r for r in rows if int(r['phase'])==2]
        assert min(int(r['free_bytes']) for r in stage)==probe['stage_a_minimum_free_bytes']
        assert max(int(r['raw_total_increment_bytes']) for r in stage)==probe['stage_a_maximum_raw_total_increment_bytes']
        assert all(int(r['raw_total_increment_bytes'])==max(0,int(r['baseline_free_bytes'])-int(r['free_bytes'])) for r in rows)
        p=by_name['pool_created'];e=by_name['optimizer_done']
        assert all(p[k]==e[k] for k in ['application_bytes','allocations','streams','events','blas'])
    summary[mode]=dict(probe=probe,cleanup=closure['cleanup'],canonical_gates=report['gates'],
        observations=len(rows),periodic_samples=len(samples),maximum_periodic_gap_ms=max(gaps),
        sample_coverage='10 ms requested cadence; scheduling/query gaps retained; not continuous peak coverage',
        comparisons=report['comparisons'],oracle_errors=report['oracle_errors'],authority_pmax=report['authority_pmax'],
        diagnostic_elapsed_sec=report['elapsed_sec'],performance_evidence=False,process=process)
write_new(out/'closure.json',dict(status='GO_L3_REPLACEMENT_COEXISTENCE_SPLIT_CONTRACT',
    schema='l3-replacement-coexistence-split-raw-capacity-v2',native_sha256=sha(NATIVE),
    protocol_sha256=sha(out/'protocol.json'),build_sha256=sha(out/'build.json'),
    original_stop_retained=True,runtime_integration='NOT_YET_GO',performance_evidence=False,
    accepted_canonical_development_median_sec=368.851,W_S_minimum_free_comparison='descriptive only',
    production_source_count=len(build['source_sha256']),runs=summary))
manifest=dict(schema='immutable-l3-coexistence-split-capacity-manifest-v2',
    files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()},
    tools={str(p.relative_to(ROOT)):sha(p) for p in sorted(TOOL.iterdir()) if p.is_file()},
    frozen_references={str(p.relative_to(ROOT)):sha(p) for p in [OLD/'manifest.json',OLD/'closure.json',OLD/'W/stage-a.bin']})
write_new(out/'manifest.json',manifest)
for group,base in [('files',out),('tools',ROOT),('frozen_references',ROOT)]:
    for p,h in manifest[group].items():assert sha(base/p)==h,p
validate(out)
print('GO_L3_REPLACEMENT_COEXISTENCE_SPLIT_CONTRACT',flush=True)
print('closure',sha(out/'closure.json'),'manifest',sha(out/'manifest.json'),flush=True)
