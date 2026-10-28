"""Run each frozen L3 qualification stage once, preserving all attempts."""
import csv
from pathlib import Path
import sys
from common import ROOT,TOOL,sha,load,write_new,check_identity,environment,monitored

out=Path(sys.argv[1]).resolve();mode=sys.argv[2]
assert mode in ('checks','memcheck','capacity','campaign','trace-W','trace-S')
b=check_identity(out);capture=load(out/'capture-closure.json')
protocol=dict(schema='l3-slots-protocol-v1',build_sha256=sha(out/'prototype-build.json'),
    declaration_sha256=sha(TOOL/'README.md'),capture_sha256=sha(out/'capture-closure.json'),
    workers=32,capacity=33,setups=4065,targets=54878,iterations=capture['census']['iterations'],
    original_windows=[64]*63+[33],warmup_each=1,balanced_pairs=6,
    maximum_go_lifecycle_ratio=.90,required_paired_lifecycle_wins=6,
    max_physical_increment_bytes=2*1024**3,minimum_free_bytes=2*1024**3,
    capacity_basis='incremental above compute-warm production context; includes full-output arena',
    production_cache_present=False,scope='captured-cohort prototype; not runtime integration',
    trace_performance_evidence=False,trace_common_native_warmup=False)
if mode=='checks':write_new(out/'protocol.json',protocol)
else:
    assert load(out/'protocol.json')==protocol
    assert load(out/'checks-summary.json')['status']=='PASS'
if mode in ('capacity','campaign','trace-W','trace-S'):
    assert load(out/'memcheck-summary.json')['status']=='PASS'
    log=(out/'memcheck.log').read_text()
    assert 'ERROR SUMMARY: 0 errors' in log and '0 bytes leaked' in log
if mode in ('campaign','trace-W','trace-S'):
    assert load(out/'capacity-summary.json')['status']=='PASS'
    rows=list(csv.DictReader((out/'capacity-runs.csv').open()))
    assert len(rows)==1 and int(rows[0]['physical_increment_bytes'])<=2*1024**3
env=environment();cmd=[str(out/'replay'),str(out/'capture-raw'),str(out/mode),mode]
if mode=='memcheck':
    cmd=['/usr/local/cuda/bin/compute-sanitizer','--tool','memcheck','--leak-check','full','--error-exitcode','99']+cmd
if mode.startswith('trace-'):
    directory=out/mode;directory.mkdir()
    env.update(CUDA_INJECTION64_PATH=str(out/'ktrace.so'),KTRACE_DIR=str(directory))
monitored(out,mode,cmd,env,dict(protocol_sha256=sha(out/'protocol.json'),
    traced=mode.startswith('trace-'),runner_sha256=sha(TOOL/'run.py')))
check_identity(out)
assert load(out/(mode+'-summary.json'))['status']=='PASS'
