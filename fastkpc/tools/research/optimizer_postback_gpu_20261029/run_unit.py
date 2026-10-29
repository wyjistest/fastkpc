"""Fail-closed input checks, then GPU reference/candidate and sanitizers."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import ROOT, TOOL, AUDIT, sha, write
sys.path.insert(0, str(TOOL.parent/'stage_b_work_pair_runtime_20261028'))
from support import environment, processes, job

out = Path(sys.argv[1]).resolve()
assert (out/'unit').is_file() and not (out/'unit-protocol.json').exists()
env = environment()
identity = json.loads((out/'source.json').read_text())
review = TOOL.parent/'optimizer_postback_audit_review_20261029'
dest = out/'host-input-check';dest.mkdir()
for p in review.iterdir():
    if p.is_file(): shutil.copyfile(p, dest/p.name)
subprocess.run([sys.executable, '-I', str(review/'export_blocks.py'),
    str(AUDIT/'inventory.json'), str(out/'blocks.tsv')], check=True)
command = ['g++', '-O2', '-std=c++17', '-ffp-contract=off', '-Wno-write-strings',
    '-DVENDOR_INC="'+str(AUDIT/'symbolic_vendor.inc')+'"',
    str(dest/'slice_check.cpp'), '-lcrypto', '-o', str(dest/'slice_check')]
with (dest/'build.log').open('x') as log:
    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
data = (out/'blocks.tsv').read_bytes()
for name, payload in [('empty', b''), ('truncated', data[:-1]), ('altered', b'0'+data[1:])]:
    (dest/(name+'.tsv')).write_bytes(payload)
results = []
for name,args,expected in [
    ('valid',[str(out/'blocks.tsv'),'--validate-only'],0),
    ('missing',[str(dest/'missing.tsv'),'--validate-only'],2),
    ('no-arguments',[],2),
    *[(name,[str(dest/(name+'.tsv')),'--validate-only'],2) for name in ['empty','truncated','altered']]]:
    p = subprocess.run([str(dest/'slice_check'), *args], capture_output=True, text=True)
    results.append(dict(name=name, exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr))
    assert p.returncode == expected, results[-1]
write(dest/'result.json', dict(status='PASS_INPUT_GATE', floating_point_evaluation=False,
    command=command, tests=results))
deps = {str(p):sha(p) for p in [out/'unit',out/'source.json',out/'blocks.tsv',
    out/'source/cuda/lapack_312_small_dgesdd.cuh',TOOL/'unit.cu',TOOL/'candidate.inc',TOOL/'run_unit.py']}
def frozen():
    assert sha(ROOT/'fastkpc/build/fastkpc_cuda.so') == identity['native_sha256']
    for p,h in identity['production_source_sha256'].items(): assert sha(ROOT/p)==h,p
    for p,h in deps.items(): assert sha(p)==h,p
frozen();assert not processes(),'external compute; no new job started'
write(out/'unit-protocol.json',dict(schema='postback-gpu-unit-v1',epoch=time.time(),dependencies=deps,
    expected_full_cases=20628,expected_sanitizer_cases=1719,physical_gpu=1,
    cuda_reference=True,cuda_candidate=True,host_numerical_authority=False,
    gates=['all C/W bits','read-only and guard bits','finite-output NaN absence','info=0',
           'memcheck errors/leaks=0','synccheck errors=0'],
    performance_claim=False,production_integration=False,
    scope='DLARFB all 1553 block shapes and complete Q/P for all 83 augmented shapes'))
unit = [str(out/'unit'), str(out/'blocks.tsv')]
for name,command in [
    ('unit-full',unit),
    ('unit-memcheck',['/usr/local/cuda/bin/compute-sanitizer','--tool','memcheck',
        '--leak-check','full','--error-exitcode','86',*unit,'--reduced']),
    ('unit-synccheck',['/usr/local/cuda/bin/compute-sanitizer','--tool','synccheck',
        '--error-exitcode','86',*unit,'--reduced'])]:
    print('START',name,flush=True);job(out,name,command,env,frozen)
    lines=(out/(name+'.log')).read_text().splitlines()
    result=json.loads(next(line for line in lines if line.startswith('{')))
    assert result['cases']==result['expected']==(20628 if name=='unit-full' else 1719)
    assert all(result[k]==0 for k in ['bit_mismatches','readonly_or_guard_changes','finite_output_nan_count','info_errors'])
    if name!='unit-full':assert any('ERROR SUMMARY: 0 errors' in line for line in lines)
    write(out/(name+'.json'),result)
write(out/'unit-closure.json',dict(status='PASS_GPU_VECTOR_AND_BLOCKED_POSTBACK',
    full_cases=20628,memcheck_cases=1719,synccheck_cases=1719,
    production_native_unchanged=True,optimizer_cohorts_qualified=False))
print('PASS isolated GPU unit qualification; optimizer cohorts remain to be checked',flush=True)
