"""Rebuild only the structured consumer TU on the separately qualified parent."""
import importlib.util
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
TOOL=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('build_identity',TOOL.parent/'build_consistency_20261029/build.py')
identity=importlib.util.module_from_spec(spec);spec.loader.exec_module(identity)
ROOT=identity.ROOT;PARENT=identity.BASE;sha=identity.sha;read=identity.read;write=identity.write
BASE=ROOT/'fastkpc/artifacts/full_cuda_ci/structured_rows_20261029'

if __name__=='__main__':
    closure=read(PARENT/'closure.json');assert closure['status']=='PASS_BUILD_FIX_PARENT'
    old=read(PARENT/'full/build.json')
    assert sha(PARENT/'full/fastkpc_cuda.so')==old['native_sha256']==closure['native_sha256']
    sources=identity.source_map()
    changed=[p for p,h in sources.items() if old['source_sha256'].get(p)!=h]
    assert changed==['fastkpc/src/cuda/full_cuda_ci_vertical.cu'],changed
    out=BASE/'candidate';out.mkdir(parents=True)
    snapshot=out/'source'
    for p in sources:
        dest=snapshot/p;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/p,dest)
    (out/'objects').mkdir()
    for name,h in old['objects'].items():
        assert sha(PARENT/'full/objects'/name)==h,name
        shutil.copyfile(PARENT/'full/objects'/name,out/'objects'/name)
    for name in ['recorded_dbdsqr.inc','recorded_dbdsqr_fixed_vt.inc']:
        shutil.copyfile(PARENT/'full'/name,out/name)
    lines=(PARENT/'full/build.log').read_text().splitlines()
    prefix=str(PARENT/'source')
    compile_lines=[line[2:] for line in lines if line.startswith('+ /usr/local/cuda/bin/nvcc ') and
                   '/full_cuda_ci_vertical.cu ' in line]
    link_lines=[line[2:] for line in lines if line.startswith('+ ') and ' -shared -Wl,-x -o ' in line]
    assert len(compile_lines)==len(link_lines)==1
    compile_cmd=[arg.replace(prefix,str(snapshot)) for arg in shlex.split(compile_lines[0])]
    compile_cmd[compile_cmd.index('-o')+1]=str(out/'objects/full_cuda_ci_vertical.o')
    assert '--fmad=false' not in compile_cmd,'preserve original consumer TU policy'
    link_cmd=shlex.split(link_lines[0])
    link_cmd[link_cmd.index('-o')+1]=str(out/'fastkpc_cuda.so')
    for i,arg in enumerate(link_cmd):
        if arg.endswith('.o'):
            assert Path(arg).name in old['objects']
            link_cmd[i]=str(out/'objects'/Path(arg).name)
    commands=[compile_cmd,link_cmd]
    write(out/'protocol.json',dict(source_sha256=sources,changed_sources=changed,commands=commands,
        parent_closure_sha256=sha(PARENT/'closure.json'),parent_native_sha256=old['native_sha256'],
        scope='one rebuilt TU; all other full-build objects retained exactly',epoch=time.time()))
    for i,cmd in enumerate(commands):
        print('START row-only build',i,flush=True)
        with (out/('build-'+str(i)+'.log')).open('x') as log:
            subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    objects={name:sha(out/'objects'/name) for name in old['objects']}
    assert all(objects[n]==h for n,h in old['objects'].items() if n!='full_cuda_ci_vertical.o')
    assert all(sha(ROOT/p)==h and sha(snapshot/p)==h for p,h in sources.items())
    write(out/'build.json',dict(status='PASS',native_path=str(out/'fastkpc_cuda.so'),
        native_sha256=sha(out/'fastkpc_cuda.so'),source_sha256=sources,objects=objects,
        previous_native_sha256=old['native_sha256'],parent_closure_sha256=sha(PARENT/'closure.json'),
        evidence_class='single-TU row-only source closure on qualified full-build parent; not promotion'))
    print('PASS row-only build',sha(out/'fastkpc_cuda.so'),flush=True)
