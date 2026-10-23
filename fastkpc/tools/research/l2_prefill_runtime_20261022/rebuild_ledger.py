"""Recompile only the corrected C++ ledger within this campaign's full build.

The other 42 objects must match the just-completed, empty-directory build.
Archive the failed revision and verify every source/object before installation.
No object from an older campaign is used.
"""
import gzip
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[4]
os.chdir(root)
out = Path(sys.argv[1]).resolve()
snapshot = out/'build-source'
build_dir = snapshot/'fastkpc/build'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
receipt = json.loads((out/'build.json').read_text())
assert receipt['status']=='PASS' and not receipt.get('ledger_repair')
native = root/'fastkpc/build/fastkpc_cuda.so'
assert sha(native)==receipt['native_sha256']==sha(build_dir/'fastkpc_cuda.so')
changed = [p for p,h in receipt['source_sha256'].items() if sha(root/p)!=h]
source = 'fastkpc/src/full_cuda_ci_one_call.cpp'
assert changed==[source],changed
assert all(sha(snapshot/p)==h for p,h in receipt['source_sha256'].items())
assert all(sha(build_dir/p)==h for p,h in receipt['objects'].items())
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                    '--format=csv,noheader'],text=True).strip()
old = out/'failed-ledger-revision'
old.mkdir()
shutil.copyfile(out/'build.json',old/'build.json')
shutil.copyfile(snapshot/source,old/'full_cuda_ci_one_call.cpp')
shutil.copyfile(out/'smoke.log',old/'smoke.log')
with native.open('rb') as inp,(old/'fastkpc_cuda.so.gz').open('wb') as dest:
    with gzip.GzipFile(fileobj=dest,mode='wb',mtime=0) as compressed:
        shutil.copyfileobj(inp,compressed)
commands = [shlex.split(line[2:]) for line in (out/'build.log').read_text().splitlines()
            if line.startswith('+ ')]
compiles = [c for c in commands if '-c' in c and str(snapshot/source) in c]
links = [c for c in commands if '-shared' in c and '-o' in c]
assert len(compiles)==len(links)==1
compile_command = compiles[0]
link_command = links[0]
temporary = build_dir/'fastkpc_cuda.so.ledger-repair'
link_command[link_command.index('-o')+1] = str(temporary)
shutil.copyfile(root/source,snapshot/source)
new_sources = {p:sha(root/p) for p in receipt['source_sha256']}
started = time.time()
with (out/'ledger-rebuild.log').open('w') as log:
    for command in (compile_command,link_command):
        log.write('+ '+shlex.join(command)+'\n');log.flush()
        subprocess.run(command,cwd=snapshot,stdout=log,stderr=subprocess.STDOUT,check=True)
new_objects = {p:sha(build_dir/p) for p in receipt['objects']}
assert [p for p in receipt['objects'] if new_objects[p]!=receipt['objects'][p]]==['full_cuda_ci_one_call.o']
assert all(sha(root/p)==h==sha(snapshot/p) for p,h in new_sources.items())
os.replace(temporary,build_dir/'fastkpc_cuda.so')
shutil.copyfile(build_dir/'fastkpc_cuda.so',native)
shutil.copyfile(build_dir/'full_cuda_ci_one_call.o',root/'fastkpc/build/full_cuda_ci_one_call.o')
receipt['ledger_repair'] = dict(reason='L2 independent exact-fixed targets were incorrectly counted as coefficient-arena gathers',
    previous_native_sha256=receipt['native_sha256'],changed_source=source,
    before_source_sha256=receipt['source_sha256'][source],after_source_sha256=new_sources[source],
    unchanged_object_count=42,elapsed_sec=time.time()-started,
    driver_sha256=sha(Path(__file__)),failed_revision='failed-ledger-revision')
receipt.update(source_sha256=new_sources,objects=new_objects,native_sha256=sha(native),
    source_build='all 43 objects from empty directory, then ledger-only C++ recompile/relink; other 42 hashes unchanged')
(out/'build.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('PASS ledger correction source-closure',receipt['native_sha256'],flush=True)
