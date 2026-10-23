"""Build external L2 observers/probes; never install or modify the native."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[4]
here=Path(__file__).resolve().parent
out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=True)
native=root/'fastkpc/build/fastkpc_cuda.so'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
expected='89a2fab0bd02b695878556a74694fc522cddcbe2c5c2f22f74c7e4da6d5d56c3'
assert sha(native)==expected
if len(sys.argv)>2 and sys.argv[2]=='--probe':
    source=root/'fastkpc/src/cuda/full_cuda_ci_vertical.cu'
    text=source.read_text()
    start=text.index('    check_cuda(cudaEventRecord(execution_context->pair_start(), stream)')
    last='               "record legacy eig pair end");'
    pair=text[start:text.index(last,start)+len(last)]
    assert pair.count('cublasDgemmBatched')==1 and pair.count('evaluate_legacy_eig_pair_batch_kernel')==1
    (out/'pair_stage.inc').write_text(pair+'\n')
    command=['/usr/local/cuda/bin/nvcc','-O3','-arch=sm_89','-std=c++17',
      '-Xcompiler=-fPIC','-shared','-Xlinker=-Bsymbolic','-I'+str(out),
      '-I'+str(root/'fastkpc/src/cuda'),'-I'+str(root/'fastkpc/src'),str(here/'probe.cu'),
      str(root/'fastkpc/build/full_cuda_ci_structured_eig.o'),str(native),
      '-Xlinker=-rpath='+str(native.parent),'-L/usr/local/lib/R/lib','-lR','-lcublas','-lcusolver',
      '-o',str(out/'probe.so')]
    print(' '.join(command),flush=True);subprocess.run(command,check=True)
    assert sha(native)==expected
    (out/'probe-build.json').write_text(json.dumps(dict(native_sha256=expected,commands=[command],
      sources={str(p.relative_to(root)):sha(p) for p in (here/'probe.cu',here/'build.py',source)},
      pair_stage_sha256=sha(out/'pair_stage.inc'),structured_object_sha256=sha(root/'fastkpc/build/full_cuda_ci_structured_eig.o'),
      probe_sha256=sha(out/'probe.so')),indent=2)+'\n')
    raise SystemExit(0)
raw=subprocess.check_output(['nm','-D','--defined-only',str(native)],text=True)
symbols=[r.split()[-1] for r in raw.splitlines()]
demangled=subprocess.check_output(['c++filt'],input='\n'.join(symbols),text=True).splitlines()
names=dict(plan='make_layer_plan',setup='fastkpc::full_cuda_ci_native_setup',
 geometry='fastkpc::full_cuda_ci_native_geometry_prepare',single='fastkpc::single_penalty_gcv_cuda_multi',
 create='fastkpc::create_multi_penalty_gcv_capacity_prepared',fixed='fastkpc::multi_penalty_gcv_capacity_evaluate_fixed_batch',
 prepared_info='fastkpc::multi_penalty_gcv_capacity_prepared_info',
 acquire='fastkpc::acquire_multi_penalty_gcv_capacity_residual_consumer_view',
 consumer='fastkpc::run_full_cuda_ci_phase35_legacy_eig_batch_impl')
lines=[]
for alias,name in names.items():
    matches=[s for s,d in zip(symbols,demangled) if name+'(' in d]
    assert len(matches)==1,(alias,matches)
    lines.append('constexpr const char* SYM_'+alias+' = "'+matches[0]+'";')
(out/'symbols.inc').write_text('\n'.join(lines)+'\n')
rcpp=subprocess.check_output(['Rscript','--vanilla','-e','cat(system.file("include",package="Rcpp"))'],text=True)
command=['g++','-O2','-std=c++17','-fPIC','-shared',str(here/'capture.cpp'),
 '-I'+str(out),'-I'+str(root/'fastkpc/src'),'-I'+str(root/'fastkpc/src/cuda'),
 '-I/usr/local/cuda/include','-I/usr/local/lib/R/include','-I'+rcpp,
 '-L/usr/local/cuda/lib64','-Wl,-rpath,/usr/local/cuda/lib64',
 '-L/usr/local/lib/R/lib','-Wl,-rpath,/usr/local/lib/R/lib','-lR','-lcudart','-lcublas','-lcrypto','-ldl',
 '-o',str(out/'capture.so')]
print(' '.join(command),flush=True);subprocess.run(command,check=True)
assert sha(native)==expected
(out/'build.json').write_text(json.dumps(dict(native_sha256=expected,commands=[command],
 sources={str(p.relative_to(root)):sha(p) for p in (here/'capture.cpp',here/'build.py')},
 capture_sha256=sha(out/'capture.so')),indent=2)+'\n')
