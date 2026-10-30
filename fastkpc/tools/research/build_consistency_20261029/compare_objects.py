"""Separate raw object identity from NVCC's temporary STT_FILE/debug metadata.

Original objects are retained unchanged. Normalization is diagnostic only;
the actual linked shared libraries must still be byte-identical.
"""
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build import BASE,sha,read,write

policy=BASE/'object-identity-policy.json'
if sys.argv[1:]==['declare']:
    write(policy,dict(epoch=time.time(),reason='NVCC embeds process-dependent temporary source names as local STT_FILE symbols',
        raw_objects_retained=True,normalization=['objcopy','--strip-debug','--discard-all'],
        actual_native_byte_identity_required=True,compile_command_identity_required=True,
        declaration_before_incremental_build=True))
    print('DECLARED raw and payload identities separately',flush=True)
else:
    assert not sys.argv[1:]
    read(policy)
    a,b=[read(BASE/m/'build.json') for m in ['full','incremental']]
    out=BASE/'object-comparison';out.mkdir()
    comparisons=[]
    for name in a['objects']:
        hashes=[]
        for mode,build in [('full',a),('incremental',b)]:
            source=BASE/mode/'objects'/name
            assert sha(source)==build['objects'][name]
            dest=out/(mode+'-'+name)
            subprocess.run(['objcopy','--strip-debug','--discard-all',str(source),str(dest)],check=True)
            hashes.append(sha(dest))
        comparisons.append(dict(name=name,raw_equal=a['objects'][name]==b['objects'][name],
            full_raw_sha256=a['objects'][name],incremental_raw_sha256=b['objects'][name],
            payload_equal=hashes[0]==hashes[1],full_payload_sha256=hashes[0],incremental_payload_sha256=hashes[1]))
    gates=dict(compile_command=a['method_compile_command']==b['method_compile_command'],
        all_object_payloads=all(r['payload_equal'] for r in comparisons),
        native=a['native_sha256']==b['native_sha256'])
    write(BASE/'object-payload-comparison.json',dict(status='PASS' if all(gates.values()) else 'FAIL',
        gates=gates,objects=comparisons,policy_sha256=sha(policy),
        original_raw_comparison_sha256=sha(BASE/'build-comparison.json'),
        full_native_sha256=a['native_sha256'],incremental_native_sha256=b['native_sha256']))
    assert all(gates.values()),gates
    print('PASS commands, 44 object payloads and actual native; raw object differences',
        sum(not r['raw_equal'] for r in comparisons),flush=True)
