"""Export uninstrumented binaries; query launch bounds without numerical work."""
import ctypes
from pathlib import Path
import re
import subprocess
import sys
from build import ROOT, read, sha, write


def instructions(text):
    result=[]
    for line in text.splitlines():
        m=re.search(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*/\* 0x[0-9a-f]+ \*/',line)
        if m:result.append((int(m[1],16),' '.join(m[2].split())))
    return result


def offline(out):
    rows={}
    for variant in ['baseline','candidate']:
        name=variant+'_timing'
        text=(out/(name+'-resources.log')).read_text()
        entries=re.findall(r'Function (.*?):\n\s+(REG:.*?)\n',text)
        selected=[(symbol,values) for symbol,values in entries if 'bidiagonal' in symbol and 'ModeE1E' in symbol]
        assert len(selected)==1,selected
        symbol,values=selected[0]
        resource={k:int(v) for k,v in re.findall(r'\b(REG|STACK|SHARED|LOCAL):(\d+)',values)}
        parts=re.split(r'Function : (.*?)\n',(out/(name+'-sass.log')).read_text())
        matching=[body for sym,body in zip(parts[1::2],parts[2::2]) if sym.strip()==symbol]
        assert len(matching)==1
        ins=instructions(matching[0]);assert len(ins)>100
        folder=out/(name+'-binary');folder.mkdir()
        (folder/'fixed.sass').write_text(''.join(f'{pc:05x}  {op}\n' for pc,op in ins))
        commands=[]
        for label,command in [
            ('extract',['/usr/local/cuda/bin/cuobjdump','--extract-elf','all',str(out/(name+'.so'))])]:
            p=subprocess.run(command,cwd=folder,capture_output=True)
            (folder/(label+'.stdout')).write_bytes(p.stdout);(folder/(label+'.stderr')).write_bytes(p.stderr)
            commands.append(dict(argv=command,exit_code=p.returncode));assert p.returncode==0
        cubins=sorted(folder.glob('*.cubin'));selected_cubins=[]
        for item in cubins:
            command=['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(item)]
            p=subprocess.run(command,capture_output=True)
            (folder/(item.name+'.resources')).write_bytes(p.stdout)
            commands.append(dict(argv=command,exit_code=p.returncode));assert p.returncode==0
            if symbol.encode() in p.stdout:selected_cubins.append(item)
        assert len(selected_cubins)==1,selected_cubins
        cubin=selected_cubins[0]
        command=['/usr/local/cuda/bin/nvdisasm','--print-code','--print-life-ranges',
                 '--life-range-mode','wide',str(cubin)]
        p=subprocess.run(command,capture_output=True)
        (folder/'liveness.txt').write_bytes(p.stdout);(folder/'liveness.stderr').write_bytes(p.stderr)
        commands.append(dict(argv=command,exit_code=p.returncode));assert p.returncode==0
        counts={op:sum(bool(re.search(r'\b'+re.escape(op)+r'(?:\.|\s)',i)) for _,i in ins)
                for op in ['LDS','STS','LDG','STG','LDL','STL','BRA','CALL','DMUL','DADD','DFMA']}
        targets=[int(m.group(1),16) for _,i in ins if (m:=re.search(r'CALL.REL.NOINC (0x[0-9a-f]+)',i))]
        pcs={pc for pc,_ in ins}
        assert all(t in pcs for t in targets),'helper export incomplete'
        row=dict(symbol=symbol,resources=resource,static_instruction_count=len(ins),
                 static_op_counts=counts,cubin=str(cubin.relative_to(out)),cubin_sha256=sha(cubin),
                 library_sha256=sha(out/(name+'.so')),extracted_cubin_sha256={p.name:sha(p) for p in cubins},
                 helper_call_sites=len(targets),
                 all_helper_targets_exported=True,commands=commands,dynamic_shared=(5 if variant=='baseline' else 3)*351*8)
        if variant=='baseline':
            accepted=ROOT/'fastkpc/artifacts/full_cuda_ci/fixed_vt_dbdsqr_binary_20261030/indexed/fixed.sass'
            reference=[(int(m[1],16),' '.join(m[2].split())) for line in accepted.read_text().splitlines()
                       if (m:=re.fullmatch(r'([0-9a-f]+)  (.*)',line))]
            row['accepted_instruction_sequence_exact']=ins==reference
            row['accepted_instruction_inventory_sha256']=sha(accepted)
        rows[variant]=row
    ok=(rows['baseline']['resources']['REG']==82 and rows['baseline']['accepted_instruction_sequence_exact'] and
        all(r['resources']['STACK']==r['resources']['LOCAL']==r['static_op_counts']['LDL']==r['static_op_counts']['STL']==0
            for r in rows.values()))
    write(out/'binary-resources.json',dict(status='PASS' if ok else 'STOP_RESOURCE',variants=rows,
          static_counts_are_not_dynamic_cost=True,hardware_occupancy_measured=False))
    print({k:r['resources'] for k,r in rows.items()},flush=True)
    assert ok,'resource/authentication gate failed; inspect retained binary exports'


def query(out):
    binary=read(out/'binary-resources.json');assert binary['status']=='PASS'
    rows={}
    for variant in ['baseline','candidate']:
        lib=ctypes.CDLL(str(out/(variant+'_timing.so')),mode=ctypes.RTLD_LOCAL)
        fn=lib.ip_resources;fn.argtypes=[ctypes.POINTER(ctypes.c_int)];fn.restype=None
        values=(ctypes.c_int*5)();fn(values)
        row=dict(zip(['occupancy_api_blocks','registers','local_bytes','static_shared','dynamic_shared'],values))
        ref=binary['variants'][variant]
        assert row['registers']==ref['resources']['REG'] and row['local_bytes']==0
        assert row['dynamic_shared']==ref['dynamic_shared'] and row['static_shared']==0
        rows[variant]=row
    write(out/'resources.json',dict(status='PASS_RESOURCE_GATE',variants=rows,
          occupancy_api_is_bound_not_measured_occupancy=True,numerical_kernels_executed=False))
    print(rows,flush=True)


if __name__=='__main__':
    out=Path(sys.argv[1]).resolve()
    if len(sys.argv)>2 and sys.argv[2]=='--offline':offline(out)
    else:query(out)
