"""Post-build read-only instruction index for the first 3N candidate.

This analysis tool was added after campaign registration. It neither changes
the frozen tools/binaries/protocol nor supplies a numerical/performance result.
The address map is manually reviewed and pinned to normalized SASS bytes.
"""
from pathlib import Path
import re
import sys
from build import read, sha, write

out=Path(sys.argv[1]).resolve()
binary=read(out/'binary-resources.json')
row=binary['variants']['candidate']
folder=out/'candidate_timing-binary'
path=folder/'fixed.sass'
assert sha(path)=='1c8e2c4c735fdb2102db485e38835ee786dc0bfc8a2f73a6c71db5eb8642f811'
assert sha(out/row['cubin'])==row['cubin_sha256']
instructions={int(m[1],16):m[2] for line in path.read_text().splitlines()
              if (m:=re.fullmatch(r'([0-9a-f]+)  (.*)',line))}
live={};active=False
for line in (folder/'liveness.txt').read_text().splitlines():
    if '\t.section\t.text.' in line:active=('.text.'+row['symbol']+',') in line
    if active and (m:=re.search(r'/\*([0-9a-f]+)\*/.*// \|\s*(\d+)\s',line)):
        live[int(m[1],16)]=int(m[2])
assert len(live)>6000

# q ranges contain only two old-value LDS and two result STS. Coefficients
# reach both expressions and log stores in registers, including backward sign.
families={
    'shifted_backward':dict(start=0xa870,end=0xaa70,q='R36',coefficients=['R56:R57','-R4:R5'],
                           backedge=0xaab0,target=0x8710),
    'shifted_forward':dict(start=0xcfd0,end=0xd190,q='R31',coefficients=['R50:R51','R48:R49'],
                          backedge=0xd1d0,target=0xae40),
    'zero_backward':dict(start=0xf410,end=0xf630,q='R34',coefficients=['R46:R47','-R28:R29'],
                        backedge=0xf6a0,target=0xd3b0),
    'zero_forward':dict(start=0x11940,end=0x11b40,q='R31',coefficients=['R54:R55','R52:R53'],
                       backedge=0x11ba0,target=0xf8a0),
}
result={}
for family,cfg in families.items():
    segment=[(pc,instructions[pc]) for pc in range(cfg['start'],cfg['end']+1,16)]
    accesses=[]
    for pc,op in segment:
        if m:=re.search(r'\b(LDS|STS)\.64 .*\[(R\d+)(?:\+[^\]]*)?\]',op):
            assert m[2]==cfg['q'],(family,pc,op)
            accesses.append(dict(pc=hex(pc),role='old_q_load' if m[1]=='LDS' else 'q_store',
                                 instruction=op,live_gpr=live[pc]))
    assert [s['role'] for s in accesses]==['old_q_load','old_q_load','q_store','q_store']
    assert all(not re.search(r'\b(?:LDS|LDG|LDL|STS|STL)(?:\.|\s)',op)
               for pc,op in segment if int(accesses[2]['pc'],16)<pc<int(accesses[3]['pc'],16))
    logs=[dict(pc=hex(pc),instruction=op,live_gpr=live[pc]) for pc,op in segment if 'STG.E.64' in op]
    assert len(logs)==3
    assert instructions[cfg['backedge']]==f'BRA {hex(cfg["target"])} ;'
    result[family]=dict(consumer_span=[hex(cfg['start']),hex(cfg['end'])],
        coefficient_register_pairs=cfg['coefficients'],shared_accesses=accesses,rotation_stores=logs,
        single_combined_loop_backedge=hex(cfg['backedge']),loop_target=hex(cfg['target']),
        min_live_gpr=min(live[pc] for pc,_ in segment),max_live_gpr=max(live[pc] for pc,_ in segment),
        work_load_store_sites_in_consumer=0,coefficient_reload_between_q_updates=0)

lower=[]
for pc in range(0x1670,0x1731,16):
    op=instructions[pc]
    if m:=re.search(r'\b(LDS|STS)\.64 .*\[(UR\d+)(?:\+[^\]]*)?\]',op):
        assert m[2] in ['UR5','UR18']
        lower.append(dict(pc=hex(pc),array='d' if m[2]=='UR5' else 'e',instruction=op))
assert len(lower)==5 and instructions[0x1730]=='BRA 0x6c0 ;'
generated=(out/'candidate_timing/recorded_dbdsqr_fixed_vt.inc').read_text()
code=re.sub(r'/\*.*?\*/|//[^\n]*','',generated,flags=re.S)
assert not re.search(r'\bwork\b',code) and 'record_dlasr_' not in code
assert code.count('ip_consume_sweep(')==4
write(out/'candidate-access-audit.json',dict(schema='immediate-pinned-instruction-audit-v1',
    status='PASS_TARGET_ACCESS_REMOVAL',tool_sha256=sha(Path(__file__)),
    analysis_tool_added_after_registration=True,build_sha256=sha(out/'build.json'),
    normalized_sass_sha256=sha(path),cubin_sha256=row['cubin_sha256'],
    interpretation='manual source/register attribution with pinned instruction assertions; static evidence only',
    candidate_has_executable_work_expression=False,lower_conversion_accesses=lower,families=result,
    no_static_to_dynamic_or_seconds_conversion=True,performance_evidence=False))
print({name:dict(live_gpr_range=[r['min_live_gpr'],r['max_live_gpr']],q_shared_sites=len(r['shared_accesses']))
       for name,r in result.items()})
