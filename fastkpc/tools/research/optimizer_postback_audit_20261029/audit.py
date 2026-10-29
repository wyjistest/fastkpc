"""Bounded source/indices/symbolic audit; no CUDA or FP numerical evaluation."""
import collections
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
TOOL = Path(__file__).resolve().parent
PARENT = ROOT / 'fastkpc/artifacts/full_cuda_ci/optimizer_l45_attribution_20261029/attempt3'
VENDOR = ROOT / 'fastkpc/src/cuda/third_party/lapack_3_12_block_reflector_device.inc'
DORMBR = ROOT / 'fastkpc/src/cuda/third_party/lapack_3_12_dormbr_device.inc'
HEADER = ROOT / 'fastkpc/src/cuda/lapack_312_small_dgesdd.cuh'
PRODUCER = ROOT / 'fastkpc/src/cuda/mgcv_multi_penalty_gcv.cu'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def compact(text):
    return re.sub(r'\s+', '', text)


def calls(text, name, line_base=0):
    """Balanced arguments for the authenticated generated source, not a parser."""
    result = []
    for match in re.finditer(r'\b' + name + r'\(', text):
        start = pos = match.end()
        depth = 1
        while depth:
            depth += (text[pos] == '(') - (text[pos] == ')')
            pos += 1
        args, part, depth = [], start, 0
        for at in range(start, pos - 1):
            if text[at] in '([':
                depth += 1
            elif text[at] in ')]':
                depth -= 1
            elif text[at] == ',' and depth == 0:
                args.append(compact(text[part:at])); part = at + 1
        args.append(compact(text[part:pos-1]))
        result.append(dict(line=line_base + text.count('\n', 0, match.start()) + 1, args=args))
    return result


def body(text, name):
    matches = list(re.finditer(r'__device__ int ' + name + r'\([^;{}]*\)\s*\{', text))
    assert len(matches) == 1, name
    start = matches[0].start()
    end = text.index('} /* ' + name + ' */', start) + len('} /* ' + name + ' */')
    return text[start:end], text.count('\n', 0, start)


def blocks(q, side, rows):
    """The unchanged DORMQR(L,N) / DORMLQ(R,N) descending block order."""
    lwork = 3*q*q + 4*q
    nb = 32 if lwork >= q*32 + 4160 else (lwork-4160)//q
    assert 2 <= nb < q-1
    k = q if side == 'Q' else q-1
    for offset in range((k-1)//nb*nb, -1, -nb):
        ib = min(nb, k-offset)
        yield dict(q=q, side=side, offset=offset, ib=ib, nb=nb, lwork=lwork,
                   active=rows-offset if side == 'Q' else q-1-offset)


def shape_inventory():
    rows, symbolic, bounds, layouts = [], set(), 0, []
    for q in (37, 46):
        threshold = q*11//6
        nb = 32 if 3*q*q+4*q >= q*32+4160 else (3*q*q+4*q-4160)//q
        layouts.append(dict(q=q, augmented_rows=[q+1, 2*q], direct_rows=[q+1, threshold-1],
            qr_reduced_rows=[threshold, 2*q], nb=nb, lwork=3*q*q+4*q,
            q_block_count=len(list(blocks(q, 'Q', q))), p_block_count=len(list(blocks(q, 'P', q))),
            work_elements=q*nb, t_start_element=q*nb, t_leading_dimension=65,
            maximum_t_element=q*nb+66*(nb-1),
            sliced_dormbr_nw1_nb=32 if 3*q*q+4*q >= 32+4160 else (3*q*q+4*q-4160)))
        for augmented in range(q+1, 2*q+1):
            direct = augmented < threshold
            source_ld = augmented if direct else q
            for side in ('Q', 'P'):
                for block in blocks(q, side, source_ld):
                    offset, ib, active = block['offset'], block['ib'], block['active']
                    ldc = source_ld if side == 'Q' else q
                    crow, ccol = (offset, 0) if side == 'Q' else (0, 1+offset)
                    vr, vc = offset, offset if side == 'Q' else 1+offset
                    m, n = (active, q) if side == 'Q' else (q, active)
                    csets, wsets = [], []
                    for owner in range(q):
                        addresses = {(crow+i)+ldc*(ccol+owner) for i in range(m)} if side == 'Q' else {
                            (crow+owner)+ldc*(ccol+j) for j in range(n)}
                        scratch = {owner+q*j for j in range(ib)}
                        assert all(0 <= i < ldc*q for i in addresses)
                        assert all(0 <= i < q*nb for i in scratch)
                        csets.append(addresses); wsets.append(scratch)
                        bounds += len(addresses)+len(scratch)
                    assert sum(map(len,csets)) == len(set.union(*csets)) == m*n
                    assert sum(map(len,wsets)) == len(set.union(*wsets)) == q*ib
                    t = {q*nb+i+65*j for j in range(ib) for i in range(j+1)}
                    assert min(t) >= q*nb and max(t) < block['lwork'] <= 3*64*64+4*64
                    assert not (set.union(*wsets) & t)
                    vrows, vcols = (active, ib) if side == 'Q' else (ib, active)
                    assert vr+vrows <= source_ld and vc+vcols <= q
                    entry = dict(augmented_rows=augmented, path='direct' if direct else 'qr_reduced',
                        **block, m=m, n=n, ldv=source_ld, ldc=ldc, ldwork=q,
                        c_row_offset=crow, c_column_offset=ccol, v_row_offset=vr, v_column_offset=vc,
                        independent_vectors=q, dtrmm_calls=3, dgemm_calls=2 if active>ib else 0,
                        dcopy_calls=ib)
                    rows.append(entry)
                    # Symbolic endpoints: minimum/maximum direct rows and QR path.
                    # All legal intermediate shapes still get integer ownership checks.
                    if augmented in (q+1, threshold-1, threshold):
                        symbolic.add((q, 'L' if side == 'Q' else 'R', m, n, ib, source_ld, ldc))
    return rows, sorted(symbolic), bounds, layouts


def main():
    assert len(sys.argv) == 2, 'audit.py NEW_OUTPUT_DIRECTORY'
    out = Path(sys.argv[1]).resolve()
    assert not out.exists(), 'preserve existing evidence'
    parent = json.loads((PARENT/'manifest.json').read_text())
    native = ROOT/'fastkpc/build/fastkpc_cuda.so'
    assert sha(native) == parent['native_sha256']
    assert len(parent['source_sha256']) == 104
    assert all(sha(ROOT/p) == h for p,h in parent['source_sha256'].items())
    for name in ('analysis.json', 'canonical/report.json', 'closure.json'):
        assert sha(PARENT/name) == parent['files'][name]
    tools = {str(p.relative_to(ROOT)):sha(p) for p in TOOL.iterdir() if p.is_file()}
    inputs = [VENDOR,DORMBR,HEADER,PRODUCER,ROOT/'fastkpc/tools/build_cuda_native.sh',
        PARENT/'manifest.json',PARENT/'analysis.json',PARENT/'canonical/report.json',PARENT/'closure.json',
        ROOT/'fastkpc/docs/PHASE10_OPTIMIZER_L45_ATTRIBUTION_REVIEW_20261029.md']
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in inputs}
    vendor, dorm, header, producer = [p.read_text() for p in (VENDOR,DORMBR,HEADER,PRODUCER)]
    hc, dc, pc = map(compact,(header,dorm,producer))
    for expected in ('return3*n*n+4*n;', 'if(*ispec==1)return32;', 'if(*ispec==2)return2;',
                     'if(lane==0){charvectors=\'Q\';', 'if(lane==0){charvectors=\'P\';'):
        assert expected in hc,expected
    assert dc.count('nb=(*lwork-4160)/ldwork;') == 2
    assert 'constintrows=q+root_rank;' in pc and 'root_rank<=0||root_rank+q>kMaximumRows' in pc
    assert 'm<mnthr' in hc and 'constintegermnthr=n*11/6;' in hc
    assert 'rows,q,decomposition,execute_complete_evaluation,' in pc
    assert 'true,force_stable_svd,true);' in pc
    functions = {name:body(vendor,name) for name in ('dcopy_','dgemm_','dtrmm_','dlarfb_')}
    full, linebase = functions['dlarfb_']
    qa = full.index('/*              W := C1**T */')
    qb = full.index('} else if (lsame_(side, "R"',qa)
    pa = full.index('/*              W := C * V**T  =  (C1*V1**T + C2*V2**T)  (stored in WORK) */')
    pb = full.index('/*           Let  V =  ( V1  V2 )    (V2: last K columns) */',pa)
    inventory = {}
    for side,start,end in [('Q',qa,qb),('P',pa,pb)]:
        section=full[start:end]
        inventory[side]={name:calls(section,name,linebase+full.count('\n',0,start)) for name in ('dcopy_','dgemm_','dtrmm_')}
        assert [len(inventory[side][n]) for n in ('dcopy_','dgemm_','dtrmm_')] == [1,2,3]
    assert [x['args'][:2] for x in inventory['Q']['dgemm_']] == [['"Transpose"','"Notranspose"'],['"Notranspose"','"Transpose"']]
    assert [x['args'][:2] for x in inventory['P']['dgemm_']] == [['"Notranspose"','"Transpose"'],['"Notranspose"','"Notranspose"']]
    expected_trmm={'Q':[['"Right"','"Lower"','"Notranspose"','"Unit"'],['"Right"','"Upper"','transt','"Non-unit"'],['"Right"','"Lower"','"Transpose"','"Unit"']],
                   'P':[['"Right"','"Upper"','"Transpose"','"Unit"'],['"Right"','"Upper"','trans','"Non-unit"'],['"Right"','"Upper"','"Notranspose"','"Unit"']]}
    for side in ('Q','P'):assert [x['args'][:4] for x in inventory[side]['dtrmm_']] == expected_trmm[side]
    assert not any(word in ''.join(x[0] for x in functions.values()) for word in ('__syncthreads','__syncwarp','threadIdx','blockIdx'))
    # The serial T builder is intentionally excluded from vector slicing.
    ft,_=body(vendor,'dlarft_')
    assert len(calls(ft,'dtrmv_')) == 2 and 'prevlastv' in ft
    rows, symbolic, address_count, layouts = shape_inventory()
    out.mkdir(parents=True)
    snapshots=out/'input-source';snapshots.mkdir()
    for p in (VENDOR,DORMBR,HEADER,PRODUCER):shutil.copyfile(p,snapshots/p.name)
    own=out/'tools';own.mkdir()
    for p in TOOL.iterdir():
        if p.is_file():shutil.copyfile(p,own/p.name)
    constants='__device__ integer c__1 = 1;\n__device__ doublereal c_b140 = 1.;\n__device__ doublereal c_b202 = -1.;\n'
    assert all(line in vendor for line in constants.strip().splitlines())
    extracted=constants+'\n\n'.join(x[0] for x in functions.values())+'\n'
    with (out/'symbolic_vendor.inc').open('x') as f:f.write(extracted)
    with (out/'shapes.tsv').open('x') as f:
        f.write('id\tq\tside\tm\tn\tk\tldv\tldc\n')
        for i,s in enumerate(symbolic):f.write('\t'.join(map(str,(i,*s)))+'\n')
    write(out/'inventory.json',dict(source=hashes,functions={n:dict(line=l+1,sha256=hashlib.sha256(b.encode()).hexdigest()) for n,(b,l) in functions.items()},blas_calls=inventory,blocks=rows,layouts=layouts))
    compiler=['g++','-O2','-std=c++17','-Wno-write-strings','-I'+str(out),str(TOOL/'symbolic.cpp'),'-o',str(out/'symbolic')]
    command=[str(out/'symbolic'),str(out/'shapes.tsv')]
    write(out/'protocol.json',dict(schema='exact-postback-static-audit-v1',tools=tools,inputs=hashes,
        native_sha256=parent['native_sha256'],commands=[compiler,command],gpu_execution=False,
        floating_point_evaluation=False,symbolic_cases=len(symbolic)*3,
        branch_patterns=['nonzero/nonunit inputs','sparse zeros','mixed zero/unit diagonal and offdiagonal'],
        status='DECLARED_BEFORE_SYMBOLIC_EXECUTION'))
    with (out/'build.log').open('x') as log:subprocess.run(compiler,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    with (out/'symbolic.json').open('x') as result, (out/'symbolic.log').open('x') as log:
        subprocess.run(command,cwd=ROOT,stdout=result,stderr=log,check=True)
    result=json.loads((out/'symbolic.json').read_text())
    assert result['status']=='PASS_SYMBOLIC_REFERENCE_DLARFB_VECTOR_SLICES'
    assert result['shapes']==len(symbolic) and result['cases']==len(symbolic)*3
    assert all(sha(ROOT/p)==h for p,h in parent['source_sha256'].items())
    assert all(sha(ROOT/p)==h for p,h in hashes.items()) and all(sha(ROOT/p)==h for p,h in tools.items())
    assert sha(native)==parent['native_sha256']
    conclusion=dict(schema='exact-postback-static-audit-result-v1',status='STATIC_VECTOR_SLICE_AUDIT_COMPLETE',
        recommendation='GO_SEPARATE_ISOLATED_DLARFB_VECTOR_SLICE_GPU_QUALIFICATION',
        runtime_integration='NOT_YET_GO',prototype_implemented=False,gpu_qualification='NOT_RUN',
        gpu_execution=False,floating_point_evaluation=False,performance_evidence=False,
        native_sha256=parent['native_sha256'],production_inputs_verified=104,
        augmented_shape_count=37+46,block_address_cases=len(rows),integer_address_checks=address_count,
        symbolic=result,layouts=layouts,extracted_function_bodies_unchanged=True,
        retained_serial=['DLARFT','NB/LWORK/reflector block order','status and WORK[0] metadata'],
        counterexamples=['R/U/N triangular columns overwrite inputs','slicing DORMBR changes L4 NB 2 to 32'],
        limitations=['source-specific symbolic execution, not compiled CUDA proof',
          'symbolic endpoint shapes; integer ownership covers all legal root ranks',
          'no measured split between DLARFT and DLARFB or Q and P',
          'no register/stack/occupancy or new elapsed-time evidence'],
        accepted_opt_in_development_median_sec=311.440,phase10='ACTIVE',candidate='NOT_FROZEN',
        holdout='SEALED',recommended=False,default_enabled=False,l8_optimizer='STOP_UNCHANGED')
    write(out/'audit.json',conclusion)
    write(out/'manifest.json',dict(schema='exact-postback-static-audit-manifest-v1',native_sha256=parent['native_sha256'],
        production_source_sha256=parent['source_sha256'],inputs=hashes,tools=tools,
        files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    print(json.dumps(dict(status=conclusion['status'],symbolic=result,block_address_cases=len(rows),
        integer_address_checks=address_count,audit_sha256=sha(out/'audit.json'),manifest_sha256=sha(out/'manifest.json')),indent=2),flush=True)


if __name__=='__main__':
    main()
