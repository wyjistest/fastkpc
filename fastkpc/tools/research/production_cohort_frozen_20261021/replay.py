"""Anchor frozen replay to every actual finalizer, then change only cohorts."""
import argparse
import ctypes as ct
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("capture", type=Path)
parser.add_argument("library", type=Path)
parser.add_argument("--memcheck-smoke", action="store_true")
args = parser.parse_args()
capture = args.capture
raw = capture/"raw"
lib = ct.CDLL(str(args.library.resolve()))
pointer = ct.POINTER(ct.c_double)
lib.replay_frozen.argtypes = [ct.c_int]*5+[pointer]*5
lib.replay_frozen.restype = ct.c_int
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def bits(values):
    return np.asarray(values, dtype='<f8').tobytes(order='F')
def identical(a, b):
    return bits(a) == bits(b)
def arrays(path, shapes):
    data = np.fromfile(path.with_suffix('.f64'), dtype='<f8')
    offset, result = 0, []
    for shape in shapes:
        size = int(np.prod(shape))
        result.append(data[offset:offset+size].reshape(shape, order='F'))
        offset += size
    assert offset == data.size
    return result
def read(path):
    return json.loads(path.read_text())
def exact_key(base, logsp):
    payload = ('schema=full-cuda-ci-residual-key-v2\ntarget='+base+
      '\nsolver=full-cuda-ci-mgcv-exact-fixed-evaluator-v1\nselected_log_sp_bits='+
      ''.join(format(int(x), '016x') for x in np.asarray(logsp, dtype='<f8').view('<u8'))+'\n')
    return hashlib.sha256(payload.encode()).hexdigest()
def gpu(X, coefficients, Y, config):
    inputs = [np.asfortranarray(x, dtype=np.float64) for x in (X, coefficients, Y)]
    n, q = X.shape
    count = coefficients.shape[1]
    fitted, residual = [np.empty((n, count), dtype=np.float64, order='F') for _ in range(2)]
    before = [bits(x) for x in inputs]
    status = lib.replay_frozen(n, q, count, config['math'], config['atomics'],
      *[x.ctypes.data_as(pointer) for x in inputs+[fitted, residual]])
    assert status == 0 and before == [bits(x) for x in inputs]
    assert np.isfinite(fitted).all() and np.isfinite(residual).all()
    return fitted, residual

metadata = read(capture/'capture.json')
assert all(metadata['gates'].values())
dataset = np.fromfile(capture/'dataset.f64', dtype='<f8').reshape(
    (metadata['n'], metadata['p']), order='F')
column_identity = {bits(dataset[:, j]): j+1 for j in range(dataset.shape[1])}
sources, by_key = {}, {}
for path in sorted(raw.glob('prefill-*.json')):
    record = read(path)
    n, q, count, penalties = (record[k] for k in ('n','q','targets','penalties'))
    coefficient, Y, logsp = arrays(path, [(q,count),(n,count),(penalties,count)])
    assert record['info'] == [0]*count and record['route'] == [2]*count
    assert all(x > 0 for x in record['rank'])
    keys = [exact_key(record['keys'][j], logsp[:,j]) for j in range(count)]
    targets = [column_identity[bits(Y[:,j])] for j in range(count)]
    assert len(set(keys)) == count and len(set(targets)) == count
    sources[path.stem] = dict(record=record, coefficient=coefficient, Y=Y,
                              logsp=logsp, keys=keys, targets=targets, consumers=[])
    for j,key in enumerate(keys):
        assert key not in by_key
        by_key[key] = (path.stem,j)

anchors = []
references = defaultdict(list)
configurations = set()
smoke_sources = ({max(sources, key=lambda k: len(sources[k]['keys'])),
                  max(sources, key=lambda k: sources[k]['record']['q'])}
                 if args.memcheck_smoke else None)
for ordinal,path in enumerate(sorted(raw.glob('finalize-*.json'))):
    record = read(path)
    n,q,count = (record[k] for k in ('n','q','targets'))
    X,coefficient,Y,fitted,residual = arrays(path, [(n,q),(q,count)]+[(n,count)]*3)
    source_ids = {by_key[key][0] for key in record['keys']}
    assert len(source_ids) == 1
    source_id = source_ids.pop()
    if smoke_sources is not None and source_id not in smoke_sources:
        continue
    source = sources[source_id]
    if 'X' in source:
        assert identical(source['X'],X)
    source['X'], source['config'] = X, record
    source['consumers'].append(path.stem)
    configurations.add((record['math'],record['atomics'],record['cublas_version']))
    for j,key in enumerate(record['keys']):
        pos = by_key[key][1]
        assert identical(coefficient[:,j], source['coefficient'][:,pos])
        assert identical(Y[:,j], source['Y'][:,pos])
        assert record['info'][j] == source['record']['info'][pos] == 0
        assert record['route'][j] == source['record']['route'][pos] == 2
        assert record['rank'][j] == source['record']['rank'][pos]
        references[key].append((path.stem,count,fitted[:,j].copy(),residual[:,j].copy()))
    got_fit,got_res = gpu(X,coefficient,Y,record)
    same = identical(got_fit,fitted) and identical(got_res,residual)
    assert same, ('original-shape GPU anchor failed',path.stem)
    reverse = np.arange(count-1,-1,-1)
    got_fit,got_res = gpu(X,coefficient[:,reverse],Y[:,reverse],record)
    reverse_same = identical(got_fit[:,reverse],fitted) and identical(got_res[:,reverse],residual)
    anchors.append(dict(cohort=path.stem,source=source_id,q=q,width=count,
                        original_exact=same,reverse_exact=reverse_same))
    if (ordinal+1)%200 == 0:
        print('anchored',ordinal+1,'actual finalizer cohorts',flush=True)
assert len(anchors) > 0
if not args.memcheck_smoke:
    assert len(anchors) == metadata['counts']['finalize']

rows, witnesses = [], []
for ordinal,(source_id,source) in enumerate(sources.items()):
    if 'X' not in source:
        continue  # Planned but never consumed: no real finalizer anchor exists.
    count = len(source['keys'])
    natural = np.argsort(source['targets'])
    shapes = [('all_natural',natural),('all_reversed',natural[::-1]),
              ('all_rotated',np.roll(natural,1))]
    # A concrete full-coverage schedule, not a collection of overlapping
    # windows: balanced distinct-target chunks, each between 2 and 16 columns.
    bounded = np.array_split(natural, (count+15)//16)
    assert all(2 <= len(chunk) <= 16 for chunk in bounded)
    assert sorted(np.concatenate(bounded)) == list(range(count))
    for chunk_id,chunk in enumerate(bounded):
        shapes.append((f'bounded16_chunk_{chunk_id}',chunk))
        shapes.append((f'bounded16_chunk_{chunk_id}_repeat',chunk))
    for width in (2,3,5,8,10,16,17,18,19):
        if width > count:
            continue
        # Overlap the last full window, avoiding fake/duplicate padding targets.
        starts = list(range(0,count-width+1,width))
        if starts[-1] != count-width:
            starts.append(count-width)
        for start in starts:
            shapes.append((f'width_{width}_offset_{start}',natural[start:start+width]))
    for shape,selection in shapes:
        fitted,residual = gpu(source['X'], source['coefficient'][:,selection],
                              source['Y'][:,selection], source['config'])
        for j,pos in enumerate(selection):
            key = source['keys'][pos]
            for cohort,original_width,ref_fit,ref_res in references.get(key, []):
                fit_same,res_same = identical(fitted[:,j],ref_fit),identical(residual[:,j],ref_res)
                row = dict(source=source_id,cohort=cohort,target=source['targets'][pos],key=key,
                  q=source['record']['q'],shape=shape,width=len(selection),
                  original_width=original_width,fitted_exact=fit_same,residual_exact=res_same,
                  fitted_max_abs=float(np.max(np.abs(fitted[:,j]-ref_fit))),
                  residual_max_abs=float(np.max(np.abs(residual[:,j]-ref_res))))
                rows.append(row)
                if not res_same and len(witnesses)<20:
                    witnesses.append(row)
    if (ordinal+1)%100 == 0:
        print('regrouped',ordinal+1,'frozen coefficient sources',flush=True)

import csv
prefix = 'smoke-' if args.memcheck_smoke else ''
for name,values in [('anchors.csv',anchors),('comparisons.csv',rows)]:
    with (capture/(prefix+name)).open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(values[0]))
        writer.writeheader();writer.writerows(values)
by_width = {}
for width in sorted({row['width'] for row in rows}):
    selected = [row for row in rows if row['width']==width]
    by_width[str(width)] = dict(comparisons=len(selected),
       fitted_mismatches=sum(not r['fitted_exact'] for r in selected),
       residual_mismatches=sum(not r['residual_exact'] for r in selected),
       max_abs=max(r['residual_max_abs'] for r in selected))
assert {r['key'] for r in rows if r['shape'].startswith('bounded16_')} == set(references)
by_q = {}
for q in sorted({r['q'] for r in anchors}):
    qa = [r for r in anchors if r['q']==q]
    qr = [r for r in rows if r['q']==q]
    by_q[str(q)] = dict(actual_cohorts=len(qa),
      target_occurrences=sum(r['width'] for r in qa),
      consumed_unique_targets=len({r['key'] for r in qr}),
      original_max_width=max(r['width'] for r in qa),
      tested_max_width=max(r['width'] for r in qr),
      bounded16_pass=all(r['fitted_exact'] and r['residual_exact'] for r in qr
                         if r['shape'].startswith('bounded16_')),
      mismatching_unique_targets=len({r['key'] for r in qr if not r['residual_exact']}))
failed = [r for r in rows if not r['residual_exact'] or not r['fitted_exact']]
summary = dict(schema='production-cohort-frozen-replay-v1',
  memcheck_smoke=args.memcheck_smoke,
  native_sha256=metadata['native_sha256'],capture_sha256=sha(capture/'capture.json'),
  replay_library_sha256=sha(args.library),replay_script_sha256=sha(Path(__file__)),
  actual_cohorts=len(anchors),actual_target_occurrences=sum(r['width'] for r in anchors),
  original_widths=dict(sorted(Counter(r['width'] for r in anchors).items())),
  source_cohorts=len(sources),source_targets=sum(len(s['keys']) for s in sources.values()),
  consumed_sources=sum('X' in s for s in sources.values()),
  consumed_unique_targets=len(references),
  original_shape_all_exact=all(r['original_exact'] for r in anchors),
  original_reverse_all_exact=all(r['reverse_exact'] for r in anchors),
  coefficient_metadata_identity=True,configurations=sorted(configurations),
  by_width=by_width,witnesses=witnesses,
  by_q=by_q,comparison_count=len(rows),mismatch_comparisons=len(failed),
  mismatching_unique_targets=len({r['key'] for r in failed}),
  mismatching_sources=len({r['source'] for r in failed}),
  mismatching_original_cohorts=len({r['cohort'] for r in failed}),
  max_fitted_abs=max(r['fitted_max_abs'] for r in rows),
  max_residual_abs=max(r['residual_max_abs'] for r in rows),
  regrouping_bitwise_pass=all(r['residual_exact'] and r['fitted_exact'] for r in rows),
  bounded16_bitwise_pass=all(r['residual_exact'] and r['fitted_exact'] for r in rows
                            if r['shape'].startswith('bounded16_')),
  bounded16_unique_targets=len({r['key'] for r in rows if r['shape'].startswith('bounded16_')}),
  bounded16_comparisons=sum(r['shape'].startswith('bounded16_') for r in rows),
  bounded16_schedule_calls=sum((len(s['keys'])+15)//16 for s in sources.values() if 'X' in s),
  production_integration='NOT_YET_GO',optimizer_reexecuted_during_replay=False,
  q_values=sorted({r['q'] for r in anchors}),
  scope='actual 22-column multi-penalty finalizer cohorts; no single-penalty claim')
(capture/(prefix+'replay.json')).write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='witnesses'},indent=2))
