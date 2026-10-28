"""Inclusion list of historical canonical L7/L8 optimizer times.

python3 -I history.py OUT_JSON

Scans fastkpc/artifacts/full_cuda_ci/*/*/report.json for reports with per-level optimizer
host times, deduplicates by receipt_sha256 (continuation trees copy receipts), and writes
one row per receipt with every path that carried it. Rows mix natives, configurations,
traced, shared-host and diagnostic runs: use them to show that a phenomenon recurs, not to
estimate the bias or gain of a current median. Added for the L6 runtime final review
(2026-10-27).
"""
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[4]
FLOOR_S = 9.372  # L7 critical persistent kernel in the cache-ON trace
rows = {}
for path in sorted(root.glob('fastkpc/artifacts/full_cuda_ci/*/*/report.json')):
    try:
        report = json.loads(path.read_text())
    except (OSError, ValueError):
        continue
    levels = report.get('stages', {}).get('optimizer_by_level')
    if not isinstance(levels, list):
        continue
    host = {x['level']: x['host_sec'] for x in levels}
    if 7 not in host:
        continue
    run = path.parent
    hook = run / 'hook.json'
    preload = bool(json.loads(hook.read_text()).get('preload')) if hook.exists() else False
    row = rows.setdefault(report['receipt_sha256'], dict(
        receipt_sha256=report['receipt_sha256'], native_sha256=report.get('native_sha256'),
        elapsed_sec=report.get('elapsed_sec'), l7_optimizer_sec=host[7],
        l8_optimizer_sec=host.get(8), l8_preload=preload, paths=[]))
    assert (row['elapsed_sec'], row['l7_optimizer_sec']) == (report.get('elapsed_sec'), host[7])
    row['paths'].append(str(run.relative_to(root)))

rows = sorted(rows.values(), key=lambda r: r['paths'][0])
excess = [r['l7_optimizer_sec'] - FLOOR_S for r in rows]
bins = [(None, 0.1), (0.1, 1.0), (1.0, 1.8), (1.8, 2.5), (2.5, None)]
summary = dict(
    reports=sum(len(r['paths']) for r in rows), unique_receipts=len(rows), l7_floor_sec=FLOOR_S,
    l7_excess_bins=[dict(low=a, high=b, count=sum(1 for e in excess if (a is None or e >= a) and (b is None or e < b)))
                    for a, b in bins],
    l7_excess_max_sec=max(excess), l7_mean_excess_sec=sum(excess) / len(excess),
    l8_preload_runs=sum(r['l8_preload'] for r in rows))
Path(sys.argv[1]).write_text(json.dumps(dict(schema='l7-l8-history-v1', summary=summary, rows=rows), indent=2) + '\n')
print(json.dumps(summary, indent=2))
