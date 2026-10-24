"""Per-level optimizer, stage A/B and level elapsed medians over run report.json files."""
import json, statistics, sys
reports = [json.load(open(p)) for p in sys.argv[1:]]
med = statistics.median
opt = {}
for r in reports:
    for x in r['stages']['optimizer_by_level']: opt.setdefault(x['level'], []).append(x['host_sec'])
for l in range(2, 9):
    a = [x['stage_a_host_ms'] / 1e3 for r in reports for x in r['level_component_prefill'] if x['level'] == l]
    b = [x['stage_b_host_ms'] / 1e3 for r in reports for x in r['level_component_prefill'] if x['level'] == l]
    e = [x['elapsed_ms'] / 1e3 for r in reports for x in r['level_trace'] if x['level'] == l]
    print('L%d optimizer %.3f  stage A %.3f  stage B %.3f  level elapsed %.3f' % (l, med(opt[l]), med(a), med(b), med(e)))
