"""Same-channel serialization of persistent optimizer kernels in window-mode levels.

python3 -I same_channel.py RUN_DIR [RUN_DIR ...]

RUN_DIR is a ktrace directory (kernels.bin, runtime.bin, kernel-names.tsv,
profile-result.json). Window-mode streams are identified as in predict.py. Each setup's
longest kernel is its persistent optimizer kernel. A persistent kernel is "late" when
it starts more than 50 ms after its launch API call. It "starts at a same-channel end"
when another stream of the same window, on the same hardware channel, ends its
persistent kernel within 2 ms of that start (a lower bound: short tail kernels between
two links push some starts past 2 ms). For L6-L8 windows the script also reports the
critical (longest) kernel, the streams sharing its channel, and the serial chain that
sets the window span: walking back from the last-ending persistent kernel through
same-channel predecessors that ended at most 150 ms before each link started.

For every level (window and slot mode) it also counts persistent-kernel starts while a
different stream's persistent kernel is running on the same channel, and for slot levels
how many channels each slot stream used. CUPTI reports the channel per launch: identical
stream ids map to different channels across runs, and streams change channels over time.

Observation only: a matching end time is consistent with FIFO false dependencies on a
shared channel (44 or 64 streams on CUDA_DEVICE_MAX_CONNECTIONS=32), not a proof; zero
overlaps in a slot trace is an observation for that run, not a guarantee.
Added for the L6 runtime final review (2026-10-27).
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

TOL_NS = 2_000_000
CHAIN_NS = 150_000_000
LATE_S = 0.05
SHIFTS_NS = [int(x * 1e9) for x in (-0.5, -0.3, -0.2, 0.2, 0.3, 0.5)]


def analyze(run):
    K = np.fromfile(run / 'kernels.bin', dtype=np.dtype([
        ('start', '<u8'), ('end', '<u8'), ('lmem_total', '<u8'), ('stream', '<u4'),
        ('corr', '<u4'), ('name', '<u4'), ('lmem', '<u4'), ('channel', '<u4'),
        ('grid', '<u4'), ('block', '<u4'), ('regs', '<u2')]))
    RT = np.fromfile(run / 'runtime.bin', dtype=np.dtype([
        ('start', '<u8'), ('end', '<u8'), ('corr', '<u4'), ('cbid', '<u4'), ('thread', '<u4')]))
    names = dict((int(a), b) for a, b in (l.split('\t', 1) for l in
                 (run / 'kernel-names.tsv').read_text().splitlines()))
    result = json.loads((run / 'profile-result.json').read_text())
    order = np.argsort(RT['corr'])
    rc, rth, rs = RT['corr'][order], RT['thread'][order], RT['start'][order]
    pos = np.clip(np.searchsorted(rc, K['corr']), 0, len(rc) - 1)
    matched = rc[pos] == K['corr']
    k_thread = np.where(matched, rth[pos], 0)
    k_launch = np.where(matched, rs[pos], 0).astype(np.int64)
    threads, counts = np.unique(RT['thread'], return_counts=True)
    main_thread = int(threads[np.argmax(counts)])
    multi = np.array(['mgcv_multi_penalty_gcv' in names[i] for i in range(len(names))])
    persistent = np.array(['optimize_multi_penalty_targets' in names[i] for i in range(len(names))])
    idx = np.nonzero(multi[K['name']] & (k_thread != main_thread))[0]
    idx = idx[np.argsort(K['start'][idx], kind='stable')]
    by_stream = collections.defaultdict(list)
    for i in idx:
        by_stream[int(K['stream'][i])].append(int(i))
    streams = sorted(by_stream, key=lambda s: K['start'][by_stream[s][0]])
    windows = [b for b in result['prefill_batches'] if b['penalty_class'] == 'multi']
    cursor, per_level = 0, collections.defaultdict(lambda: [0, 0, 0])
    excess = collections.defaultdict(lambda: [0, 0.0, 0.0, 0])
    other = collections.defaultdict(lambda: [0, 0, 0, 0])
    sharing = collections.defaultdict(lambda: [0, 0])
    print(run)
    for w in windows:
        n = w['optimizer_setup_count']
        width = min(32, n) if n > 64 else n
        seg = streams[cursor:cursor + width]
        cursor += width
        # Channel sharing among all persistent kernels of the window's streams: a start
        # while a different stream's persistent kernel runs on the same channel. CUPTI
        # reports the channel per launch; a stream may use several channels over time.
        launches = sorted((i for s in seg for i in by_stream[s] if persistent[K['name'][i]]),
                          key=lambda i: K['start'][i])
        running, overlaps = collections.defaultdict(list), 0
        for i in launches:
            ch, t = int(K['channel'][i]), int(K['start'][i])
            running[ch] = [j for j in running[ch] if int(K['end'][j]) > t]
            overlaps += any(int(K['stream'][j]) != int(K['stream'][i]) for j in running[ch])
            running[ch].append(i)
        lifetime = collections.Counter(len({int(K['channel'][i]) for i in by_stream[s] if persistent[K['name'][i]]})
                                       for s in seg)
        sharing[w.get('level')][0] += len(launches)
        sharing[w.get('level')][1] += overlaps
        if n > 64:
            print(f"  L{w.get('level')} slot mode: {len(seg)} streams, {len(launches)} persistent kernels, "
                  f"channels used per stream over the level {min(lifetime)}-{max(lifetime)}, "
                  f"starts while another stream's persistent kernel runs on the same channel {overlaps}")
            continue  # slot mode: one stream per slot, refilled
        P = {}
        for s in seg:
            ks = by_stream[s]
            d = K['end'][ks] - K['start'][ks]
            j = ks[int(np.argmax(d))]
            P[s] = dict(start=int(K['start'][j]), end=int(K['end'][j]), launch=int(k_launch[j]),
                        channel=int(K['channel'][j]), dur=(int(K['end'][j]) - int(K['start'][j])) / 1e9)
        late = [s for s, p in P.items() if (p['start'] - p['launch']) / 1e9 > LATE_S]

        def ends_near(s, t):
            near = [o for o, q in P.items() if o != s and abs(q['end'] - t) <= TOL_NS]
            same = any(P[o]['channel'] == P[s]['channel'] for o in near)
            return same, bool(near) and not same

        at_end = [s for s in late if ends_near(s, P[s]['start'])[0]]
        level = w.get('level')
        acc = per_level[level]
        acc[0] += len(P); acc[1] += len(late); acc[2] += len(at_end)
        # Chance baseline: the same test at shifted times inside the window.
        lo = min(p['start'] for p in P.values())
        hi = max(p['end'] for p in P.values())
        for s in late:
            other[level][0] += ends_near(s, P[s]['start'])[1]
            for off in SHIFTS_NS:
                t = P[s]['start'] + off
                if lo <= t <= hi:
                    same, oth = ends_near(s, t)
                    other[level][1] += 1; other[level][2] += same; other[level][3] += oth
        t0 = min(p['start'] for p in P.values())
        span = (max(p['end'] for p in P.values()) - t0) / 1e9
        crit = max(P, key=lambda s: P[s]['dur'])
        c = P[crit]
        sharers = sorted((q['start'], o) for o, q in P.items() if o != crit and q['channel'] == c['channel'])
        # Walk back from the kernel that ends last (the span setter): its predecessor is
        # the same-channel persistent kernel that ended at most CHAIN_NS before it started
        # (short per-stream tail kernels sit between links).
        setter = max(P, key=lambda s: P[s]['end'])
        chain, s = [setter], setter
        while True:
            cands = [o for o, q in P.items() if o not in chain and q['channel'] == P[s]['channel']
                     and 0 <= P[s]['start'] - q['end'] <= CHAIN_NS]
            if not cands:
                break
            s = max(cands, key=lambda o: P[o]['end'])
            chain.append(s)
        chain.reverse()
        gaps = [round((P[chain[i + 1]]['start'] - P[chain[i]]['end']) / 1e6, 1) for i in range(len(chain) - 1)]
        # Span not covered by the chain's busy time: chain-head wait after its window's
        # first start plus link gaps. Small values mean the chain accounts for the span.
        uncovered = (P[chain[0]]['start'] - t0) / 1e9 + sum(gaps) / 1e3
        excess[level][0] += 1
        excess[level][1] += span - c['dur']
        excess[level][2] += uncovered
        excess[level][3] += len(chain) > 1
        if level not in (6, 7, 8):
            continue
        channels = collections.Counter(p['channel'] for p in P.values())
        print(f"  L{level} w{w.get('window_id')}: setups {len(P)}, channels {len(channels)}, "
              f"streams on shared channels {sum(1 for p in P.values() if channels[p['channel']] > 1)}, "
              f"span {span:.3f} s, critical {c['dur']:.3f} s "
              f"(launch->start {(c['start'] - c['launch']) / 1e9:.3f} s, channel {c['channel']}, "
              f"same-channel sharers {[round(P[o]['dur'], 3) for st, o in sharers]}); "
              f"span setter chain on channel {P[setter]['channel']}: {[round(P[o]['dur'], 3) for o in chain]} "
              f"(link gaps ms {gaps}, head launch->start {(P[chain[0]]['start'] - P[chain[0]]['launch']) / 1e9:.3f} s); "
              f"late {len(late)}, late at same-channel end {len(at_end)}")
    for level in sorted(sharing):
        launches, overlaps = sharing[level]
        print(f"  L{level} channel sharing: {overlaps}/{launches} persistent kernels start while another "
              f"stream's persistent kernel runs on the same channel")
    for level in sorted(per_level):
        n, late, at_end = per_level[level]
        windows_n, beyond, uncovered, chained = excess[level]
        other_end, trials, chance_same, chance_other = other[level]
        print(f"  L{level} total: window-mode setups {n}, late persistent kernels {late}, "
              f"of which start at a same-channel end {at_end}, at only an other-channel end {other_end} "
              f"(chance at shifted times: same {100 * chance_same / max(trials, 1):.1f}%, "
              f"other-only {100 * chance_other / max(trials, 1):.1f}% of {trials}); windows {windows_n}, "
              f"sum(span - critical) {beyond:.3f} s, windows whose span-setting chain has >1 kernel "
              f"{chained}, sum(span not covered by that chain) {uncovered:.3f} s")


for arg in sys.argv[1:]:
    analyze(Path(arg))
