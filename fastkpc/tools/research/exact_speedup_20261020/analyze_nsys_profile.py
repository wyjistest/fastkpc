"""Summarize an nsys SQLite export of a full-CUDA skeleton run: kernel
concurrency, multi-penalty optimizer block concurrency, host-side waits inside
the optimizer phases, and the legacy full-eig phases.

Profile the retained eig configuration (component cache, residual bypass,
8 eig lanes); run_subset_profile.R sets it by default.  Without it every
component batch builds a temporary single-lane eig context.

  nsys profile --trace=cuda,osrt --sample=none --cpuctxsw=none -o prof \
      Rscript fastkpc/tools/research/exact_speedup_20261020/run_subset_profile.R
  nsys export --type=sqlite -o prof.sqlite prof.nsys-rep
  python3 -I analyze_nsys_profile.py prof.sqlite
"""
import bisect
import collections
import sqlite3
import sys

OPTIMIZER_KERNEL = "optimize_multi_penalty_targets_kernel"
EIG_MARKER_KERNEL = "sytrd4_cta"
EIG_KERNELS = {"sytrd4_gpu", "sytrd4_cta", "laed2_par", "laed4_par",
               "laed3_epi1", "laed3_epi2", "steqr_ker", "larft_T_64_batch"}
BATCH_MARKER_KERNEL = "build_distance_rows_batch_kernel"
NORM_KERNEL = "lansy_M_stage2"
LARGE_GRID = 128
PHASE_GAP_NS = 200e6


def union_length(intervals):
    total, cur_start, cur_end = 0, None, None
    for start, end in sorted(intervals):
        if cur_start is None:
            cur_start, cur_end = start, end
        elif start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
    if cur_start is not None:
        total += cur_end - cur_start
    return total


def phases_of(intervals):
    intervals = sorted(intervals)
    phases = []
    start, end = intervals[0]
    for s, e in intervals[1:]:
        if s - end > PHASE_GAP_NS:
            phases.append((start, end))
            start, end = s, e
        else:
            end = max(end, e)
    phases.append((start, end))
    return phases


def overlap(start, end, phases):
    return sum(max(0, min(end, b) - max(start, a)) for a, b in phases)


def main(path):
    con = sqlite3.connect(path)
    cur = con.cursor()
    names = dict(cur.execute("select id, value from StringIds"))
    kernels = cur.execute(
        "select start, end, shortName, gridX * gridY * gridZ, streamId "
        "from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
    by_name = collections.defaultdict(list)
    for start, end, short, grid, stream in kernels:
        by_name[names[short]].append((start, end, grid, stream))
    span = max(k[1] for k in kernels) - min(k[0] for k in kernels)
    print(f"kernels: {len(kernels)} over {span / 1e9:.1f} s")
    print(f"{'kernel':48s} {'launches':>9s} {'sum s':>8s} {'union s':>8s} "
          f"{'concurrency':>11s} {'avg grid':>9s}")
    rows = []
    for name, items in by_name.items():
        total = sum(e - s for s, e, _, _ in items)
        rows.append((total, name, items))
    for total, name, items in sorted(rows, reverse=True)[:14]:
        un = union_length([(s, e) for s, e, _, _ in items])
        grid = sum(g for _, _, g, _ in items) / len(items)
        print(f"{name[:48]:48s} {len(items):9d} {total / 1e9:8.2f} "
              f"{un / 1e9:8.2f} {total / max(un, 1):11.2f} {grid:9.1f}")

    mem_kind = dict(cur.execute("select id, label from ENUM_CUDA_MEM_KIND"))
    copy_kind = dict(cur.execute("select id, label from ENUM_CUDA_MEMCPY_OPER"))
    copies = cur.execute(
        "select start, end, copyKind, srcKind, dstKind, bytes "
        "from CUPTI_ACTIVITY_KIND_MEMCPY").fetchall()
    runtime = cur.execute(
        "select start, end, nameId, globalTid "
        "from CUPTI_ACTIVITY_KIND_RUNTIME").fetchall()
    osrt = cur.execute("select start, end, nameId, globalTid from OSRT_API").fetchall()

    counts, volume = collections.Counter(), collections.Counter()
    for _, _, ck, sk, dk, nbytes in copies:
        key = (copy_kind[ck], mem_kind[sk], mem_kind[dk])
        counts[key] += 1
        volume[key] += nbytes
    print("\nhost/device copies (whole run):")
    for key, value in counts.most_common():
        print(f"  {key[0]:16s} {key[1]:>8s} -> {key[2]:8s} {value:8d} "
              f"{volume[key] / 1e6:10.1f} MB")

    merged_kernels = []
    for start, end in sorted((k[0], k[1]) for k in kernels):
        if merged_kernels and start <= merged_kernels[-1][1]:
            merged_kernels[-1][1] = max(merged_kernels[-1][1], end)
        else:
            merged_kernels.append([start, end])
    merged_starts = [s for s, _ in merged_kernels]

    def kernel_overlap(start, end):
        i = max(bisect.bisect_right(merged_starts, start) - 1, 0)
        total = 0
        while i < len(merged_kernels) and merged_kernels[i][0] < end:
            total += max(0, min(end, merged_kernels[i][1]) -
                         max(start, merged_kernels[i][0]))
            i += 1
        return total

    def report_phase(label, phases, show_waits, show_copy_overlap=False):
        length = sum(e - s for s, e in phases)
        print(f"\n{label}: {len(phases)} windows (marker kernels merged across "
              f"gaps < {PHASE_GAP_NS / 1e6:.0f} ms), {length / 1e9:.2f} s")
        api, api_count, threads = collections.Counter(), collections.Counter(), set()
        for s, e, nid, tid in runtime:
            d = overlap(s, e, phases)
            if d > 0:
                api[names[nid]] += d
                api_count[names[nid]] += 1
                threads.add(tid)
        print(f"  host threads issuing CUDA calls: {len(threads)}")
        for name, value in api.most_common(6):
            print(f"  {name[:44]:44s} {api_count[name]:9d} calls {value / 1e9:9.2f} s")
        if show_copy_overlap:
            # Host time in copies while some kernel executes is mostly waiting
            # for submitted work, not transfer overhead.
            memcpy_ids = {k for k, v in names.items() if v.startswith("cudaMemcpyAsync")}
            copy_ns = overlapping_ns = 0
            for s, e, nid, _ in runtime:
                if nid not in memcpy_ids:
                    continue
                for a, b in phases:
                    cs, ce = max(s, a), min(e, b)
                    if ce > cs:
                        copy_ns += ce - cs
                        overlapping_ns += kernel_overlap(cs, ce)
            print(f"  cudaMemcpyAsync host time {copy_ns / 1e9:.3f} s, of which "
                  f"{overlapping_ns / 1e9:.3f} s overlaps kernel execution and "
                  f"{(copy_ns - overlapping_ns) / 1e9:.3f} s has no kernel running")
        phase_copies = collections.Counter()
        for s, e, ck, sk, dk, _ in copies:
            if overlap(s, e, phases) > 0:
                phase_copies[(copy_kind[ck], mem_kind[sk], mem_kind[dk])] += 1
        for key, value in phase_copies.most_common():
            print(f"  copy {key[0]:16s} {key[1]:>8s} -> {key[2]:8s} {value:8d}")
        if show_waits:
            memcpy_ids = {k for k, v in names.items() if v.startswith("cudaMemcpyAsync")}
            per_thread = collections.defaultdict(list)
            for s, e, nid, tid in runtime:
                if nid in memcpy_ids:
                    per_thread[tid].append((s, e))
            for tid in per_thread:
                per_thread[tid].sort()
            mutex_ids = {k for k, v in names.items() if v == "pthread_mutex_lock"}
            inside = outside = 0
            inside_ns = outside_ns = 0
            for s, e, nid, tid in osrt:
                if nid not in mutex_ids or overlap(s, e, phases) == 0:
                    continue
                spans = per_thread.get(tid, [])
                i = bisect.bisect_right(spans, (s, float("inf"))) - 1
                if i >= 0 and spans[i][0] <= s and e <= spans[i][1]:
                    inside += 1
                    inside_ns += e - s
                else:
                    outside += 1
                    outside_ns += e - s
            print(f"  pthread_mutex_lock inside cudaMemcpyAsync: {inside} calls "
                  f"{inside_ns / 1e9:.2f} s; outside: {outside} calls {outside_ns / 1e9:.2f} s")

    optimizer = by_name.get(OPTIMIZER_KERNEL, [])
    if optimizer:
        phases = phases_of([(s, e) for s, e, _, _ in optimizer])
        length = sum(e - s for s, e in phases)
        resident = sum((e - s) * g for s, e, g, _ in optimizer) / length
        durations = sorted(e - s for s, e, _, _ in optimizer)

        def q(f):
            return durations[min(len(durations) - 1, int(f * len(durations)))] / 1e6

        print(f"\noptimizer kernel: {len(optimizer)} launches; launched-block "
              f"concurrency (sum of duration x grid / phase span, an upper bound "
              f"on resident blocks) {resident:.1f}; duration ms p50 {q(.5):.1f} "
              f"p90 {q(.9):.1f} p99 {q(.99):.1f} max {durations[-1] / 1e6:.1f}")
        report_phase("multi-penalty optimizer phases", phases, show_waits=True)

    eig = by_name.get(EIG_MARKER_KERNEL, [])
    if eig:
        streams = collections.Counter(st for name in EIG_KERNELS
                                      for _, _, _, st in by_name.get(name, []))
        print(f"\nfull-eig kernels use {len(streams)} distinct streams; busiest 12: "
              f"{dict(streams.most_common(12))}")
        phases = phases_of([(s, e) for s, e, _, _ in eig])
        length = sum(e - s for s, e in phases)

        def clipped(items):
            return [(max(s, a), min(e, b)) for s, e in items
                    for a, b in phases if e > a and s < b]

        any_kernel = union_length(clipped([(k[0], k[1]) for k in kernels]))
        large = union_length(clipped([(k[0], k[1]) for k in kernels
                                      if k[3] >= LARGE_GRID]))
        batches = len(by_name.get(BATCH_MARKER_KERNEL, []))
        print(f"full-eig phases: any kernel running {100 * any_kernel / length:.0f}% "
              f"of the time, a kernel with >= {LARGE_GRID} blocks running "
              f"{100 * large / length:.0f}%; syevd calls {len(eig)}, distance-row "
              f"batches {batches} ({len(eig) / max(batches, 1):.2f} syevd per batch)")
        # cuSOLVER reads the matrix norm back to the host inside every syevd:
        # a device-to-host copy right after the norm kernel on the same stream.
        per_stream = collections.defaultdict(list)
        for start, _, short, _, stream in kernels:
            per_stream[stream].append((start, names[short]))
        for stream in per_stream:
            per_stream[stream].sort()
        starts = {st: [s for s, _ in v] for st, v in per_stream.items()}
        copy_streams = cur.execute(
            "select start, copyKind, streamId from CUPTI_ACTIVITY_KIND_MEMCPY").fetchall()
        after_norm = 0
        for start, ck, stream in copy_streams:
            if copy_kind[ck] != "Device-to-Host" or stream not in starts:
                continue
            i = bisect.bisect_left(starts[stream], start)
            if i > 0 and per_stream[stream][i - 1][1] == NORM_KERNEL:
                after_norm += 1
        print(f"device-to-host copies directly after {NORM_KERNEL}: {after_norm}")
        report_phase("full-eig phases", phases, show_waits=False,
                     show_copy_overlap=True)


if __name__ == "__main__":
    main(sys.argv[1])
