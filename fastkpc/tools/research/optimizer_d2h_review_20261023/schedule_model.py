"""Idealized schedule model from per-setup service times measured in the A profile.
service = own last kernel end - own H2D device start (+ measured D2H device time).
Checks the model against measured A (pairs k, k+32 share a hardware queue within a
64-setup window), then estimates refill with 32 workers on 32 distinct queues and no
D2H serialization. These are idealized times, not predictions."""
import csv, heapq, sys
rows = list(csv.DictReader(open(sys.argv[1])))
s = [float(r['last_kernel_end_ms']) - float(r['h2d_start_ms']) + float(r['d2h_device_duration_ms']) for r in rows]
assert len(s) == 128
def window_hol(ws):  # 64 setups, setup k and k+32 serialized on one queue
    return max(ws[c] + ws[c + 32] for c in range(32))
def rolling(xs, w):
    free = [0.0] * w; heapq.heapify(free); end = 0.0
    for x in xs:
        t = heapq.heappop(free) + x; end = max(end, t); heapq.heappush(free, t)
    return end
a_model = window_hol(s[:64]) + window_hol(s[64:])
print('sum service %.0f ms  max %.0f ms  sum/32 %.0f ms' % (sum(s), max(s), sum(s) / 32))
print('A model (paired queues, barrier)      %.0f ms' % a_model)
print('rolling 32 queues, original order     %.0f ms' % rolling(s, 32))
print('rolling 32 queues, longest first      %.0f ms' % rolling(sorted(s, reverse=True), 32))
print('A barrier but 32 queues, no pairing   %.0f ms' % (rolling(s[:64], 32) + rolling(s[64:], 32)))
