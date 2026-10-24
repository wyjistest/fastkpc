"""Free-memory decrement per created prepared vs its application-visible bytes."""
import collections, csv, sys
MiB = 1 << 20
rows = list(csv.DictReader(open(sys.argv[1])))
c = collections.Counter()
for r in rows:
    d = int(r['free_before']) - int(r['free_after'])
    c[(int(r['targets']), round(int(r['application_bytes']) / MiB, 2), round(d / MiB, 2))] += 1
for (t, a, d), n in sorted(c.items(), key=lambda x: -x[1]):
    print('targets %d  application %.2f MiB  free decrement %.2f MiB  setups %d' % (t, a, d, n))
