"""Per-kernel launch shape and resources in an Nsight sqlite (grouped by shape)."""
import sqlite3, sys
db = sqlite3.connect('file:' + sys.argv[1] + '?mode=ro', uri=True)
S = dict(db.execute('select id, value from StringIds'))
q = '''select shortName, gridX*gridY*gridZ, blockX*blockY*blockZ, registersPerThread, staticSharedMemory,
       dynamicSharedMemory, localMemoryPerThread, count(*), sum(end-start), max(end-start)
       from CUPTI_ACTIVITY_KIND_KERNEL group by 1,2,3,4,5,6,7 order by sum(end-start) desc limit 12'''
for n, g, b, reg, ss, ds, lm, c, tot, mx in db.execute(q):
    print('%-40s grid %3d block %4d regs %3d smem %6d+%d local %d n=%4d total %.1f s max %.3f s'
          % (S[n][:40], g, b, reg, ss, ds, lm, c, tot / 1e9, mx / 1e9))
print('GPU:', db.execute('select name, smCount, maxRegistersPerBlock, maxWarpsPerSm from TARGET_INFO_GPU').fetchone())
