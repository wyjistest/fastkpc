"""Query compiled occupancy upper bounds, without launching numerical kernels."""
import ctypes
import json
from pathlib import Path
import sys

out = Path(sys.argv[1]).resolve()
rows = {}
for name in ['baseline_timing', 'candidate_timing']:
    lib = ctypes.CDLL(str(out/(name+'.so')), mode=ctypes.RTLD_LOCAL)
    fn = lib.wp_resources
    fn.argtypes = [ctypes.POINTER(ctypes.c_int)]
    fn.restype = None
    values = (ctypes.c_int*5)()
    fn(values)
    rows[name] = dict(zip(['occupancy_api_blocks','registers','local_bytes','static_shared','dynamic_shared'], values))
with (out/'resources.json').open('x') as f:
    json.dump(rows,f,indent=2)
    f.write('\n')
print(json.dumps(rows),flush=True)
