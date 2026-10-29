"""Compiled resource/occupancy bounds only, without numerical work."""
import ctypes
import sys
from pathlib import Path
from build import write

out = Path(sys.argv[1]).resolve()
rows = {}
for tile in [36,16,8]:
    lib = ctypes.CDLL(str(out/f'tile{tile}_timing.so'),mode=ctypes.RTLD_LOCAL)
    fn = lib.rt_resources
    fn.argtypes=[ctypes.POINTER(ctypes.c_int)]
    fn.restype=None
    values=(ctypes.c_int*5)()
    fn(values)
    rows[str(tile)]=dict(zip(['occupancy_api_blocks','registers','local_bytes','static_shared','dynamic_shared'],values))
write(out/'resources.json',rows)
print(rows,flush=True)
