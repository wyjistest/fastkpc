"""Run the declared six fresh-process L8 timings, retaining every attempt.

Pause between runs if GPU0 becomes free, so the original isolated cache pair
can take priority. Resume only by skipping successful immutable timing results.
"""
import json
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[4];out=Path(sys.argv[1]).resolve()
read=lambda p:json.loads(Path(p).read_text())
assert read(out/'trace-audit.json')['status']=='PASS_UNCHANGED_L8_DEVICE_WORK'
for name,mode in [('off1','0'),('on1','1'),('on2','1'),('off2','0'),('off3','0'),('on3','1')]:
    if (out/(name+'-process.json')).exists():
        p=read(out/(name+'-process.json'));r=read(out/name/'report.json');h=read(out/name/'hook.json')
        assert p['exit_code']==0 and not p['unexpected_gpu_pids'] and all(r['gates'].values()) and all(h['gates'].values())
        print('KEEP completed',name,r['elapsed_sec'],flush=True);continue
    other=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    cache_completion=root/'fastkpc/artifacts/full_cuda_ci/device_properties_cache_20261025_isolated_completion/closure.json'
    if not other and not cache_completion.exists():
        print('PAUSE_GPU0_FREE: prioritize the pending isolated device-properties continuation',flush=True)
        sys.exit(75)
    subprocess.run([sys.executable,str(root/'fastkpc/tools/research/late_optimizer_20261025/run.py'),str(out),name,mode,'timing'],check=True)
print('PASS six declared timing runs',flush=True)
