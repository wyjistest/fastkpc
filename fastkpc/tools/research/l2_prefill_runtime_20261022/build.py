"""Reuse the production-only full-build recipe for the bounded L2 opt-in."""
import hashlib
import json
from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parents[4]
recipe = root / 'fastkpc/tools/research/structured_prefill_canonical_20261022/build.py'
out = Path(sys.argv[1]).resolve()
runpy.run_path(str(recipe), run_name='__main__')
receipt = json.loads((out / 'build.json').read_text())
receipt['schema'] = 'l2-prefill-runtime-full-build-v1'
receipt['build_driver_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
receipt['reused_build_recipe'] = str(recipe.relative_to(root))
receipt['reused_build_recipe_sha256'] = hashlib.sha256(recipe.read_bytes()).hexdigest()
(out / 'build.json').write_text(json.dumps(receipt, indent=2) + '\n')
