"""Write the sealed postback-audit block inventory as TSV for slice_check.cpp."""
import json
import sys
import hashlib
from pathlib import Path

if len(sys.argv) != 3:
    raise SystemExit('usage: export_blocks.py INVENTORY_JSON OUTPUT_TSV')
data = Path(sys.argv[1]).read_bytes()
if hashlib.sha256(data).hexdigest() != 'e2e1bda96227ac10103f12e712cabcb921bcda6aeb5ff8e3dcfe5a227e73488c':
    raise SystemExit('sealed inventory identity mismatch')
inventory = json.loads(data)
if len(inventory['blocks']) != 1553:
    raise SystemExit('incomplete inventory')
with open(sys.argv[2], 'x') as out:
    for b in inventory['blocks']:
        out.write('\t'.join(str(b[k]) for k in ('q', 'side', 'augmented_rows', 'path', 'offset', 'ib', 'm', 'n', 'ldv', 'ldc')) + '\n')
