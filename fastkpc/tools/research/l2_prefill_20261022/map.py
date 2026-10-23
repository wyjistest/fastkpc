"""Restore directed physical identities from the actual consumer capture."""
import csv
import json
from pathlib import Path
import sys

out = Path(sys.argv[1]).resolve()
identity = json.loads((out / 'keys.json').read_text())['identity']
seen_sequences = set()
seen_tasks = set()
with (out / 'capture/physical-map.tsv').open('w') as stream:
    writer = csv.writer(stream, delimiter='\t')
    writer.writerow(['x', 'y', 'S_key', 'sequence', 'consumer', 'pair'])
    with (out / 'capture/raw/groups.jsonl').open() as groups:
        for group_number, line in enumerate(groups):
            group = json.loads(line)
            for pair_number, pair in enumerate(group['pairs']):
                sequence, left, right, _ = pair
                a = identity[group['keys'][left]]
                b = identity[group['keys'][right]]
                assert a['S'] == b['S']
                key = (a['target'], b['target'], a['S'])
                assert sequence not in seen_sequences and key not in seen_tasks
                seen_sequences.add(sequence)
                seen_tasks.add(key)
                writer.writerow([*key, sequence, group_number, pair_number])
assert len(seen_tasks) == 125293
print('PASS directed physical mapping', len(seen_tasks), flush=True)
