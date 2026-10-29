"""Compare the embedded cubins of two natives by SHA-256 (host-only review evidence).

python3 -I cubin_compare.py OLD_NATIVE OLD_SHA256 NEW_NATIVE NEW_SHA256 EXPECTED_NEW_ONLY OUTPUT_JSON

Fails (exit 1) on a native identity mismatch, an empty extraction, any old cubin missing
from the new native, or a new-only count different from EXPECTED_NEW_ONLY. Refuses to
overwrite OUTPUT_JSON. Byte-identical cubins show that reused objects were linked
unchanged; they do not show what a clean rebuild from current sources would produce.
"""
import collections
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

CUOBJDUMP = '/usr/local/cuda-12.5/bin/cuobjdump'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def extract(native, directory):
    subprocess.run([CUOBJDUMP, '-xelf', 'all', str(native)], cwd=directory, check=True, capture_output=True)
    files = sorted(Path(directory).glob('*.cubin'))
    if not files:
        raise SystemExit('no cubins extracted from ' + str(native))
    return {f.name: dict(sha256=sha(f), bytes=f.stat().st_size) for f in files}


def resources(cubin):
    text = subprocess.run([CUOBJDUMP, '-res-usage', str(cubin)], check=True, capture_output=True, text=True).stdout
    lines, rows = text.splitlines(), []
    for i, line in enumerate(lines):
        m = re.match(r'\s*Function (\S+):', line)
        if m and i + 1 < len(lines):
            reg = re.search(r'REG:(\d+)', lines[i + 1]); stack = re.search(r'STACK:(\d+)', lines[i + 1])
            rows.append(dict(function=m.group(1), registers=int(reg.group(1)), stack=int(stack.group(1))))
    return rows


def main():
    if len(sys.argv) != 7:
        raise SystemExit(__doc__)
    old, old_sha, new, new_sha, expected, output = sys.argv[1:]
    old, new = str(Path(old).resolve()), str(Path(new).resolve())
    for path, value in ((old, old_sha), (new, new_sha)):
        if sha(path) != value:
            raise SystemExit('native identity mismatch: ' + path)
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        before, after = extract(old, a), extract(new, b)
        old_hashes = collections.Counter(v['sha256'] for v in before.values())
        new_hashes = collections.Counter(v['sha256'] for v in after.values())
        old_only, new_only = old_hashes - new_hashes, new_hashes - old_hashes
        added = [dict(name=n, **v, functions=resources(Path(b) / n)) for n, v in after.items() if v['sha256'] in new_only]
    version = subprocess.run([CUOBJDUMP, '--version'], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    status = 'PASS' if not old_only and sum(new_only.values()) == int(expected) else 'FAIL'
    result = dict(status=status, cuobjdump=version, old_native=dict(path=old, sha256=old_sha, cubins=before),
                  new_native=dict(path=new, sha256=new_sha, cubins=after),
                  common=sum((old_hashes & new_hashes).values()), old_only=sum(old_only.values()),
                  new_only=sum(new_only.values()), added=added,
                  scope='installed-binary identity only; not clean-rebuild equivalence')
    with open(output, 'x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(status, 'common', result['common'], 'old_only', result['old_only'], 'new_only', result['new_only'])
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
