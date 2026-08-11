#!/usr/bin/env python3

import argparse
import re
import subprocess
import tempfile
from pathlib import Path


FUNCTION_START = re.compile(
    r"^(?:/\* Subroutine \*/ int|integer|doublereal|logical) "
    r"([a-z][a-z0-9]*_)\("
)


def make_f2c_compatible(text):
    """Lower the small amount of structured DO syntax rejected by f2c."""
    output = []
    loops = []
    next_label = 90000
    for line in text.splitlines():
        do_match = re.match(
            r"^\s+DO\s+([A-Z][A-Z0-9]*)\s*=\s*(.+)$", line
        )
        if do_match is not None:
            continue_label = next_label
            exit_label = next_label + 1
            next_label += 2
            loops.append((continue_label, exit_label))
            output.append(
                f"      DO {continue_label} {do_match.group(1)} = "
                f"{do_match.group(2)}"
            )
            continue
        if re.match(r"^\s+END\s*DO\s*$", line):
            if not loops:
                raise ValueError("unmatched END DO in Fortran input")
            continue_label, exit_label = loops.pop()
            output.append(f"{continue_label:5d} CONTINUE")
            output.append(f"{exit_label:5d} CONTINUE")
            continue
        exit_match = re.match(r"^(\s*)IF\s*\((.*)\)\s*EXIT\s*$", line)
        if exit_match is not None:
            if not loops:
                raise ValueError("EXIT outside a structured DO")
            output.append(
                f"{exit_match.group(1)}IF( {exit_match.group(2)} ) "
                f"GO TO {loops[-1][1]}"
            )
            continue
        output.append(line)
    if loops:
        raise ValueError("unterminated structured DO in Fortran input")
    return "\n".join(output) + "\n"


def collect_prototypes(lines):
    prototypes = []
    index = 0
    while index < len(lines):
        match = FUNCTION_START.match(lines[index])
        if match is None:
            index += 1
            continue
        signature = [lines[index]]
        while not signature[-1].rstrip().endswith(")"):
            index += 1
            signature.append(lines[index])
        text = "\n".join(signature)
        text = re.sub(r"^/\* Subroutine \*/ int", "__device__ int", text)
        text = re.sub(
            r"^(integer|doublereal|logical)", r"__device__ \1", text
        )
        prototypes.append(text + ";")
        index += 1
    return prototypes


def transform(lines):
    output = []
    dropping_extern = False
    dropping_builtins = False
    for line in lines:
        if line == '#include "f2c.h"':
            continue
        if dropping_extern:
            if ";" in line:
                dropping_extern = False
            continue
        if dropping_builtins:
            if not line.strip():
                dropping_builtins = False
            continue
        if "/* Builtin functions */" in line:
            dropping_builtins = True
            continue
        if re.match(r"\s+extern\b", line):
            if ";" not in line:
                dropping_extern = True
            continue
        if re.match(r"^static (integer|doublereal|logical) c", line):
            line = re.sub(r"^static ", "__device__ ", line)
        elif re.match(r"\s+static ", line):
            line = re.sub(r"^(\s+)static ", r"\1", line)
        line = re.sub(
            r"^/\* Subroutine \*/ int ([a-z][a-z0-9]*_)\(",
            r"__device__ int \1(",
            line,
        )
        line = re.sub(
            r"^(integer|doublereal|logical) ([a-z][a-z0-9]*_)\(",
            r"__device__ \1 \2(",
            line,
        )
        output.append(line)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--append-input", action="append", type=Path, default=[]
    )
    parser.add_argument(
        "--f2c", type=Path,
        help="translate the ordered Fortran inputs before transformation",
    )
    args = parser.parse_args()

    inputs = [args.input, *args.append_input]
    if args.f2c is None:
        lines = []
        for input_path in inputs:
            lines.extend(input_path.read_text().splitlines())
    else:
        with tempfile.TemporaryDirectory(prefix="fastkpc-lapack-f2c-") as tmp:
            temporary = Path(tmp)
            ordered = temporary / "ordered.f"
            ordered.write_text(make_f2c_compatible(
                "\n".join(path.read_text() for path in inputs)
            ))
            subprocess.run(
                [str(args.f2c), ordered.name], cwd=temporary, check=True
            )
            lines = (temporary / "ordered.c").read_text().splitlines()
    prototypes = collect_prototypes(lines)
    body = transform(lines)
    args.output.write_text(
        "/* Generated from LAPACK 3.12.0 sources via f2c. */\n\n"
        + "\n".join(prototypes)
        + "\n\n"
        + "\n".join(body)
        + "\n"
    )


if __name__ == "__main__":
    main()
