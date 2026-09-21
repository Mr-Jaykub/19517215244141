#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Run the ```bash blocks of README.md, in order, as one script (release gate R6).

A block may be preceded by an HTML comment that says where it can run:

    <!-- check_readme: local -->   needs NUPACK, a GPU or conda: run locally, not in CI
    <!-- check_readme: never -->   needs a file the user downloads (NUPACK): never run

Blocks without a comment run everywhere. `--ci` runs only those; without it, the local
blocks run too (do that inside a fresh environment on a CUDA machine with NUPACK).
The blocks run in a temporary copy of the repository, so nothing is written here; an
existing ./oxDNA build is linked into the copy rather than rebuilt.

    python scripts/check_readme.py --ci        # what CI runs
    python scripts/check_readme.py --list      # show the blocks and their modes
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODE_RE = re.compile(r"<!--\s*check_readme:\s*(local|never)\s*-->")


def bash_blocks(path):
    """[(mode, first_line_number, code)] for every ```bash block."""
    lines = open(path).read().splitlines()
    blocks, i = [], 0
    while i < len(lines):
        if lines[i].strip() == "```bash":
            prev = next((lines[k] for k in range(i - 1, -1, -1) if lines[k].strip()), "")
            m = MODE_RE.search(prev)
            start, i = i + 1, i + 1
            while lines[i].strip() != "```":
                i += 1
            blocks.append((m.group(1) if m else "ci", start + 1, "\n".join(lines[start:i])))
        i += 1
    return blocks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ci", action="store_true", help="run only the blocks marked for CI")
    ap.add_argument("--list", action="store_true", help="list the blocks and exit")
    args = ap.parse_args()

    blocks = bash_blocks(os.path.join(REPO, "README.md"))
    run_modes = {"ci"} if args.ci else {"ci", "local"}
    for mode, line, code in blocks:
        mark = "run " if mode in run_modes else "skip"
        print(f"[{mark}] README.md:{line} ({mode}): {code.splitlines()[0]}")
    if args.list:
        return 0

    script = ["set -euo pipefail"]
    for mode, line, code in blocks:
        if mode in run_modes:
            script += [f"echo '>>> README.md:{line}'", code]
    with tempfile.TemporaryDirectory(prefix="check readme ") as tmp:   # a space on purpose
        work = os.path.join(tmp, "repo")
        shutil.copytree(REPO, work, ignore=shutil.ignore_patterns(
            ".git", "oxDNA", "*_oxdna", "__pycache__", "*.egg-info", "build", "dist"))
        if os.path.isdir(os.path.join(REPO, "oxDNA")):
            os.symlink(os.path.join(REPO, "oxDNA"), os.path.join(work, "oxDNA"))
        proc = subprocess.run(["bash", "-c", "\n".join(script)], cwd=work)
    print("README commands: " + ("PASS" if proc.returncode == 0 else f"FAIL (exit {proc.returncode})"))
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
