# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared fixtures and automatic skips for tests that need NUPACK, oxDNA or a GPU."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

from seq2oxdna.fold import nupack_available
from seq2oxdna.simulate import find_oxdna

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(REPO, "examples")
REFERENCE = os.path.join(REPO, "tests", "data", "reference")


def pytest_collection_modifyitems(config, items):
    has_nupack = nupack_available()
    oxdna = find_oxdna()
    cuda = os.environ.get("SEQ2OXDNA_CUDA_TESTS") == "1"
    for item in items:
        if "nupack" in item.keywords and not has_nupack:
            item.add_marker(pytest.mark.skip(reason="NUPACK not installed"))
        if "oxdna" in item.keywords and not oxdna:
            item.add_marker(pytest.mark.skip(reason="no oxDNA ($OXDNA_BIN or on PATH)"))
        if "cuda" in item.keywords and not cuda:
            item.add_marker(pytest.mark.skip(reason="set SEQ2OXDNA_CUDA_TESTS=1 on a CUDA machine"))


def run_cli(*args, env=None):
    """Run `python -m seq2oxdna ...` in a subprocess; return the CompletedProcess."""
    return subprocess.run([sys.executable, "-m", "seq2oxdna", *map(str, args)],
                          capture_output=True, text=True, env=env)


@pytest.fixture
def spaced_tmp(tmp_path):
    """A temporary folder whose path contains a space (a real-user case that once broke)."""
    path = tmp_path / "with space"
    path.mkdir()
    return path


@pytest.fixture
def examples(spaced_tmp):
    """Copies of the shipped example inputs inside a spaced path."""
    for name in ("examples.fasta", "examples_given.db"):
        shutil.copy(os.path.join(EXAMPLES, name), spaced_tmp / name)
    return spaced_tmp
