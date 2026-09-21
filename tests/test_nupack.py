# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests that fold with NUPACK. Skipped automatically when NUPACK is not installed (CI)."""

from __future__ import annotations

import csv
import filecmp
import os
import re

import pytest

from seq2oxdna.fold import dot_bracket_to_mfe, nupack_version, pinned_dna_parameter_set

from conftest import REFERENCE, run_cli

pytestmark = pytest.mark.nupack

FILES = ("sequence.top", "sequence.dat", "forces.txt")


def _manifest(outdir):
    with open(os.path.join(outdir, "manifest.csv")) as fh:
        return list(csv.DictReader(fh))


def test_pinned_dna_parameter_set():
    version = tuple(int(p) for p in re.findall(r"\d+", nupack_version())[:3])
    assert pinned_dna_parameter_set() == ("dna04.1" if version >= (4, 0, 2) else "dna04")


def test_hairpin_mfe():
    assert dot_bracket_to_mfe("GCGTTGCTTCTCCAACGC", "dna") == "((((((......))))))"


def test_examples_fold_to_the_reference_structures(examples):
    out = examples / "out"
    proc = run_cli("prepare", examples / "examples.fasta", "--outdir", out, "--jobs", "4")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    rows = _manifest(out)
    assert {r["structure_source"] for r in rows} == {"nupack"}
    for r in rows:
        with open(os.path.join(REFERENCE, r["name"], f"{r['name']}.db")) as fh:
            assert r["dot_bracket"] == fh.read().strip(), r["name"]
        assert r["nupack_dna_parameter_set"] == pinned_dna_parameter_set()


@pytest.mark.reference
@pytest.mark.skipif(os.environ.get("SEQ2OXDNA_EXACT") != "1",
                    reason="byte-exact only on the validation machine (set SEQ2OXDNA_EXACT=1)")
def test_examples_byte_exact_against_reference(examples):
    out = examples / "out"
    for name in ("examples.fasta", "examples_given.db"):
        proc = run_cli("prepare", examples / name, "--outdir", out, "--overwrite")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        for row in _manifest(out):
            rec = row["name"]
            for f in (*FILES, f"{rec}.db"):
                assert filecmp.cmp(out / rec / f, os.path.join(REFERENCE, rec, f), shallow=False), (rec, f)


def test_dot_bracket_route_matches_nupack_route(examples):
    """Feeding NUPACK's own MFE back in as a dot-bracket must give byte-identical inputs."""
    folded_dir, given_dir = examples / "out nupack", examples / "out given"
    proc = run_cli("prepare", examples / "examples.fasta", "--outdir", folded_dir)
    assert proc.returncode == 0, proc.stderr
    folded = _manifest(folded_dir)
    seqs = {}
    with open(examples / "examples.fasta") as fh:
        name = None
        for line in fh:
            if line.startswith(">"):
                name = line[1:].strip()
            elif line.strip():
                seqs[name] = line.strip()
    db = examples / "as given.db"
    db.write_text("".join(f">{r['name']}\n{seqs[r['name']]}\n{r['dot_bracket']}\n" for r in folded))
    proc = run_cli("prepare", db, "--outdir", given_dir)
    assert proc.returncode == 0, proc.stderr
    given = _manifest(given_dir)
    assert {r["structure_source"] for r in given} == {"input"}
    for r in folded:
        a, b = folded_dir / r["name"], given_dir / r["name"]
        files = sorted(os.listdir(a))
        assert files == sorted(os.listdir(b))
        _match, mismatch, errors = filecmp.cmpfiles(a, b, files, shallow=False)
        assert not mismatch and not errors, (r["name"], mismatch, errors)


def test_mixed_file_of_sequences_and_structures(tmp_path):
    path = tmp_path / "mixed.fasta"
    path.write_text(">folded\nGCGTTGCTTCTCCAACGC\n>given\nGCGTTGCTTCTCCAACGC\n((((((......))))))\n")
    proc = run_cli("prepare", path, "--outdir", tmp_path / "out")
    assert proc.returncode == 0, proc.stderr
    rows = {r["name"]: r for r in _manifest(tmp_path / "out")}
    assert rows["folded"]["structure_source"] == "nupack"
    assert rows["given"]["structure_source"] == "input"
    for f in FILES:
        assert filecmp.cmp(tmp_path / "out" / "folded" / f, tmp_path / "out" / "given" / f, shallow=False)
