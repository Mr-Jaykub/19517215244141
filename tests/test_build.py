# SPDX-License-Identifier: GPL-3.0-or-later
"""Layout, writers and the reference inputs (dot-bracket route: no NUPACK needed).

The committed references in tests/data/reference/ were produced by the validation run
(NUPACK 4.0.2.0, DNA parameter set dna04.1). Feeding each record's reference structure
back in as a dot-bracket must reproduce its inputs: numerically (1e-6) everywhere, and
byte for byte on the validation machine (SEQ2OXDNA_EXACT=1).
"""

from __future__ import annotations

import filecmp
import os

import numpy as np
import pytest

from seq2oxdna.build import build_sequence_inputs
from seq2oxdna.fold import parse_pairs
from seq2oxdna.layout import BB_TARGET, PAIR_TARGET, build_structure

from conftest import REFERENCE

RECORDS = sorted(os.listdir(REFERENCE))
FILES = ("sequence.top", "sequence.dat", "forces.txt")


def _reference(record):
    folder = os.path.join(REFERENCE, record)
    with open(os.path.join(folder, "sequence.top")) as fh:
        top = fh.read().split("\n")
    # topology is 3'->5'; the base letters reversed give the 5'->3' sequence
    seq = "".join(line.split()[1] for line in top[1:] if line.strip())[::-1]
    with open(os.path.join(folder, f"{record}.db")) as fh:
        db = fh.read().strip()
    return folder, seq, db


def _build(record, out):
    _folder, seq, db = _reference(record)
    return build_sequence_inputs(name=record, sequence=seq, folder=str(out / record), temp_c=37.0,
                                 concentration=1e-6, stiffness=1.0, rest_length=1.2,
                                 dot_bracket=db)


def _numbers(path):
    out = []
    with open(path) as fh:
        for token in fh.read().split():
            try:
                out.append(float(token))
            except ValueError:
                out.append(token)
    return out


@pytest.mark.parametrize("record", RECORDS)
def test_reference_inputs_numerically(record, spaced_tmp):
    res = _build(record, spaced_tmp)
    ref = os.path.join(REFERENCE, record)
    for f in FILES:
        got, want = _numbers(os.path.join(spaced_tmp, record, f)), _numbers(os.path.join(ref, f))
        assert len(got) == len(want), f
        for g, w in zip(got, want):
            if isinstance(w, float):
                assert isinstance(g, float) and abs(g - w) <= 1e-6, (f, g, w)
            else:
                assert g == w, (f, g, w)
    assert filecmp.cmp(os.path.join(spaced_tmp, record, f"{record}.db"),
                       os.path.join(ref, f"{record}.db"), shallow=False)
    assert res.structure_source == "input" and res.geom_overlaps == 0 and res.geom_threads == 0


@pytest.mark.reference
@pytest.mark.skipif(os.environ.get("SEQ2OXDNA_EXACT") != "1",
                    reason="byte-exact only on the validation machine (set SEQ2OXDNA_EXACT=1)")
@pytest.mark.parametrize("record", RECORDS)
def test_reference_inputs_byte_exact(record, tmp_path):
    _build(record, tmp_path)
    for f in (*FILES, f"{record}.db"):
        assert filecmp.cmp(tmp_path / record / f, os.path.join(REFERENCE, record, f), shallow=False), f


@pytest.mark.parametrize("record", RECORDS)
def test_files_are_consistent(record, tmp_path):
    res = _build(record, tmp_path)
    n = len(res.sequence)
    top = open(res.top_path).read().splitlines()
    assert top[0] == f"{n} 1" and len(top) == n + 1
    conf = open(res.conf_path).read().splitlines()
    assert conf[0] == "t = 0" and conf[1].startswith("b = ") and len(conf) == n + 3
    assert all(len(line.split()) == 15 for line in conf[3:])
    assert open(tmp_path / record / f"{record}.dat").read() == open(res.conf_path).read()

    # Force file: two reciprocal traps per base pair, nothing else.
    traps = [block for block in open(res.force_path).read().split("}") if "mutual_trap" in block]
    pairs = parse_pairs(res.oxdna_dot_bracket)[1]
    assert len(traps) == 2 * len(pairs)
    got = set()
    for block in traps:
        kv = dict(line.split(" = ") for line in block.strip().strip("{").strip().splitlines())
        got.add((int(kv["particle"]), int(kv["ref_particle"])))
        assert kv["stiff"] == "1.0" and kv["r0"] == "1.2" and kv["PBC"] == "1"
    assert got == {(i, j) for i, j in pairs} | {(j, i) for i, j in pairs}


@pytest.mark.parametrize("record", RECORDS)
def test_layout_bond_and_pair_lengths_are_exact(record):
    _folder, seq, db = _reference(record)
    from seq2oxdna.fold import reverse_dot_bracket
    from seq2oxdna.layout import MATERIAL_TWIST
    pair_map, pairs = parse_pairs(reverse_dot_bracket(db))
    material = "rna" if "U" in seq else "dna"
    pos, a1, a3 = build_structure(pair_map, twist=MATERIAL_TWIST[material])
    bonds = np.linalg.norm(pos[1:] - pos[:-1], axis=1)
    assert np.allclose(bonds, BB_TARGET, atol=1e-6)
    assert np.allclose([np.linalg.norm(pos[i] - pos[j]) for i, j in pairs], PAIR_TARGET, atol=1e-6)


def test_non_nested_structure_is_rejected_by_the_layout():
    pair_map, _ = parse_pairs("((..[[..))..]]")
    with pytest.raises(ValueError, match="pseudoknotted"):
        build_structure(pair_map)
