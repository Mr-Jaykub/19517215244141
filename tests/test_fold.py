# SPDX-License-Identifier: GPL-3.0-or-later
"""Materials, dot-bracket parsing and every validator rule (no NUPACK needed)."""

from __future__ import annotations

import re

import pytest

from seq2oxdna.fold import (NupackMissingError, check_nupack_domain, detect_material, is_nested,
                            normalize_sequence, parse_pairs, reverse_dot_bracket,
                            validate_dot_bracket)


@pytest.mark.parametrize("seq,db,material", [
    ("GCGTTGCTTCTCCAACGC", "((((((......))))))", "dna"),
    ("GGGAAAUCC", "(((...)))", "rna"),
    ("GGGAAAUUC", "(((...)))", "rna"),        # G-U wobble is canonical for RNA
    ("ACGTACGT", "........", "dna"),          # unstructured is fine
])
def test_valid_structures_pass(seq, db, material):
    assert validate_dot_bracket(seq, f"  {db} \n", material) == db


@pytest.mark.parametrize("seq,db,material,message", [
    ("GGGAAACCC", "(((.&.)))", "dna", "multi-strand"),
    ("GGGAAACCC", "(((.+.)))", "dna", "multi-strand"),
    ("GGGAAACCC", "((([.])))", "dna", "pseudoknots are not supported"),
    ("GGGAAACCC", "(((.x.)))", "dna", "unsupported dot-bracket character"),
    ("GGGAAACCC", "(((...))", "dna", "does not match sequence length"),
    ("GGGAAACCC", ")((...)))", "dna", "has no matching '('"),
    ("GGGAAACCC", "(((....))", "dna", "is never closed"),
    ("GGGAAATCC", "(((...)))", "dna", "G-T is not a pair"),
    ("GAGAAAUCC", "(((...)))", "rna", "not a canonical RNA pair"),
    ("GGGAACCC", "(((..)))", "dna", "hairpin loop"),
])
def test_invalid_structures_rejected(seq, db, material, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        validate_dot_bracket(seq, db, material)


def test_error_positions_are_one_based():
    with pytest.raises(ValueError, match=r"positions 1-9 pair G-A"):
        validate_dot_bracket("GGGAAACCA", "(((...)))", "dna")


@pytest.mark.parametrize("seq,expected", [("ACGU", "rna"), ("ACGT", "dna"), ("ACGA", None)])
def test_detect_material(seq, expected):
    assert detect_material(seq) == expected


def test_detect_material_rejects_mixed():
    with pytest.raises(ValueError, match="mixes T and U"):
        detect_material("ACGTU")


def test_normalize_sequence():
    assert normalize_sequence(" acgu ", "dna") == "ACGT"
    assert normalize_sequence("ACGT", "rna") == "ACGU"
    with pytest.raises(ValueError, match="unsupported bases"):
        normalize_sequence("ACGN", "dna")
    with pytest.raises(ValueError, match="Unknown material"):
        normalize_sequence("ACGT", "xna")


def test_reverse_and_parse_round_trip():
    db = "((..((...))..)).."
    rev = reverse_dot_bracket(db)
    assert rev == "..((..((...))..))"
    assert reverse_dot_bracket(rev) == db
    pair_map, pairs = parse_pairs(db)
    assert pairs == [(0, 14), (1, 13), (4, 10), (5, 9)]
    assert all(pair_map[j] == i for i, j in pairs) and is_nested(pair_map)


def test_is_nested_detects_crossing_pairs():
    pair_map, _ = parse_pairs("((..[[..))..]]")
    assert not is_nested(pair_map)


@pytest.mark.parametrize("kwargs", [dict(sodium=1.5), dict(magnesium=0.3), dict(temp_c=-5)])
def test_nupack_domain_checked_before_nupack(kwargs):
    with pytest.raises(ValueError, match="outside NUPACK's accepted range"):
        check_nupack_domain(**kwargs)


def test_missing_nupack_gives_install_hint(monkeypatch):
    import builtins
    from seq2oxdna import fold

    real_import = builtins.__import__

    def no_nupack(name, *args, **kwargs):
        if name == "nupack" or name.startswith("nupack."):
            raise ImportError("No module named 'nupack'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_nupack)
    assert fold.nupack_version() is None and not fold.nupack_available()
    with pytest.raises(NupackMissingError, match="www.nupack.org"):
        fold.dot_bracket_to_mfe("GGGAAACCC", "dna")
