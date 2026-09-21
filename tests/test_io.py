# SPDX-License-Identifier: GPL-3.0-or-later
"""Input parsing: every format, named and unnamed, with and without structures."""

from __future__ import annotations

import pandas as pd
import pytest

from seq2oxdna.io import read_named_sequences, resolve_tasks, safe_name


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_fasta_with_structures_multiline_and_rnafold_energy(tmp_path):
    path = _write(tmp_path, "in.fasta",
                  ">a\nGGGAAA\nCCC\n(((...\n)))\n"
                  ">b\nGGGAAAUCC\n(((...)))  (-3.40)\n"
                  ">c\nACGTACGT\n")
    assert read_named_sequences(path) == [
        ("a", "GGGAAACCC", "(((...)))"),
        ("b", "GGGAAAUCC", "(((...)))"),
        ("c", "ACGTACGT", None),
    ]


def test_plain_file_structure_attaches_to_previous_line(tmp_path):
    path = _write(tmp_path, "in.txt", "GGGAAACCC\n(((...)))\nACGTACGT\nGGGAAAUCC\n(((...)))\n")
    assert read_named_sequences(path) == [
        (None, "GGGAAACCC", "(((...)))"),
        (None, "ACGTACGT", None),
        (None, "GGGAAAUCC", "(((...)))"),
    ]


def test_fasta_without_structures(tmp_path):
    path = _write(tmp_path, "in.fasta", ">x\nACGT\nACGT\n>y\nGGGAAACCC\n")
    assert read_named_sequences(path) == [("x", "ACGTACGT", None), ("y", "GGGAAACCC", None)]


def test_plain_file_without_names(tmp_path):
    path = _write(tmp_path, "in.txt", "ACGTACGT\n\nGGGAAAUCC\n")
    assert read_named_sequences(path) == [(None, "ACGTACGT", None), (None, "GGGAAAUCC", None)]


@pytest.mark.parametrize("text,message", [
    ("(((...)))\nGGGAAACCC\n", "no sequence before it"),
    (">a\nGGGAAA\n(((...\nCCC\n", "sequence line after the structure"),
    (">a\n(((...)))\n", "no sequence before it"),
])
def test_malformed_structure_placement(tmp_path, text, message):
    path = _write(tmp_path, "in.fasta", text)
    with pytest.raises(ValueError, match=message):
        read_named_sequences(path)


def test_csv_structure_column_with_names_and_header(tmp_path):
    path = _write(tmp_path, "in.csv", "name,sequence,structure\na,GGGAAACCC,(((...)))\nb,ACGTACGT,\n")
    assert read_named_sequences(path) == [("a", "GGGAAACCC", "(((...)))"), ("b", "ACGTACGT", None)]


def test_csv_structure_column_is_not_taken_as_names(tmp_path):
    # Without structure-column detection the dot-brackets became the record names and
    # the structures were silently discarded.
    path = _write(tmp_path, "in.csv", "GGGAAACCC,(((...)))\nGGGAAAUCC,(((...)))\n")
    assert read_named_sequences(path) == [
        (None, "GGGAAACCC", "(((...)))"), (None, "GGGAAAUCC", "(((...)))")]


def test_csv_unnamed_single_column(tmp_path):
    path = _write(tmp_path, "in.csv", "ACGTACGT\nGGGAAAUCC\n")
    assert read_named_sequences(path) == [(None, "ACGTACGT", None), (None, "GGGAAAUCC", None)]


def test_tsv_named(tmp_path):
    path = _write(tmp_path, "in.tsv", "id\tseq\nh1\tGCGTTGCTTCTCCAACGC\n")
    assert read_named_sequences(path) == [("h1", "GCGTTGCTTCTCCAACGC", None)]


def test_xlsx_named_with_structure(tmp_path):
    path = tmp_path / "in.xlsx"
    pd.DataFrame({"name": ["a", "b"], "sequence": ["GGGAAACCC", "ACGTACGT"],
                  "structure": ["(((...)))", ""]}).to_excel(path, index=False)
    assert read_named_sequences(str(path)) == [("a", "GGGAAACCC", "(((...)))"), ("b", "ACGTACGT", None)]


def test_table_without_sequence_column_is_an_error(tmp_path):
    path = _write(tmp_path, "in.csv", "name,value\na,1\nb,2\n")
    with pytest.raises(ValueError, match="Could not identify"):
        read_named_sequences(path)


def test_resolve_tasks_validates_and_skips_bad_structures():
    records = [
        ("good", "GGGAAACCC", "(((...)))"),
        ("pk", "GGGAAACCC", "((([.])))"),
        ("gt_as_dna", "GGGAAATCC", "(((...)))"),
        ("folded", "ACGTACGT", None),
    ]
    tasks, skipped = resolve_tasks(records, material_mode="auto")
    assert [(t[1], t[3], t[4]) for t in tasks] == [("good", "dna", "(((...)))"), ("folded", "dna", None)]
    reasons = dict(skipped)
    assert "pseudoknots" in reasons["pk"]
    assert "G-T is not a pair" in reasons["gt_as_dna"]


def test_resolve_tasks_names_material_and_duplicates():
    records = [(None, "acgt", None), ("x y", "GGGAAAUCC", None), ("x y", "GGGAAAUCC", None),
               ("bad", "ACGUT", None), ("forced", "GGGAAACCC", None)]
    tasks, skipped = resolve_tasks(records)
    assert [(t[1], t[2], t[3]) for t in tasks] == [
        ("seq0", "ACGT", "dna"), ("x_y", "GGGAAAUCC", "rna"), ("x_y_2", "GGGAAAUCC", "rna"),
        ("forced", "GGGAAACCC", "dna")]
    assert "mixes T and U" in dict(skipped)["bad"]
    tasks, _ = resolve_tasks([("r", "GGGAAACCC", None)], material_mode="rna")
    assert tasks[0][3] == "rna"


def test_safe_name():
    assert safe_name(">my seq/1 ") == "my_seq_1"
    assert safe_name("...") == "seq"


def test_resolve_tasks_names_avoid_reserved_and_case_collisions():
    """Record folders must not take logs/manifest.csv/run_parameters.json, nor collide with each
    other on a case-insensitive filesystem (macOS). Other names are unchanged."""
    records = [("logs", "ACGTACGT", None), ("manifest.csv", "ACGTACGT", None),
               ("Run_Parameters.JSON", "ACGTACGT", None), ("hp", "ACGTACGT", None),
               ("HP", "ACGTACGT", None), ("hp", "ACGTACGT", None), ("x", "ACGTACGT", None),
               ("x_2", "ACGTACGT", None), ("x", "ACGTACGT", None)]
    tasks, _ = resolve_tasks(records)
    assert [t[1] for t in tasks] == ["logs_2", "manifest.csv_2", "Run_Parameters.JSON_2", "hp",
                                     "HP_2", "hp_3", "x", "x_2", "x_3"]
