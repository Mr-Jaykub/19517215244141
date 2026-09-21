# SPDX-License-Identifier: GPL-3.0-or-later
"""Input parsing: FASTA / plain / CSV / TSV / Excel, with or without names and structures."""

from __future__ import annotations

import os
import re

import pandas as pd

from .fold import DEFAULT_MATERIAL, detect_material, normalize_sequence, validate_dot_bracket

_DNA_CHARS = set("ACGTUN")


def safe_name(raw: str) -> str:
    """Filesystem-safe token from a sequence name or file stem."""
    raw = str(raw).strip().lstrip(">").strip()
    raw = re.sub(r"\s+", "_", raw)
    raw = re.sub(r"[^A-Za-z0-9._-]", "_", raw)
    raw = raw.strip("_.")
    return raw or "seq"


def _looks_like_dna(value, min_len: int = 4, min_frac: float = 0.9) -> bool:
    """True if `value` is plausibly a nucleotide sequence (mostly A/C/G/T/U/N)."""
    if value is None:
        return False
    s = str(value).strip().upper().replace(" ", "")
    if len(s) < min_len:
        return False
    good = sum(c in _DNA_CHARS for c in s)
    return good / len(s) >= min_frac


# A structure line: dot-bracket characters only. Pseudoknot brackets and strand breaks are
# recognised here so the validator can reject them with a clear message instead of the
# line being mistaken for sequence.
_STRUCTURE_LINE_RE = re.compile(r"^[.()\[\]{}<>&+]+$")
# RNAfold appends the MFE after whitespace: "((((....))))  (-3.40)".
_RNAFOLD_ENERGY_RE = re.compile(r"\s+\(\s*[-+]?\d+(?:\.\d+)?\s*\)$")


def _structure_line(line: str):
    """Return the dot-bracket part of `line` if it is a structure line, else None."""
    candidate = _RNAFOLD_ENERGY_RE.sub("", line).strip()
    return candidate if candidate and _STRUCTURE_LINE_RE.match(candidate) else None


def _read_fasta_or_plain(path: str):
    """Return [(name|None, raw_seq, structure|None)] from a FASTA/.db or one-per-line file.

    A record may carry a dot-bracket structure on the line(s) after its sequence
    (ViennaRNA / RNAfold ".db" layout). In a header-less file each sequence line is a
    record, and a structure line attaches to the sequence line before it.
    """
    records = []
    cur_name = None
    cur_seq = []
    cur_struct = []
    saw_header = False

    def flush():
        if cur_seq or cur_struct:
            if not cur_seq:
                raise ValueError(f"{path}: record {cur_name!r} has a structure but no sequence")
            records.append((cur_name, "".join(cur_seq), "".join(cur_struct) or None))

    with open(path) as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                cur_seq, cur_struct = [], []
                cur_name = line[1:].strip() or None
                saw_header = True
                continue
            structure = _structure_line(line)
            if structure is not None:
                if not cur_seq:
                    raise ValueError(f"{path}:{lineno}: structure line with no sequence before it")
                cur_struct.append(structure)
                continue
            if cur_struct:
                if saw_header:
                    raise ValueError(f"{path}:{lineno}: sequence line after the structure in "
                                     f"record {cur_name!r}")
                flush()
                cur_seq, cur_struct = [], []
            if saw_header:
                cur_seq.append(line)
            else:
                flush()
                cur_seq, cur_struct = [line], []
    flush()
    return records


def _records_from_dataframe(df: "pd.DataFrame", seq_col=None, name_col=None):
    """Pull (name|None, raw_seq, structure|None) rows out of a header-less DataFrame.

    The sequence column is the one whose cells look most like DNA. A first row
    that is not DNA-like in that column is treated as a header and dropped. A column
    whose cells are all dot-bracket strings is the structure column (without this it
    would be taken as the name column and the structures silently dropped). The first
    remaining non-empty column becomes the name column.
    """
    df = df.fillna("").apply(lambda col: col.map(lambda x: str(x).strip()))
    df = df.loc[~(df == "").all(axis=1)]  # drop fully empty rows
    df = df.reset_index(drop=True)
    if df.empty:
        return []
    ncols = df.shape[1]

    if seq_col is None:
        best, best_frac = None, 0.0
        for c in range(ncols):
            vals = [v for v in df.iloc[:, c].tolist() if v != ""]
            if not vals:
                continue
            frac = sum(_looks_like_dna(v) for v in vals) / len(vals)
            if frac > best_frac:
                best, best_frac = c, frac
        if best is None or best_frac < 0.5:
            raise ValueError("Could not identify a DNA sequence column in the table")
        seq_col = best

    start = 1 if not _looks_like_dna(df.iat[0, seq_col]) else 0

    struct_col = None
    for c in range(ncols):
        if c == seq_col:
            continue
        vals = [v for v in df.iloc[start:, c].tolist() if v != ""]
        if vals and all(_structure_line(v) is not None for v in vals):
            struct_col = c
            break

    if name_col is None:
        for c in range(ncols):
            if c in (seq_col, struct_col):
                continue
            if any(v != "" for v in df.iloc[start:, c].tolist()):
                name_col = c
                break

    records = []
    for r in range(start, df.shape[0]):
        seq = df.iat[r, seq_col]
        if seq == "" or not _looks_like_dna(seq):
            continue
        name = df.iat[r, name_col] if name_col is not None else ""
        structure = _structure_line(df.iat[r, struct_col]) if struct_col is not None else None
        records.append((name or None, seq, structure))
    return records


def read_named_sequences(path: str, seq_col=None, name_col=None):
    """Dispatch on file extension and return a list of (name|None, raw_seq, structure|None)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv",):
        df = pd.read_csv(path, header=None, dtype=str, keep_default_na=False, engine="python")
        return _records_from_dataframe(df, seq_col, name_col)
    if ext in (".tsv", ".tab"):
        df = pd.read_csv(path, header=None, dtype=str, sep="\t", keep_default_na=False, engine="python")
        return _records_from_dataframe(df, seq_col, name_col)
    if ext in (".xlsx", ".xls", ".xlsm"):
        df = pd.read_excel(path, header=None, dtype=str)
        return _records_from_dataframe(df, seq_col, name_col)
    # .fasta/.fa/.fna/.txt and everything else -> FASTA or plain lines
    return _read_fasta_or_plain(path)


def resolve_tasks(records, material_mode: str = "auto"):
    """Normalise sequences, pick each one's material, and assign unique names.

    `material_mode` is 'auto' (detect DNA vs RNA per sequence from U/T, defaulting
    to DNA for A/C/G-only sequences), or a forced 'dna'/'rna'. Returns (tasks,
    skipped) where tasks is a list of (index, name, seq, material, structure|None) and
    skipped is a list of (raw_name, reason). A supplied structure is validated here, so
    a bad one is skipped with its reason before any work starts.
    """
    tasks = []
    skipped = []
    seen = {}
    idx = 0
    for raw_name, raw_seq, raw_structure in records:
        try:
            if material_mode == "auto":
                material = detect_material(raw_seq) or DEFAULT_MATERIAL
            else:
                material = material_mode
            seq = normalize_sequence(raw_seq, material)
            structure = (validate_dot_bracket(seq, raw_structure, material)
                         if raw_structure is not None else None)
        except ValueError as exc:
            skipped.append((raw_name or raw_seq[:20], str(exc)))
            continue
        base = safe_name(raw_name) if raw_name else f"seq{idx}"
        count = seen.get(base, 0)
        name = base if count == 0 else f"{base}_{count + 1}"
        seen[base] = count + 1
        tasks.append((idx, name, seq, material, structure))
        idx += 1
    return tasks, skipped

