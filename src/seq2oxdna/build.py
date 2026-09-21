# SPDX-License-Identifier: GPL-3.0-or-later
"""Build the oxDNA input files for one record: structure -> layout -> top/dat/forces/db."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

from .fold import (DEFAULT_MAGNESIUM, DEFAULT_MATERIAL, DEFAULT_SODIUM, MATERIALS,
                   detect_material, dot_bracket_to_mfe, normalize_sequence, parse_pairs,
                   reverse_dot_bracket, validate_dot_bracket)
from .layout import (MATERIAL_TWIST, STERIC_CLEAR, _close_nonbonded_pairs, build_structure,
                     count_threaded_loops)
from .writers import write_conf, write_force_file, write_topology


@dataclass
class SequenceResult:
    name: str
    sequence: str
    dot_bracket: str
    oxdna_dot_bracket: str
    top_path: str
    conf_path: str
    force_path: str
    material: str = DEFAULT_MATERIAL
    interaction_type: str = "DNA2"
    geom_overlaps: int = 0          # residual branch overlaps in the starting geometry
    geom_threads: int = 0           # residual threaded loops in the starting geometry
    structure_source: str = "nupack"  # "nupack" (MFE prediction) or "input" (user dot-bracket)


def build_sequence_inputs(
    name: str,
    sequence: str,
    folder: str,
    temp_c: float,
    concentration: float,
    stiffness: float,
    rest_length: float,
    material: str | None = None,
    sodium: float = DEFAULT_SODIUM,
    magnesium: float = DEFAULT_MAGNESIUM,
    dot_bracket: str | None = None,
) -> SequenceResult:
    """Build oxDNA inputs for one sequence in `folder`.

    Writes sequence.top, sequence.dat (and an identical copy <name>.dat), forces.txt and
    <name>.db. `material` is 'dna', 'rna', or None to auto-detect from the sequence
    (U->RNA, T->DNA, ambiguous A/C/G-only -> DEFAULT_MATERIAL). `sodium`/`magnesium`
    (molar) feed NUPACK's salt correction.

    `dot_bracket`, if given, is used as the structure instead of NUPACK's MFE (after
    `validate_dot_bracket`); NUPACK is then not called. Everything downstream -- layout,
    geometry QC, topology/conf/forces -- is the same code path either way.
    """
    if material is None:
        material = detect_material(sequence) or DEFAULT_MATERIAL
    material = material.lower()
    mat_cfg = MATERIALS[material]

    sequence = normalize_sequence(sequence, material)
    if dot_bracket is not None:
        dot_bracket = validate_dot_bracket(sequence, dot_bracket, material)
        structure_source = "input"
    else:
        dot_bracket = dot_bracket_to_mfe(sequence, material=material, temp_c=temp_c,
                                         concentration=concentration, sodium=sodium,
                                         magnesium=magnesium)
        structure_source = "nupack"

    # Everything downstream works in oxDNA particle order (3' -> 5').
    oxdna_sequence = sequence[::-1]
    oxdna_dot_bracket = reverse_dot_bracket(dot_bracket)
    pair_map, pairs = parse_pairs(oxdna_dot_bracket)

    pos, a1, a3 = build_structure(pair_map, twist=MATERIAL_TWIST[material])

    # QC the starting geometry: it must be free of branch overlaps and threaded
    # loops, otherwise the downstream MC/MD can relax the defect into a knot.
    n_overlaps = len(_close_nonbonded_pairs(pos, pair_map, STERIC_CLEAR))
    n_threads = count_threaded_loops(pair_map, pos)
    if n_overlaps or n_threads:
        print(f"[{name}] WARNING starting geometry still has {n_overlaps} branch "
              f"overlap(s) and {n_threads} threaded loop(s) after resolution")

    os.makedirs(folder, exist_ok=True)
    top_path = os.path.join(folder, "sequence.top")
    conf_path = os.path.join(folder, "sequence.dat")
    force_path = os.path.join(folder, "forces.txt")

    write_topology(oxdna_sequence, top_path)
    write_conf(oxdna_sequence, pos, a1, a3, conf_path)
    shutil.copy(conf_path, os.path.join(folder, f"{name}.dat"))
    write_force_file(pairs, force_path, stiffness=stiffness, rest_length=rest_length)

    with open(os.path.join(folder, f"{name}.db"), "w") as handle:
        handle.write(dot_bracket + "\n")

    return SequenceResult(
        name=name,
        sequence=sequence,
        dot_bracket=dot_bracket,
        oxdna_dot_bracket=oxdna_dot_bracket,
        top_path=top_path,
        conf_path=conf_path,
        force_path=force_path,
        material=material,
        interaction_type=mat_cfg["interaction"],
        geom_overlaps=n_overlaps,
        geom_threads=n_threads,
        structure_source=structure_source,
    )
