# SPDX-License-Identifier: GPL-3.0-or-later
"""oxDNA file writers: topology, configuration and mutual-trap force file."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from .layout import unit


def write_topology(sequence: str, output_path: str) -> None:
    """Write an oxDNA topology file in the repository's existing format.

    ``sequence`` is already in oxDNA particle order (3' -> 5'); particle 0 is the
    3' end. Each line is ``strand base 3'-neighbour 5'-neighbour``.
    """
    n = len(sequence)
    with open(output_path, "w") as handle:
        handle.write(f"{n} 1\n")
        for idx, base in enumerate(sequence):
            neighbour_3 = idx - 1
            neighbour_5 = idx + 1 if idx < n - 1 else -1
            handle.write(f"1 {base} {neighbour_3} {neighbour_5}\n")


def write_conf(sequence: str, pos: np.ndarray, a1: np.ndarray, a3: np.ndarray, output_path: str) -> None:
    """Write an oxDNA configuration (.dat) with positions and orthonormal orientations."""
    n = len(sequence)
    lo = pos.min(axis=0)
    hi = pos.max(axis=0)
    span = float((hi - lo).max())
    box = max(span + 40.0, 50.0)
    shifted = pos - (lo + hi) / 2.0 + box / 2.0

    with open(output_path, "w") as handle:
        handle.write("t = 0\n")
        handle.write(f"b = {box} {box} {box}\n")
        handle.write("E = 0 0 0\n")
        for idx in range(n):
            a3v = unit(a3[idx], fallback=np.array((0.0, 0.0, 1.0)))
            a1v = a1[idx] - np.dot(a1[idx], a3v) * a3v
            a1v = unit(a1v, fallback=np.cross(a3v, np.array((1.0, 0.0, 0.0))))
            x, y, z = shifted[idx]
            handle.write(
                f"{x} {y} {z} "
                f"{a1v[0]} {a1v[1]} {a1v[2]} "
                f"{a3v[0]} {a3v[1]} {a3v[2]} "
                f"0 0 0 0 0 0\n"
            )


def write_force_file(pairs: Sequence[Tuple[int, int]], output_path: str, stiffness: float, rest_length: float) -> None:
    """Write reciprocal mutual traps for every base pair (matches seq2force.py/db_to_force.py).

    An oxDNA ``mutual_trap`` is one-directional: it pulls ``particle`` toward
    ``ref_particle`` only. Holding a base pair therefore needs two traps -- one in
    each direction -- exactly as db_to_force.py emits by iterating the (symmetric)
    pair map over every nucleotide. Writing only one direction leaves each pair's
    partner unrestrained, so the structure never settles during relaxation.
    """
    def trap(particle: int, ref_particle: int) -> str:
        return (
            "{\n"
            "type = mutual_trap\n"
            f"particle = {particle}\n"
            f"ref_particle = {ref_particle}\n"
            f"stiff = {stiffness}\n"
            f"r0 = {rest_length}\n"
            "PBC = 1\n"
            "}\n\n"
        )

    with open(output_path, "w") as handle:
        for left, right in pairs:
            handle.write(trap(left, right))
            handle.write(trap(right, left))
