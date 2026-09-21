# SPDX-License-Identifier: GPL-3.0-or-later
"""oxDNA-native starting geometry for a nested secondary structure.

The structure is decomposed into stems, loops and termini and built so that bond lengths
are a hard constraint of the construction: stems are antiparallel helices (the rung twists
by the material's twist per base pair, with the axial rise chosen so the backbone bond
stays fixed), and every loop is a circle whose radius is solved so that its perimeter
equals the contour length of the strand around it (one backbone-bond chord per
nucleotide, one base-pair chord per closing helix). Every pair of adjacent nucleotides
therefore ends up a safe distance apart -- none are crushed together (which would cause
exponential FENE blow-up) or over-stretched -- and base partners sit at the mutual-trap
rest length.

Because that bond-correct radius is small, a flat drawing would overlap its own branches,
so each child subtree is tilted out of its loop plane by rotating it about its
closing-pair axis. oxDNA's relaxation then twists the ladders into helices and the
mutual traps in forces.txt hold the base pairs during the early simulation stages.

Only nested structures are supported (pseudoknots are rejected before they reach here).
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

from .fold import is_nested

# --- layout target distances, in oxDNA simulation units (1 su ~ 0.8518 nm)
# These are chosen so every backbone bond is a safe FENE length and base pairs sit
# at the mutual-trap rest length, by construction of the diagram (see build_layout).
BB_TARGET = 0.72                      # backbone bond length between adjacent nucleotides
PAIR_TARGET = 1.20                    # COM-COM distance across a base pair (mutual-trap r0)
UMBRELLA_TILT = math.radians(30.0)    # out-of-plane tilt given to subtrees (3D spread, thread-safe)

# Right-handed helical twist applied per base pair within a stem (~10.5 bp/turn, B-form).
# The axial rise is derived so the backbone bond stays exactly BB_TARGET: a strand step
# combines the rise (along the axis) with the rung's rotation (radius PAIR_TARGET/2), and
# sqrt(rise^2 + (PAIR_TARGET*sin(twist/2))^2) == BB_TARGET. Twist 0 reproduces a flat ladder.
HELIX_TWIST = math.radians(34.3)
HELIX_RISE = math.sqrt(max(BB_TARGET ** 2 - (PAIR_TARGET * math.sin(HELIX_TWIST / 2.0)) ** 2, 1e-6))

# A-form twist for RNA (~11 bp/turn, 32.7 deg/bp). oxRNA relaxes stems to A-form,
# so RNA layouts are seeded with A-form twist; DNA keeps B-form (HELIX_TWIST above).
# BB_TARGET / PAIR_TARGET are shared: they are COM-COM *starting* targets (the
# mutual-trap rest length and a safe FENE backbone chord), and MC/MD under the
# correct interaction_type reshapes the helix to the material's true geometry.
RNA_HELIX_TWIST = math.radians(32.7)

# Starting helix twist per material (the development pipeline kept this in MATERIALS[...]["twist"]).
MATERIAL_TWIST = {"dna": HELIX_TWIST, "rna": RNA_HELIX_TWIST}


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------

def unit(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        if fallback is None:
            return np.array((1.0, 0.0, 0.0), dtype=float)
        fallback = np.asarray(fallback, dtype=float)
        fnorm = float(np.linalg.norm(fallback))
        if fnorm < 1e-12:
            return np.array((1.0, 0.0, 0.0), dtype=float)
        return fallback / fnorm
    return vector / norm


# ---------------------------------------------------------------------------
# oxDNA-native coordinate builder
# ---------------------------------------------------------------------------

def _ordered_loop_elements(pair_map: Sequence[int], lo: int, hi: int) -> List[Tuple[str, int, int]]:
    """Walk a chain range and list its child stems and unpaired bases in order."""
    elements: List[Tuple[str, int, int]] = []
    i = lo
    while i <= hi:
        j = pair_map[i]
        if j != -1 and j > i:
            elements.append(("stem", i, j))
            i = j + 1
        else:
            elements.append(("unpaired", i, -1))
            i += 1
    return elements


def _chord_angle(chord: float, r: float) -> float:
    """Central angle subtended by a chord of given length on a circle of radius r."""
    return 2.0 * math.asin(min(chord / (2.0 * r), 1.0))


def _solve_loop_radius(n_bb: int, n_pair: int) -> float:
    """Radius of a loop circle whose perimeter is exactly the right contour length.

    Walking around the loop, each backbone bond is a chord of length BB_TARGET and
    each closing base pair a chord of length PAIR_TARGET. We pick the radius so the
    chord angles sum to 2*pi, which guarantees every adjacent pair of nucleotides
    ends up the intended distance apart (no too-short or over-stretched bonds).
    """
    def total(r: float) -> float:
        return n_bb * _chord_angle(BB_TARGET, r) + n_pair * _chord_angle(PAIR_TARGET, r)

    r_lo = PAIR_TARGET / 2.0 + 1e-6        # smaller than this and a pair chord cannot fit
    r_hi = max(2.0 * r_lo, 1.0)
    while total(r_hi) > 2.0 * math.pi:
        r_hi *= 1.5
    for _ in range(100):                   # bisection: total() is monotonically decreasing in r
        mid = 0.5 * (r_lo + r_hi)
        if total(mid) > 2.0 * math.pi:
            r_lo = mid
        else:
            r_hi = mid
    return r_hi


def build_layout(
    pair_map: Sequence[int],
    tilt_angle: float = UMBRELLA_TILT,
    twist: float = HELIX_TWIST,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[bool]]:
    """Place every nucleotide with correct bond lengths, spreading branches into 3D.

    Bond lengths are a hard constraint of the construction, not an emergent property:

      * Each stem is an antiparallel ladder. Consecutive nucleotides on a strand are
        spaced by exactly BB_TARGET and the two strands of a pair by PAIR_TARGET.
      * Each loop is a circle whose radius is solved (`_solve_loop_radius`) so that
        stepping around it advances one BB_TARGET chord per backbone bond and one
        PAIR_TARGET chord per closing base pair. The stem's distal pair and each
        child stem sit on this circle as chords; children then grow outward. Because
        the perimeter equals the contour length, every adjacent pair of nucleotides
        ends up the intended distance apart -- nothing is crushed together (which
        would cause exponential FENE blow-up) or over-stretched.

    A loop drawn flat would have to overlap its own branches (the bond-correct radius
    is small), so each child subtree is tilted out of the loop plane by rotating it
    rigidly about its closing-pair axis. That axis passes through the two beads that
    stay on the loop circle, so the tilt changes no bond or pair distance at all; it
    only swings the branch into 3D. Tilts alternate sign between siblings, so adjacent
    branches separate to opposite sides and the structure opens up like a 3D tree.
    oxDNA's relaxation then twists the ladders into B-form helices, the mutual traps
    hold the pairs, and excluded volume settles any residual clashes.

    Each subtree is built in a local frame (stem axis +x, outer-pair midpoint at the
    origin, ladder in the xy-plane); the parent applies a 3D rotation+translation,
    carrying the a1/a3 frame vectors with it.
    """
    n = len(pair_map)
    half = PAIR_TARGET / 2.0
    # Axial rise keeping the backbone bond at BB_TARGET for this twist (twist 0 -> flat).
    rise = math.sqrt(max(BB_TARGET ** 2 - (PAIR_TARGET * math.sin(twist / 2.0)) ** 2, 1e-6))

    # Each cloud entry: idx -> [pos(3,), a1(3,) or None, a3(3,) or None]. Paired bases
    # get frames here; unpaired frames are assigned later from geometry.
    def build_subtree(i: int, j: int) -> dict:
        # Extend the stack inward while it keeps stacking (bulges/loops break it).
        stem: List[Tuple[int, int]] = []
        a, b = i, j
        while True:
            stem.append((a, b))
            na, nb = a + 1, b - 1
            if na < nb and pair_map[na] == nb:
                a, b = na, nb
            else:
                break

        # Antiparallel B-form helix about the local +x axis: each base pair's rung is
        # rotated by HELIX_TWIST per step, the strands sit at radius half on opposite
        # sides, and the axial rise keeps the backbone bond at BB_TARGET. The proximal
        # pair (k=0) keeps its rung along +y so it still attaches to the parent loop.
        cloud: dict = {}
        x_axis = np.array([1.0, 0.0, 0.0])
        for k, (left, right) in enumerate(stem):
            angle = k * twist
            rung = np.array([0.0, math.cos(angle), math.sin(angle)])  # u_0 = +y
            axis_point = np.array([k * rise, 0.0, 0.0])
            # a1 points inward (toward the axis / partner); a3 along the stacking axis.
            cloud[left] = [axis_point + half * rung, -rung, x_axis]
            cloud[right] = [axis_point - half * rung, rung, -x_axis]

        a_in, b_in = stem[-1]
        last = len(stem) - 1
        distal_axis = last * rise
        distal_rung = np.array([0.0, math.cos(last * twist), math.sin(last * twist)])
        _place_loop(cloud, a_in, b_in, distal_axis, distal_rung)
        return cloud

    def _merge(cloud: dict, child: dict, origin: np.ndarray, rot: np.ndarray) -> None:
        for idx, (xyz, c_a1, c_a3) in child.items():
            cloud[idx] = [
                origin + rot @ xyz,
                (rot @ c_a1) if c_a1 is not None else None,
                (rot @ c_a3) if c_a3 is not None else None,
            ]

    def _place_loop(cloud: dict, a_in: int, b_in: int, distal_axis: float, distal_rung: np.ndarray) -> None:
        lo, hi = a_in + 1, b_in - 1
        if lo > hi:
            return
        elements = _ordered_loop_elements(pair_map, lo, hi)

        # Bonds encountered walking the loop from a_in to b_in (the a_in-b_in closing
        # pair is the remaining chord). Each unpaired base and each child 5' end is
        # reached by a backbone bond; each child also contributes a pair chord.
        n_bb = 1                       # final bond into b_in
        n_pair_loop = 0
        for kind, _, _ in elements:
            n_bb += 1
            if kind == "stem":
                n_pair_loop += 1
        r = _solve_loop_radius(n_bb, n_pair_loop + 1)

        # The loop lies in the plane spanned by the helix axis and the (twisted) distal
        # rung, so it joins smoothly onto the helix end. e_norm is the loop normal.
        e_axis = np.array([1.0, 0.0, 0.0])
        e_rung = distal_rung
        e_norm = np.cross(e_axis, e_rung)

        d_center = math.sqrt(max(r * r - half * half, 0.0))
        center = (distal_axis + d_center) * e_axis
        # a_in sits at distal_axis*e_axis + half*e_rung; walk clockwise to b_in.
        theta = math.atan2(half, -d_center)
        bb_step = _chord_angle(BB_TARGET, r)
        pair_step = _chord_angle(PAIR_TARGET, r)

        def on_circle(t: float) -> np.ndarray:
            return center + r * (math.cos(t) * e_axis + math.sin(t) * e_rung)

        child_index = 0
        for kind, p, q in elements:
            if kind == "unpaired":
                theta -= bb_step
                cloud[p] = [on_circle(theta), None, None]
            else:
                theta -= bb_step                       # backbone bond into child 5' end
                theta_p = theta
                theta -= pair_step                     # across the child's closing pair
                theta_mid = 0.5 * (theta_p + theta)
                radial = math.cos(theta_mid) * e_axis + math.sin(theta_mid) * e_rung
                tangent = -math.sin(theta_mid) * e_axis + math.cos(theta_mid) * e_rung  # closing-pair axis
                # child outer-pair midpoint lies inside the circle by cos(half pair angle)
                origin = center + r * math.cos(pair_step / 2.0) * radial
                # Tilt the child out of the loop plane about its closing-pair (tangent)
                # axis; sign alternates so neighbouring branches go opposite ways.
                tilt = tilt_angle if child_index % 2 == 0 else -tilt_angle
                col_x = math.cos(tilt) * radial + math.sin(tilt) * e_norm   # local +x (stem axis)
                col_y = tangent                                            # local +y (pair direction)
                col_z = -math.sin(tilt) * radial + math.cos(tilt) * e_norm
                rot = np.column_stack([col_x, col_y, col_z])
                _merge(cloud, build_subtree(p, q), origin, rot)
                child_index += 1

    # Exterior loop: a horizontal baseline. Unpaired bases step by BB_TARGET; each
    # top-level stem rises from a PAIR_TARGET-wide foot, tilted alternately +/- out
    # of the baseline plane so neighbouring exterior stems separate in 3D.
    placed: dict = {}
    x = 0.0
    ext_index = 0
    for kind, p, q in _ordered_loop_elements(pair_map, 0, n - 1):
        if kind == "unpaired":
            placed[p] = [np.array([x, 0.0, 0.0]), None, None]
            x += BB_TARGET
        else:
            tilt = tilt_angle if ext_index % 2 == 0 else -tilt_angle
            # local +x -> rise (+y tilted about the foot/x axis); local +y -> -x (foot
            # stays on the z=0 baseline); local +z -> tilted out of plane.
            rot = np.array([
                [0.0, -1.0, 0.0],
                [math.cos(tilt), 0.0, -math.sin(tilt)],
                [math.sin(tilt), 0.0, math.cos(tilt)],
            ])
            origin = np.array([x + half, 0.0, 0.0])
            _merge(placed, build_subtree(p, q), origin, rot)
            x += PAIR_TARGET + BB_TARGET                # past the 3' foot, then a bond gap
            ext_index += 1

    pos = np.zeros((n, 3), dtype=float)
    a1 = np.zeros((n, 3), dtype=float)
    a3 = np.zeros((n, 3), dtype=float)
    placed_flags = [False] * n
    for idx in range(n):
        if idx not in placed:
            pos[idx] = np.array((float(idx) * BB_TARGET, 0.0, 0.0))
            placed_flags[idx] = True
            continue
        xyz, frame_a1, frame_a3 = placed[idx]
        pos[idx] = xyz
        if frame_a1 is not None:
            a1[idx] = frame_a1
            a3[idx] = frame_a3
        placed_flags[idx] = True

    return pos, a1, a3, placed_flags


def assign_unpaired_frames(
    pos: np.ndarray,
    a1: np.ndarray,
    a3: np.ndarray,
    pair_map: Sequence[int],
) -> None:
    """Fill a1/a3 for unpaired bases; paired bases keep their helix-template frames."""
    n = len(pos)
    for i in range(n):
        if pair_map[i] != -1:
            continue
        prev = max(i - 1, 0)
        nxt = min(i + 1, n - 1)
        a3[i] = unit(pos[nxt] - pos[prev], fallback=np.array((0.0, 0.0, 1.0)))

    # Propagate a1 along the chain so the base plane does not flip abruptly.
    ref = np.array((1.0, 0.0, 0.0))
    for i in range(n):
        if pair_map[i] != -1:
            ref = a1[i]
            continue
        base = a1[i - 1] if i > 0 else ref
        proj = base - np.dot(base, a3[i]) * a3[i]
        if np.linalg.norm(proj) < 1e-6:
            proj = np.cross(a3[i], np.array((1.0, 0.0, 0.0)))
            if np.linalg.norm(proj) < 1e-6:
                proj = np.cross(a3[i], np.array((0.0, 1.0, 0.0)))
        a1[i] = unit(proj)
        ref = a1[i]


def count_threaded_loops(pair_map: Sequence[int], pos: np.ndarray) -> int:
    """Count loops the backbone threads through (a strand passing inside a loop ring).

    For each loop-closing base pair the loop forms a closed ring; the minimal
    surface spanning it is triangulated as a fan from the ring centroid. A strand
    segment that pierces that surface an odd number of times is topologically
    threaded through the loop -- a tangle that no relaxation can undo. Returns the
    number of loops with such a thread (0 means the configuration is clean).
    """
    n = len(pos)
    if n < 4:
        return 0
    seg_p = pos[:-1]
    seg_q = pos[1:]
    seg_d = seg_q - seg_p
    threaded = 0
    for i in range(n):
        j = pair_map[i]
        if j <= i:
            continue
        if i + 1 < j and pair_map[i + 1] == j - 1:
            continue  # interior stem pair, not a loop closer -> degenerate ring
        ring = pos[i:j + 1]
        centroid = ring.mean(axis=0)
        crossings = np.zeros(n - 1, dtype=int)
        m_ring = len(ring)
        for t in range(m_ring):
            a = centroid
            e1 = ring[t] - a
            e2 = ring[(t + 1) % m_ring] - a
            h = np.cross(seg_d, e2)
            det = h @ e1
            ok = np.abs(det) > 1e-12
            inv = np.divide(1.0, det, out=np.zeros_like(det), where=ok)
            s = seg_p - a
            u = inv * (s * h).sum(axis=1)
            qv = np.cross(s, e1)
            v = inv * (seg_d * qv).sum(axis=1)
            tt = inv * (e2 * qv).sum(axis=1)
            hit = ok & (u >= -1e-9) & (u <= 1 + 1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (tt > 1e-9) & (tt < 1 - 1e-9)
            crossings += hit.astype(int)
        mask = np.ones(n - 1, dtype=bool)
        mask[max(i - 1, 0):j + 1] = False        # ignore the loop's own (and entering) segments
        if int(crossings[mask].sum()) % 2 == 1:
            threaded += 1
    return threaded


# Target COM separation when opening interpenetrating branches apart. A clean
# idealized layout bottoms out at ~0.93 su (the i,i+3 spacing), so this cutoff
# never fires on a good structure but comfortably clears the deep branch overlaps
# (COM distances of 0.15-0.45) that large multiloop topologies can produce.
STERIC_CLEAR = 0.75


def _close_nonbonded_pairs(pos: np.ndarray, pair_map: Sequence[int],
                           cutoff: float, sep: int = 2) -> List[Tuple[int, int]]:
    """Non-bonded COM pairs (i<j, |i-j| > sep, not base partners) closer than cutoff."""
    n = len(pos)
    try:
        from scipy.spatial import cKDTree
        candidates = cKDTree(pos).query_pairs(cutoff)
    except Exception:                              # numpy fallback, no scipy needed
        diff = pos[:, None, :] - pos[None, :, :]
        dd = np.sqrt((diff * diff).sum(axis=2))
        iu = np.triu_indices(n, k=1)
        candidates = [(int(a), int(b)) for a, b in zip(*iu) if dd[a, b] < cutoff]
    out = []
    for i, j in candidates:
        if abs(i - j) <= sep or pair_map[i] == j:
            continue
        out.append((min(i, j), max(i, j)))
    return out


def _penetration(pos: np.ndarray, pair_map: Sequence[int], cutoff: float) -> float:
    """Total steric overlap depth over all colliding non-bonded COM pairs."""
    return sum(cutoff - float(np.linalg.norm(pos[i] - pos[j]))
               for i, j in _close_nonbonded_pairs(pos, pair_map, cutoff))


def _rotate_block(pos: np.ndarray, a1: np.ndarray, a3: np.ndarray,
                  i: int, j: int, angle: float):
    """Rigidly rotate the contiguous block [i..j] about the axis through pos[i],pos[j].

    For a nested structure the base pair (i,j) bounds exactly the indices i..j, and
    both endpoints lie on the rotation axis, so this preserves every backbone bond
    and base-pair distance in the whole molecule (including the external bonds to
    i-1 and j+1, whose anchor beads i and j do not move). Orientation frames inside
    the block are carried along so the helices stay rigid."""
    axis = pos[j] - pos[i]
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return pos, a1, a3
    axis = axis / norm
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    R = np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)
    npos, na1, na3 = pos.copy(), a1.copy(), a3.copy()
    sl = slice(i, j + 1)
    npos[sl] = (R @ (pos[sl] - pos[i]).T).T + pos[i]
    na1[sl] = (R @ a1[sl].T).T
    na3[sl] = (R @ a3[sl].T).T
    return npos, na1, na3


def resolve_branch_collisions(pair_map: Sequence[int], pos: np.ndarray,
                              a1: np.ndarray, a3: np.ndarray,
                              cutoff: float = STERIC_CLEAR, max_passes: int = 200):
    """Open interpenetrating branches apart with topology-preserving hinge rotations.

    The idealized tree has exact bond/pair distances, but for large multiloop
    structures it can place whole subtrees so close that their backbones overlap --
    a defect the loop-threading test does not catch and which the downstream MC/MD
    then relaxes into a genuine thread or knot. Because the structure is nested,
    every base pair (i,j) bounds the contiguous block [i..j]; rotating that block
    about the i-j axis swings the branch through space while preserving every bond
    and pair distance exactly (see _rotate_block). We greedily apply the hinge+angle
    that most reduces total steric penetration, never accepting a move that creates
    a new threaded loop, until no branch overlaps remain or no move helps."""
    pos, a1, a3 = pos.copy(), a1.copy(), a3.copy()
    pairs = [(i, pair_map[i]) for i in range(len(pos)) if pair_map[i] > i]
    if not pairs:
        return pos, a1, a3
    angles = [math.radians(d) for d in
              (20, -20, 45, -45, 75, -75, 110, -110, 150, -150, 180)]
    base_threads = count_threaded_loops(pair_map, pos)
    for _ in range(max_passes):
        collisions = _close_nonbonded_pairs(pos, pair_map, cutoff)
        if not collisions:
            break
        cur = _penetration(pos, pair_map, cutoff)
        hot = {b for pair in collisions for b in pair}
        # Hinges whose block contains a colliding bead can move it; try the
        # widest-spanning (coarsest) branches first, then finer ones.
        hinges = sorted((p for p in pairs if any(p[0] <= b <= p[1] for b in hot)),
                        key=lambda p: p[1] - p[0], reverse=True)
        best = None
        for (i, j) in hinges:
            for ang in angles:
                tpos, ta1, ta3 = _rotate_block(pos, a1, a3, i, j, ang)
                pen = _penetration(tpos, pair_map, cutoff)
                if pen < cur - 1e-6 and (best is None or pen < best[0]):
                    if count_threaded_loops(pair_map, tpos) <= base_threads:
                        best = (pen, tpos, ta1, ta3)
            if best is not None and best[0] < 0.5 * cur:
                break                              # good enough; commit this pass
        if best is None:
            break                                  # stalled
        _, pos, a1, a3 = best
    return pos, a1, a3


def build_structure(pair_map: Sequence[int], twist: float = HELIX_TWIST) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (positions, a1, a3) for an oxDNA configuration from the pairing map.

    `twist` is the per-base-pair helical twist to seed the stems (B-form ~34.3 deg
    for DNA, A-form ~32.7 deg for RNA); the thread-safety fallbacks scan down from
    it. It only sets the *starting* geometry — MC/MD under the material's
    interaction_type relaxes the stem to its true helical form.

    The static diagram has exact bond and pair distances by construction, so no
    relaxation (which could thread a strand through a loop) is applied. We try a set
    of configurations in order of decreasing realism and take the first that is free
    of topological threads:

      * Phase 1 – full B-form twist (34.3°/bp) at decreasing tilt (30°→20°→10°).
        This is the common case; most structures are thread-free here.
      * Phase 2 – reduced twist scan: when full B-form twist causes threading for
        every tilt, scan downward from 34° to 5° in 1° steps at maximum tilt (30°)
        to find the highest per-base-pair twist that is topology-safe.  Structures
        where the full twist angles a hairpin into the path of a nearby
        backbone strand are resolved here — they receive a visibly helical starting
        geometry (e.g. 27°/bp ≈ 13 bp/turn) rather than a flat ladder.
      * Phase 3 – flat ladders (twist=0) at decreasing tilt, then a fully planar
        (tilt=0, twist=0) layout that is provably thread-free.

    Helical twist is kept whenever it does not introduce a thread (the usual case);
    only the rare structure whose helices would tangle at every tested twist falls
    back to the flat form.

    A thread-free layout can still have whole subtrees interpenetrating in 3D (the
    loop-threading test does not catch two branches passing through each other), so
    a final `resolve_branch_collisions` pass opens any overlapping branches apart
    with bond/pair/topology-preserving hinge rotations before the frames are fixed.
    """
    if not is_nested(pair_map):
        # The development pipeline had a spring-relaxed fallback here that no run ever
        # reached; it is not shipped (never validated). validate_dot_bracket rejects
        # pseudoknots and NUPACK's MFE is always nested, so this is unreachable.
        raise ValueError("pseudoknotted structures are not supported")

    third = UMBRELLA_TILT / 3.0
    pos = a1 = a3 = None

    # Phase 1: full (material) twist at decreasing tilt.
    for tilt in (UMBRELLA_TILT, 2.0 * third, third):
        pos, a1, a3, _ = build_layout(pair_map, tilt_angle=tilt, twist=twist)
        if count_threaded_loops(pair_map, pos) == 0:
            break
    else:
        # Phase 2: scan for the maximum reduced twist that avoids threading.
        # Stepping 1° at a time from just below the full twist down to 5° gives the
        # highest achievable twist (most helix-like) for difficult topologies.
        max_twist_deg = int(math.degrees(twist))
        for twist_deg in range(max_twist_deg - 1, 4, -1):
            pos, a1, a3, _ = build_layout(pair_map, tilt_angle=UMBRELLA_TILT, twist=math.radians(twist_deg))
            if count_threaded_loops(pair_map, pos) == 0:
                break
        else:
            # Phase 3: flat stems (no twist); the final (0°,0°) is the planar
            # guarantee that is always thread-free for a properly nested structure.
            for tilt in (UMBRELLA_TILT, 2.0 * third, third, 0.0):
                pos, a1, a3, _ = build_layout(pair_map, tilt_angle=tilt, twist=0.0)
                if count_threaded_loops(pair_map, pos) == 0:
                    break

    # Open any interpenetrating branches apart (bonds, pairs and threads preserved).
    pos, a1, a3 = resolve_branch_collisions(pair_map, pos, a1, a3)
    assign_unpaired_frames(pos, a1, a3, pair_map)
    return pos, a1, a3
