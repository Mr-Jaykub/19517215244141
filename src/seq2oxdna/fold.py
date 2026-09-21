# SPDX-License-Identifier: GPL-3.0-or-later
"""Materials, NUPACK MFE folding, and dot-bracket parsing / validation.

NUPACK is imported lazily: only records without a given structure need it, so a user
without NUPACK can still build inputs from dot-bracket records.
"""

from __future__ import annotations

import functools
import re
from typing import List, Sequence, Tuple

NUPACK_INSTALL_HINT = (
    "NUPACK is needed to fold sequences that come without a dot-bracket structure, but it "
    "is not installed in this Python environment. NUPACK is free for non-commercial "
    "academic use and cannot be redistributed, so it is not bundled: download it from "
    "https://www.nupack.org and install it as described in the README ('Install NUPACK'). "
    "Records that include a dot-bracket structure do not need NUPACK."
)


class NupackMissingError(RuntimeError):
    """Raised when a fold is requested but NUPACK cannot be imported."""


def _import_nupack():
    try:
        import nupack
    except ImportError as exc:
        raise NupackMissingError(NUPACK_INSTALL_HINT) from exc
    return nupack


def nupack_available() -> bool:
    try:
        _import_nupack()
    except NupackMissingError:
        return False
    return True


def nupack_version() -> str | None:
    """NUPACK's version string, or None if it is not installed."""
    try:
        return _import_nupack().__version__
    except NupackMissingError:
        return None


# Per-material configuration.
#   interaction  -> oxDNA interaction_type. DNA2/RNA2 are the oxDNA2/oxRNA2
#                   salt-dependent models: they are the ONLY interactions that read
#                   salt_concentration, and both are CUDA-capable.
#   thy          -> topology letter written at a T/U position (RNA=U, DNA=T). oxDNA
#                   decode_base() treats T and U identically, so this is for
#                   correctness/readability; the RNA physics comes from interaction.
# The NUPACK parameter set is resolved lazily by nupack_material(); the starting helix
# twist lives in layout.MATERIAL_TWIST.
MATERIALS = {
    "dna": {"interaction": "DNA2", "thy": "T"},
    "rna": {"interaction": "RNA2", "thy": "U"},
}
DEFAULT_MATERIAL = "dna"


@functools.lru_cache(maxsize=None)
def pinned_dna_parameter_set() -> str:
    """NUPACK DNA parameter set with the free energies this package was validated on.

    NUPACK 4.0.2.0 changed what `material='dna'` means: 'dna' -> 'dna04' -> 'dna04.2',
    which rewrites the mismatch and interior-loop free energies and changed 35 of 123
    reference folds in our validation. Its 'dna04.1' has dG identical to 4.0.1.8's
    'dna04' and reproduced every reference DNA fold. 4.0.1.8 has no 'dna04.1', so use
    its 'dna04' there, which is the same set.
    """
    nupack = _import_nupack()
    version = tuple(int(part) for part in re.findall(r"\d+", nupack.__version__)[:3])
    if version < (4, 0, 2):
        return "dna04"
    # From 4.0.2 on, 'dna04' itself means dna04.2, so there is no safe fallback: if
    # dna04.1 cannot be loaded, stop rather than silently fold with other parameters.
    try:
        nupack.Model(material="dna04.1", celsius=37.0)
    except Exception as exc:
        raise RuntimeError(
            f"NUPACK {nupack.__version__} cannot load the pinned DNA parameter set 'dna04.1' "
            f"({exc}). Falling back to 'dna' or 'dna04' would use dna04.2 and change DNA "
            f"folds.") from exc
    return "dna04.1"


def nupack_material(material: str) -> str:
    """NUPACK `material` argument for 'dna' or 'rna' (DNA uses the pinned set)."""
    return pinned_dna_parameter_set() if material == "dna" else "rna"


# --- NUPACK input domains -----------------------------------------------------
# These are HARD limits asserted inside NUPACK's C++ core: a value outside them
# aborts the fold with an opaque `RuntimeError: "Assertion failure"
# (Constants.cc, line 61)`, not a Python error, so it is worth checking first.
NUPACK_SODIUM_RANGE = (0.05, 1.1)
NUPACK_MAGNESIUM_RANGE = (0.0, 0.2)
NUPACK_CELSIUS_RANGE = (0.0, 100.0)

# oxDNA2 (Snodin et al. 2015) fitted its salt dependence over this monovalent range.
# Outside it the model still runs, but it is extrapolation -- advisory, not an error.
OXDNA2_FITTED_SODIUM_RANGE = (0.1, 0.5)

DEFAULT_SODIUM = 1.0
DEFAULT_MAGNESIUM = 0.0


def check_nupack_domain(temp_c=None, sodium=None, magnesium=None, context: str = "") -> None:
    """Raise ValueError if a NUPACK-facing parameter is outside NUPACK's accepted domain.

    Called before every Model(...) construction so that an out-of-range value gives an
    actionable message naming the offending flag and its limit, instead of a C++
    assertion from deep inside NUPACK. `context` names the caller for the message.
    """
    where = f" [{context}]" if context else ""
    for name, val, (lo, hi), unit in (
        ("sodium", sodium, NUPACK_SODIUM_RANGE, "M Na+"),
        ("magnesium", magnesium, NUPACK_MAGNESIUM_RANGE, "M Mg2+"),
        ("temp_c", temp_c, NUPACK_CELSIUS_RANGE, "C"),
    ):
        if val is not None and not (lo <= val <= hi):
            raise ValueError(
                f"{name}={val} {unit} is outside NUPACK's accepted range [{lo}, {hi}]{where}. "
                f"NUPACK enforces this in its C++ core, so this value would abort the fold "
                f"with an opaque assertion failure."
            )



def detect_material(seq: str) -> str | None:
    """Infer 'dna' or 'rna' from base composition.

    A U (and no T) => RNA; a T (and no U) => DNA. A sequence containing both is
    ambiguous and rejected. A sequence with neither (only A/C/G/N) returns None so
    the caller can apply its default material.
    """
    s = seq.strip().upper()
    has_t = "T" in s
    has_u = "U" in s
    if has_u and not has_t:
        return "rna"
    if has_t and not has_u:
        return "dna"
    if has_u and has_t:
        raise ValueError("Sequence mixes T and U; cannot infer material (specify one explicitly)")
    return None


def normalize_sequence(seq: str, material: str = DEFAULT_MATERIAL) -> str:
    """Upper-case a sequence, coerce T/U to the material's alphabet, and validate.

    DNA collapses U->T; RNA collapses T->U. Any other character is an error.
    """
    material = material.lower()
    if material not in MATERIALS:
        raise ValueError(f"Unknown material: {material!r} (expected one of {sorted(MATERIALS)})")
    s = seq.strip().upper()
    if material == "rna":
        s = s.replace("T", "U")
        allowed = {"A", "C", "G", "U"}
    else:
        s = s.replace("U", "T")
        allowed = {"A", "C", "G", "T"}
    invalid = sorted(set(ch for ch in s if ch not in allowed))
    if invalid:
        raise ValueError(f"Sequence contains unsupported bases for {material}: {''.join(invalid)}")
    return s


def dot_bracket_to_mfe(seq: str, material: str = DEFAULT_MATERIAL, temp_c: float = 37.0,
                       concentration: float = 1e-6, sodium: float = DEFAULT_SODIUM,
                       magnesium: float = DEFAULT_MAGNESIUM) -> str:
    """Predict the MFE dot-bracket structure with NUPACK.

    `material` selects the DNA or RNA thermodynamic model; `sodium`/`magnesium`
    (molar) apply NUPACK's salt correction. `seq` must already be normalised to the
    material's alphabet (RNA uses U).
    """
    nupack = _import_nupack()
    check_nupack_domain(temp_c=temp_c, sodium=sodium, magnesium=magnesium,
                        context="dot_bracket_to_mfe")
    model = nupack.Model(material=nupack_material(material), celsius=temp_c,
                         sodium=sodium, magnesium=magnesium)
    strand = nupack.Strand(seq, name="a")
    complex_ = nupack.Complex([strand], name="c")
    tube = nupack.Tube({strand: concentration},
                       complexes=nupack.SetSpec(max_size=1, include=[complex_]), name="t1")

    result = nupack.tube_analysis(
        [tube],
        model=model,
        compute=["mfe"],
        options={"num_sample": 1, "energy_gap": 0.5},
    )
    return str(result[complex_].mfe[0].structure)


def reverse_dot_bracket(dot_bracket: str) -> str:
    """Reverse a dot-bracket string into oxDNA particle order."""
    table = str.maketrans("()[]{}<>", ")(][}{><")
    return dot_bracket.translate(table)[::-1]


def parse_pairs(dot_bracket: str) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Parse a dot-bracket string into pair map and unique base-pair tuples."""
    pair_map = [-1] * len(dot_bracket)
    stacks = {"(": [], "[": [], "{": [], "<": []}
    matching = {
        ")": "(",
        "]": "[",
        "}": "{",
        ">": "<",
    }
    pairs: List[Tuple[int, int]] = []

    for idx, char in enumerate(dot_bracket):
        if char in stacks:
            stacks[char].append(idx)
        elif char in matching:
            opener = matching[char]
            if not stacks[opener]:
                raise ValueError(f"Unbalanced dot-bracket string at position {idx}")
            left = stacks[opener].pop()
            pair_map[left] = idx
            pair_map[idx] = left
            pairs.append((left, idx))
        elif char != ".":
            raise ValueError(f"Unsupported dot-bracket character: {char}")

    for opener, stack in stacks.items():
        if stack:
            raise ValueError(f"Unbalanced dot-bracket string: leftover '{opener}'")

    pairs.sort()
    return pair_map, pairs


def is_nested(pair_map: Sequence[int]) -> bool:
    """Return True if the base pairs form a properly nested (non-pseudoknotted) structure."""
    stack: List[int] = []
    for i, j in enumerate(pair_map):
        if j == -1:
            continue
        if j > i:
            stack.append(j)
        else:  # closing position; the matching opener's partner must be i
            if not stack or stack[-1] != i:
                return False
            stack.pop()
    return not stack


# --- user-supplied dot-bracket structures ------------------------------------------
# A structure given directly skips NUPACK, so it can describe things NUPACK's MFE never
# emits and this package has therefore never built or relaxed. Only what the validated
# path produces is accepted:
#   * nested '(' ')' '.' only -- pseudoknots have no validated layout;
#   * hairpin loops of >= 3 unpaired nt -- the loop-radius solver has only seen those;
#   * canonical pairs for the material -- the traps would hold anything together, but
#     oxDNA can only hydrogen-bond these (oxDNA2 DNA has no G-T wobble).
MIN_HAIRPIN_LOOP = 3
CANONICAL_PAIRS = {
    "dna": {"AT", "TA", "GC", "CG"},
    "rna": {"AU", "UA", "GC", "CG", "GU", "UG"},
}
DOT_BRACKET_CHARS = set(".()")
PSEUDOKNOT_CHARS = set("[]{}<>")


def validate_dot_bracket(sequence: str, dot_bracket: str, material: str) -> str:
    """Check a user-supplied dot-bracket against `sequence` (already normalised to `material`).

    Returns the stripped structure, or raises ValueError naming the first problem with
    1-based positions so the user can find it.
    """
    db = dot_bracket.strip()
    if "&" in db or "+" in db:
        raise ValueError("multi-strand structures ('&' or '+') are not supported; "
                         "give one strand per record")
    if set(db) & PSEUDOKNOT_CHARS:
        raise ValueError("pseudoknots are not supported: use only nested '(' ')' '.' "
                         f"(found {''.join(sorted(set(db) & PSEUDOKNOT_CHARS))})")
    bad = sorted(set(db) - DOT_BRACKET_CHARS)
    if bad:
        raise ValueError(f"unsupported dot-bracket character(s): {''.join(bad)!r}")
    if len(db) != len(sequence):
        raise ValueError(f"structure length {len(db)} does not match sequence length {len(sequence)}")

    stack: List[int] = []
    pairs: List[Tuple[int, int]] = []
    for i, ch in enumerate(db):
        if ch == "(":
            stack.append(i)
        elif ch == ")":
            if not stack:
                raise ValueError(f"unbalanced structure: ')' at position {i + 1} has no matching '('")
            pairs.append((stack.pop(), i))
    if stack:
        raise ValueError(f"unbalanced structure: '(' at position {stack[-1] + 1} is never closed")

    allowed = CANONICAL_PAIRS[material]
    for i, j in sorted(pairs):
        pair = sequence[i] + sequence[j]
        if pair not in allowed:
            hint = ""
            if material == "dna" and pair in ("GT", "TG", "GU", "UG"):
                hint = " (G-T is not a pair in the DNA models; if this is RNA, use U or --material rna)"
            raise ValueError(f"positions {i + 1}-{j + 1} pair {sequence[i]}-{sequence[j]}, which is not "
                             f"a canonical {material.upper()} pair{hint}")
        # A hairpin closing pair encloses no other pair. The structure is balanced and
        # nested, so "no '(' between i and j" means everything between them is '.'.
        inner = db[i + 1:j]
        if "(" not in inner and len(inner) < MIN_HAIRPIN_LOOP:
            raise ValueError(f"hairpin loop closed by positions {i + 1}-{j + 1} has {len(inner)} "
                             f"unpaired nt; at least {MIN_HAIRPIN_LOOP} are required")
    return db
