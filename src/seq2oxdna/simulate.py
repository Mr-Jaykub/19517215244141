# SPDX-License-Identifier: GPL-3.0-or-later
"""oxDNA stages: render the input templates, run MC (CPU) then MD (CUDA), write the manifest."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess as sp
from datetime import datetime

import pandas as pd

from .build import build_sequence_inputs

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Templates, shipped as package data. They are the validated base inputs; render_input()
# overrides only the material / temperature / salt / seed / step keys, so every other
# simulation setting stays exactly as validated. The rendered files keep these names.
MC_TEMPLATE = "input_mc"
MD_TEMPLATE = "input_md_gpu"

# oxDNA's sequence-dependent parameter files (shipped as package data, from oxDNA commit
# dbdc7265, GPL-3.0). They are always used: without them oxDNA runs in average-sequence
# mode, where A-U and G-C get the same hydrogen-bond strength and RNA G-U wobbles cannot
# bond at all. Each run folder gets its own copy, referenced by a relative name, so paths
# containing spaces never reach the oxDNA input parser.
SEQ_DEP_FILES = {
    "DNA2": "oxDNA2_sequence_dependent_parameters.txt",
    "RNA2": "rna_sequence_dependent_parameters.txt",
}

# The oxDNA build every published result was validated with.
VALIDATED_OXDNA_COMMIT = "dbdc7265"


def render_input(src_path: str, dst_path: str, overrides: dict) -> None:
    """Copy an oxDNA input template, overriding `key = value` lines from `overrides`.

    Only keys already present in the template are replaced (values are stringified);
    all other lines are preserved verbatim. This keeps template edits surgical:
    interaction_type / T / salt_concentration change per run, nothing else does.
    An override whose value is None deletes that key's line instead.
    """
    remaining = dict(overrides)
    out_lines = []
    with open(src_path) as fh:
        for line in fh:
            m = re.match(r"^(\s*)([A-Za-z0-9_]+)(\s*)=\s*.*$", line)
            if m and m.group(2) in remaining:
                key = m.group(2)
                val = remaining.pop(key)
                if val is not None:
                    out_lines.append(f"{key} = {val}\n")
            else:
                out_lines.append(line)
    # Any override not already in the template is appended (keeps behaviour explicit).
    remaining = {k: v for k, v in remaining.items() if v is not None}
    if remaining:
        if out_lines and not out_lines[-1].endswith("\n"):
            out_lines.append("\n")
        for key, val in remaining.items():
            out_lines.append(f"{key} = {val}\n")
    with open(dst_path, "w") as fh:
        fh.writelines(out_lines)

# Patterns that flag trouble in oxDNA stdout (displacement warnings during MC are
# normal and intentionally NOT included here).
WARNING_PATTERNS = [
    re.compile(r"exponential", re.IGNORECASE),
    re.compile(r"\bnan\b", re.IGNORECASE),
    re.compile(r"\binf\b", re.IGNORECASE),
    re.compile(r"\bERROR\b"),
    re.compile(r"backbone.*force.*exceeded", re.IGNORECASE),
    re.compile(r"energy.*too\s+large", re.IGNORECASE),
    re.compile(r"^WARNING:", re.IGNORECASE),
    re.compile(r"simulation\s+aborted", re.IGNORECASE),
    re.compile(r"bad\s+conf", re.IGNORECASE),
    re.compile(r"out\s+of\s+(memory|bounds)", re.IGNORECASE),
]

def template_value(template: str, key: str) -> str | None:
    """Value of `key = value` in a shipped template, or None."""
    with open(os.path.join(DATA_DIR, template)) as fh:
        for line in fh:
            m = re.match(r"^\s*([A-Za-z0-9_]+)\s*=\s*(.*?)\s*$", line)
            if m and m.group(1) == key:
                return m.group(2)
    return None


def data_file_hashes() -> dict:
    """sha256 of every shipped template and parameter file (for run_parameters.json)."""
    out = {}
    for fname in sorted(os.listdir(DATA_DIR)):
        path = os.path.join(DATA_DIR, fname)
        if os.path.isfile(path):
            with open(path, "rb") as fh:
                out[fname] = hashlib.sha256(fh.read()).hexdigest()
    return out


# ---------------------------------------------------------------------------
# Locating oxDNA
# ---------------------------------------------------------------------------

def find_oxdna(flag: str | None = None) -> str | None:
    """oxDNA executable: --oxdna-bin, else $OXDNA_BIN, else `oxDNA` on PATH."""
    for candidate in (flag, os.environ.get("OXDNA_BIN")):
        if candidate:
            return os.path.abspath(os.path.expanduser(candidate))
    return shutil.which("oxDNA")


# ---------------------------------------------------------------------------
# oxDNA execution
# ---------------------------------------------------------------------------

SEED_PATTERN = re.compile(r"seeding the RNG with (-?\d+)")
RELEASE_PATTERN = re.compile(r"RELEASE:\s*(\S+)")
COMMIT_PATTERN = re.compile(r"GIT COMMIT:\s*(\S+)")
# What an oxDNA built without CUDA says when an input asks for the CUDA backend.
NO_CUDA_PATTERN = re.compile(r"Backend 'CUDA' not supported|CUDA.*not (supported|compiled)", re.IGNORECASE)


def run_oxdna(oxdna_bin: str, tag: str, input_file: str, folder: str, log_path: str):
    """Run one oxDNA step in `folder`; stream to log; return (ok, alerts, seed).

    oxDNA is started directly with `cwd=folder` rather than through a shell, so paths
    containing spaces or shell metacharacters work. `seed` is the RNG seed oxDNA
    reports using (its own /dev/urandom pick when the input sets none), or None.
    """
    cmd = [oxdna_bin, input_file]
    start = datetime.now()
    alerts = []
    seed = None
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as logf:
        logf.write(f"Command: {os.path.basename(oxdna_bin)} {input_file}\nStarted: {start}\n\n")
        try:
            proc = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.STDOUT, text=True, cwd=folder)
            for line in proc.stdout:
                logf.write(line)
                if seed is None:
                    m = SEED_PATTERN.search(line)
                    if m:
                        seed = int(m.group(1))
                for pat in WARNING_PATTERNS:
                    if pat.search(line):
                        alerts.append(line.rstrip())
                        break
            proc.wait()
            rc = proc.returncode
        except Exception as exc:  # e.g. the oxDNA binary does not exist
            logf.write(f"\nException: {exc}\n")
            print(f"[{tag}] {input_file}: FAILED to start oxDNA: {exc}", flush=True)
            return False, [str(exc)], None
    elapsed = (datetime.now() - start).total_seconds()
    status = "OK" if rc == 0 else f"FAILED(rc={rc})"
    print(f"[{tag}] {input_file}: {status} in {elapsed:.0f}s"
          + (f" ({len(alerts)} alert(s))" if alerts else ""), flush=True)
    if rc == -4:  # SIGILL
        print(f"[{tag}] oxDNA was killed by an illegal CPU instruction: it was compiled for a "
              f"different CPU (scripts/install_oxdna.sh builds for the machine it runs on). "
              f"Rebuild oxDNA on this machine.", flush=True)
    return rc == 0, alerts, seed


def _final_energy_per_nuc(energy_file: str):
    """Last potential-energy-per-nucleotide value from an oxDNA energy file."""
    if not os.path.exists(energy_file):
        return None
    with open(energy_file) as fh:
        lines = [l for l in fh if l.strip()]
    if not lines:
        return None
    try:
        return float(lines[-1].split()[1])
    except (IndexError, ValueError):
        return None


def _move(src: str, dst: str) -> None:
    if os.path.exists(src):
        os.replace(src, dst)


def _log_contains(log_path: str, needle: str) -> bool:
    if not os.path.exists(log_path):
        return False
    with open(log_path) as fh:
        return needle in fh.read()



def _oxdna_build(log_path: str) -> tuple:
    """(release, git commit) that oxDNA printed at start-up, or (None, None)."""
    release = commit = None
    if os.path.exists(log_path):
        with open(log_path) as fh:
            for line in fh:
                release = release or (RELEASE_PATTERN.search(line) or [None, None])[1]
                commit = commit or (COMMIT_PATTERN.search(line) or [None, None])[1]
    return release, commit


# ---------------------------------------------------------------------------
# Per-sequence worker (top-level so it is picklable by the spawn Pool)
# ---------------------------------------------------------------------------

class _NoSlot:
    """Stand-in for the GPU semaphore when only one worker runs."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def process_sequence(task):
    index, name, seq, material, structure, cfg = task
    outdir = cfg["outdir"]
    folder = os.path.join(outdir, name)
    log_base = os.path.join(outdir, "logs", name)

    result = {
        "index": index, "name": name, "length": len(seq), "folder": folder,
        "material": material, "interaction_type": "",
        "dot_bracket": "", "mc_ok": None, "md_ok": None,
        "final_pot_per_nuc": None, "n_alerts": 0, "status": "",
        "geom_overlaps": 0, "geom_threads": 0,
        "structure_source": "input" if structure is not None else "nupack",
        "mc_seed": None, "md_seed": None,
        "md_backend": "cuda" if not cfg["inputs_only"] else "",
        "oxdna_release": None, "oxdna_commit": None,
    }

    # 1. structure -> oxDNA inputs ---------------------------------------------
    try:
        # Inside the try: a non-folder already at this path (e.g. a stray file) must fail this
        # record only, not abort the whole worker pool.
        os.makedirs(folder, exist_ok=True)
        res = build_sequence_inputs(
            name=name, sequence=seq, folder=folder,
            temp_c=cfg["temp_c"], concentration=cfg["concentration"],
            stiffness=cfg["pair_stiffness"], rest_length=cfg["pair_rest_length"],
            material=material, sodium=cfg["sodium"], magnesium=cfg["magnesium"],
            dot_bracket=structure,
        )
        result["interaction_type"] = res.interaction_type
        result["dot_bracket"] = res.dot_bracket
        result["geom_overlaps"] = res.geom_overlaps
        result["geom_threads"] = res.geom_threads
        geom_note = ""
        if res.geom_overlaps or res.geom_threads:
            geom_note = (f"  [GEOM WARN: {res.geom_overlaps} overlap(s), "
                         f"{res.geom_threads} thread(s)]")
        label = "MFE" if res.structure_source == "nupack" else "given"
        print(f"[{name}] inputs ready ({len(seq)} nt, {material}/{res.interaction_type})  "
              f"{label}: {res.dot_bracket}{geom_note}", flush=True)
    except Exception as exc:
        result["status"] = f"input-build failed: {exc}"
        print(f"[{name}] ERROR building inputs: {exc}", flush=True)
        return result

    if cfg["inputs_only"]:
        result["status"] = "prepared"
        return result

    # Render MC and MD inputs for this material: override interaction_type, per-stage
    # temperature and salt; every other validated key is preserved from the template.
    # Sequence-dependent parameters are always on, in both stages, so MC and MD relax
    # under the same force field.
    sd_name = SEQ_DEP_FILES[res.interaction_type]
    shutil.copy(os.path.join(DATA_DIR, sd_name), os.path.join(folder, sd_name))
    stage_overrides = {
        MC_TEMPLATE: {"interaction_type": res.interaction_type,
                      "T": f"{cfg['mc_temp_c']}C", "salt_concentration": cfg["sodium"],
                      "use_average_seq": "false", "seq_dep_file": sd_name},
        MD_TEMPLATE: {"interaction_type": res.interaction_type,
                      "T": f"{cfg['md_temp_c']}C", "salt_concentration": cfg["sodium"],
                      "use_average_seq": "false", "seq_dep_file": sd_name},
    }
    if cfg.get("mc_steps") is not None:
        stage_overrides[MC_TEMPLATE]["steps"] = cfg["mc_steps"]
    if cfg.get("md_steps") is not None:
        stage_overrides[MD_TEMPLATE]["steps"] = cfg["md_steps"]
    if cfg.get("seed") is not None:
        # Same seed for both stages: they are separate processes with separate RNG
        # streams, so reusing it only makes the whole run reproducible.
        for tmpl in (MC_TEMPLATE, MD_TEMPLATE):
            stage_overrides[tmpl]["seed"] = cfg["seed"]
    for tmpl in (MC_TEMPLATE, MD_TEMPLATE):
        render_input(os.path.join(DATA_DIR, tmpl), os.path.join(folder, tmpl), stage_overrides[tmpl])

    oxdna_bin = cfg["oxdna_bin"]
    all_alerts = []

    # 2. MC (CPU) --------------------------------------------------------------
    mc_log = os.path.join(log_base, "input_mc.log")
    mc_ok, alerts, result["mc_seed"] = run_oxdna(oxdna_bin, name, MC_TEMPLATE, folder, mc_log)
    result["oxdna_release"], result["oxdna_commit"] = _oxdna_build(mc_log)
    all_alerts += alerts
    result["mc_ok"] = mc_ok
    # Move (not copy) the MC energy aside: if MD then fails before writing its own
    # energy.dat, a copy left behind would be renamed to energy_md_* and reported as the
    # MD energy of a run that never happened.
    _move(os.path.join(folder, "energy.dat"), os.path.join(folder, f"energy_mc_{name}.dat"))
    _move(os.path.join(folder, "mc_traj.dat"), os.path.join(folder, f"{name}_mc_traj.dat"))
    if not mc_ok:
        result["status"] = "MC failed"
        result["n_alerts"] = len(all_alerts)
        return result

    # 3. MD (CUDA) — reads mc.conf, so rename confs only afterwards ------------
    # At most cfg["gpu_jobs"] MD stages run at once (the semaphore), however many
    # workers build inputs and run MC in parallel.
    md_log = os.path.join(log_base, "input_md_gpu.log")
    with (cfg.get("md_slots") or _NoSlot()):
        md_ok, alerts, result["md_seed"] = run_oxdna(oxdna_bin, name, MD_TEMPLATE, folder, md_log)
        # A compact fold in a generous box can overflow oxDNA's cell list
        # ("_max_n_per_cell"): with few particles spread over a large box the
        # auto-computed per-cell cap is tiny, and once the structure compacts one cell
        # exceeds it. This hits both very short molecules and larger RNAs that fold
        # tightly. Retry with auto-optimisation off AND a much larger
        # max_density_multiplier (the remedy oxDNA's own error message recommends).
        # That cannot help a molecule of only a few dozen nucleotides: oxDNA clamps the
        # per-cell cap to the particle count (CUDASimpleVerletList.cu:138-140), so the
        # overflow can recur however large the multiplier. The second retry therefore drops
        # the cell list altogether (CUDA_list = no: all-pairs forces, which cannot overflow and
        # are cheap for small systems). Neither retry changes the physics.
        retries = [
            ("cells_auto_optimisation=false, max_density_multiplier=50",
             {"cells_auto_optimisation": "false", "max_density_multiplier": 50}, "input_md_gpu_retry.log"),
            ("CUDA_list=no (no cell list)", {"CUDA_list": "no"}, "input_md_gpu_retry2.log"),
        ]
        last_log = md_log
        for label, overrides, log_name in retries:
            if md_ok or not _log_contains(last_log, "_max_n_per_cell"):
                break
            print(f"[{name}] MD hit cell-list overflow; retrying with {label}", flush=True)
            md_path = os.path.join(folder, MD_TEMPLATE)
            render_input(md_path, md_path, overrides)
            last_log = os.path.join(log_base, log_name)
            md_ok, retry_alerts, retry_seed = run_oxdna(oxdna_bin, name, MD_TEMPLATE, folder, last_log)
            result["md_seed"] = retry_seed if retry_seed is not None else result["md_seed"]
            alerts = retry_alerts if md_ok else alerts + retry_alerts
    if not md_ok and _log_matches(last_log, NO_CUDA_PATTERN):
        print(f"[{name}] MD needs an oxDNA built with CUDA (scripts/install_oxdna.sh --cuda); "
              f"this oxDNA has no CUDA backend. CPU MD is not supported by seq2oxdna.", flush=True)
    all_alerts += alerts
    result["md_ok"] = md_ok
    _move(os.path.join(folder, "mc.conf"), os.path.join(folder, f"mc_{name}.conf"))
    _move(os.path.join(folder, "md.conf"), os.path.join(folder, f"md_{name}.conf"))
    _move(os.path.join(folder, "md_traj.dat"), os.path.join(folder, f"{name}_md_traj.dat"))
    _move(os.path.join(folder, "energy.dat"), os.path.join(folder, f"energy_md_{name}.dat"))

    pot = _final_energy_per_nuc(os.path.join(folder, f"energy_md_{name}.dat"))
    result["final_pot_per_nuc"] = pot
    result["n_alerts"] = len(all_alerts)
    if not md_ok:
        result["status"] = "MD failed"
    elif pot is None:
        result["status"] = "done (no energy)"
    elif abs(pot) > 100:
        result["status"] = "done — HIGH ENERGY"
    else:
        result["status"] = "done"
    return result


def _log_matches(log_path: str, pattern) -> bool:
    if not os.path.exists(log_path):
        return False
    with open(log_path) as fh:
        return bool(pattern.search(fh.read()))


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def write_manifest(results, outdir: str, nupack_version: str, dna_parameter_set: str) -> str:
    rows = sorted(results, key=lambda r: r["index"])
    df = pd.DataFrame([{
        "index": r["index"], "name": r["name"], "length_nt": r["length"],
        "material": r.get("material", ""), "interaction_type": r.get("interaction_type", ""),
        "geom_overlaps": r["geom_overlaps"], "geom_threads": r["geom_threads"],
        "mc_ok": r["mc_ok"], "md_ok": r["md_ok"],
        "final_pot_per_nuc": r["final_pot_per_nuc"], "n_alerts": r["n_alerts"],
        "status": r["status"], "dot_bracket": r["dot_bracket"],
        "structure_source": r.get("structure_source", ""),
        "mc_seed": r.get("mc_seed"), "md_seed": r.get("md_seed"), "md_backend": r.get("md_backend", ""),
        "nupack_version": nupack_version, "nupack_dna_parameter_set": dna_parameter_set,
    } for r in rows])
    path = os.path.join(outdir, "manifest.csv")
    df.to_csv(path, index=False)
    return path


def summarise(results, outdir: str, inputs_only: bool, nupack_version: str,
              dna_parameter_set: str) -> str:
    print("\n" + "=" * 64)
    print("SUMMARY" + ("  (prepare: inputs only)" if inputs_only else "  (MC on CPU + MD on CUDA)"))
    print("=" * 64)
    for r in sorted(results, key=lambda x: x["index"]):
        energy = f"{r['final_pot_per_nuc']:.2f}/nuc" if r["final_pot_per_nuc"] is not None else "—"
        flag = "  <-- CHECK" if (r["final_pot_per_nuc"] is not None and abs(r["final_pot_per_nuc"]) > 100) else ""
        print(f"  {r['name']:<24} {r['status']:<22} E={energy}{flag}")
    manifest = write_manifest(results, outdir, nupack_version, dna_parameter_set)
    print(f"\nManifest: {manifest}")
    print("=" * 64)
    return manifest
