# SPDX-License-Identifier: GPL-3.0-or-later
"""Command line: `seq2oxdna prepare` (inputs only) and `seq2oxdna run` (inputs + MC + MD).

Input is a FASTA / plain / CSV / TSV / Excel file of single-stranded DNA or RNA
sequences, with or without names. A record may carry its own dot-bracket structure
(a structure line after the sequence in FASTA/plain files, RNAfold ".db" layout, a
trailing "(-3.40)" energy is ignored; or a dot-bracket column in a table). Records
with a structure skip NUPACK; the others are folded to their NUPACK MFE structure.
Only nested structures with canonical pairs and hairpin loops of at least 3 nt are
accepted.

Output: one folder per record under the output directory, plus manifest.csv and
run_parameters.json.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import platform
import shutil
import sys
from datetime import datetime

from . import __version__
from .fold import (DEFAULT_MAGNESIUM, DEFAULT_SODIUM, NUPACK_INSTALL_HINT,
                   OXDNA2_FITTED_SODIUM_RANGE, check_nupack_domain, nupack_available,
                   nupack_material, nupack_version, pinned_dna_parameter_set)
from .io import read_named_sequences, resolve_tasks, safe_name
from .simulate import (MC_TEMPLATE, MD_TEMPLATE, VALIDATED_OXDNA_COMMIT, data_file_hashes,
                       find_oxdna, process_sequence, summarise, template_value)

OUTPUT_MARKERS = ("manifest.csv", "run_parameters.json")


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("seqfile", help="Input sequences: FASTA, plain text, CSV, TSV or Excel "
                                        "(.xlsx), optionally with dot-bracket structures")
    common.add_argument("--outdir", default=None,
                        help="Output folder (default: <input-stem>_oxdna next to the input)")
    common.add_argument("--overwrite", action="store_true",
                        help="Replace an existing seq2oxdna output folder instead of refusing")
    common.add_argument("--jobs", type=int, default=min(12, multiprocessing.cpu_count()),
                        help="Records processed in parallel (input building and CPU MC); "
                             "default min(12, CPU cores)")
    common.add_argument("--dry-run", action="store_true",
                        help="Only parse the input and list the records; no folding, no files")
    common.add_argument("--material", choices=["auto", "dna", "rna"], default="auto",
                        help="auto-detect per record from U/T (default; A/C/G-only -> DNA), "
                             "or force dna/rna")
    common.add_argument("--temp-c", type=float, default=37.0,
                        help="Temperature (C) for NUPACK folding and, unless overridden, "
                             "for MC and MD (default 37)")
    common.add_argument("--sodium", type=float, default=DEFAULT_SODIUM,
                        help="Na+ (M) for NUPACK's salt correction and oxDNA's "
                             "salt_concentration (default 1.0)")
    common.add_argument("--magnesium", type=float, default=DEFAULT_MAGNESIUM,
                        help="Mg2+ (M) for NUPACK's salt correction only (default 0)")
    common.add_argument("--concentration", type=float, default=1e-6,
                        help="NUPACK strand concentration (M) (default 1e-6)")
    common.add_argument("--pair-stiffness", type=float, default=1.0,
                        help="Mutual-trap stiffness in forces.txt (default 1.0)")
    common.add_argument("--pair-rest-length", type=float, default=1.2,
                        help="Mutual-trap rest length in forces.txt (default 1.2)")
    common.add_argument("--seq-col", type=int, default=None,
                        help="0-based sequence column for CSV/TSV/Excel (overrides auto-detection)")
    common.add_argument("--name-col", type=int, default=None,
                        help="0-based name column for CSV/TSV/Excel (overrides auto-detection)")

    parser = argparse.ArgumentParser(
        prog="seq2oxdna", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"seq2oxdna {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", parents=[common],
                   help="Build oxDNA inputs only (topology, configuration, forces); no oxDNA needed")
    run = sub.add_parser("run", parents=[common],
                         help="Build inputs, then relax: MC on CPU, then MD on a CUDA GPU")
    run.add_argument("--mc-temp-c", type=float, default=None, help="MC temperature (C); default --temp-c")
    run.add_argument("--md-temp-c", type=float, default=None, help="MD temperature (C); default --temp-c")
    run.add_argument("--mc-steps", type=int, default=None,
                     help=f"MC steps (default {template_value(MC_TEMPLATE, 'steps')})")
    run.add_argument("--md-steps", type=int, default=None,
                     help=f"MD steps (default {template_value(MD_TEMPLATE, 'steps')})")
    run.add_argument("--seed", type=int, default=None,
                     help="oxDNA RNG seed for MC and MD (default: oxDNA picks one; the seeds "
                          "used are recorded in manifest.csv). CUDA MD is not bit-reproducible "
                          "even with a fixed seed.")
    run.add_argument("--gpu-jobs", type=int, default=1,
                     help="MD stages allowed on the GPU at once (default 1)")
    run.add_argument("--oxdna-bin", default=None,
                     help="oxDNA executable (default: $OXDNA_BIN, else `oxDNA` on PATH)")
    return parser


def validate_parameters(args) -> None:
    """Fail fast on out-of-domain parameters, before any record is processed."""
    try:
        check_nupack_domain(temp_c=args.temp_c, sodium=args.sodium,
                            magnesium=args.magnesium, context="NUPACK fold")
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")
    if args.jobs < 1 or getattr(args, "gpu_jobs", 1) < 1:
        sys.exit("ERROR: --jobs and --gpu-jobs must be at least 1")
    for flag in ("mc_steps", "md_steps"):
        value = getattr(args, flag, None)
        if value is not None and value < 1:
            sys.exit(f"ERROR: --{flag.replace('_', '-')} must be at least 1")

    # Advisory only, and only for a deliberate excursion: the default (1.0 M) already sits
    # outside oxDNA2's fitted band (see README), so warning on it would fire every run.
    lo, hi = OXDNA2_FITTED_SODIUM_RANGE
    if args.sodium != DEFAULT_SODIUM and not (lo <= args.sodium <= hi):
        print(f"NOTE: --sodium {args.sodium} M is outside oxDNA2's fitted salt range "
              f"[{lo}, {hi}] M; the model will extrapolate.")
    if args.pair_stiffness >= 3.0:
        print(f"NOTE: --pair-stiffness {args.pair_stiffness} is high; stiff traps can prevent "
              f"relaxation. The validated default is 1.0.")


def validate_materials(tasks, args) -> None:
    """Construct a real NUPACK Model for every material that will be folded."""
    from nupack import Model
    for material in sorted({t[3] for t in tasks if t[4] is None}):
        try:
            Model(material=nupack_material(material), celsius=args.temp_c,
                  sodium=args.sodium, magnesium=args.magnesium)
        except Exception as exc:
            sys.exit(f"ERROR: NUPACK rejected the {material.upper()} model with "
                     f"T={args.temp_c} C, Na+={args.sodium} M, "
                     f"Mg2+={args.magnesium} M:\n  {exc}")


def prepare_outdir(outdir: str, overwrite: bool) -> None:
    """Refuse to mix a new run into an existing output folder unless --overwrite."""
    if os.path.exists(outdir) and not os.path.isdir(outdir):
        sys.exit(f"ERROR: output path {outdir} exists and is not a folder")
    if os.path.isdir(outdir) and os.listdir(outdir):
        if not overwrite:
            sys.exit(f"ERROR: output folder {outdir} is not empty. Choose another --outdir, "
                     f"or pass --overwrite to replace it.")
        if not any(os.path.exists(os.path.join(outdir, m)) for m in OUTPUT_MARKERS):
            sys.exit(f"ERROR: {outdir} is not empty and does not look like a seq2oxdna output "
                     f"folder (no manifest.csv or run_parameters.json); refusing to delete it "
                     f"even with --overwrite.")
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)


def _package_versions() -> dict:
    out = {}
    for mod in ("numpy", "scipy", "pandas"):
        try:
            out[mod] = __import__(mod).__version__
        except ImportError:
            out[mod] = None
    return out


def write_run_parameters(outdir, args, cfg, results, nupack_ver, dna_set, oxdna_bin) -> str:
    """Everything needed to reproduce the run; paths are relative, never absolute."""
    releases = {(r.get("oxdna_release"), r.get("oxdna_commit")) for r in results
                if r.get("oxdna_release") or r.get("oxdna_commit")}
    params = {
        "material": args.material, "temp_c": args.temp_c, "sodium": args.sodium,
        "magnesium": args.magnesium, "concentration": args.concentration,
        "pair_stiffness": args.pair_stiffness, "pair_rest_length": args.pair_rest_length,
    }
    if args.command == "run":
        params.update({
            "mc_temp_c": cfg["mc_temp_c"], "md_temp_c": cfg["md_temp_c"],
            "mc_steps": cfg["mc_steps"] or int(template_value(MC_TEMPLATE, "steps")),
            "md_steps": cfg["md_steps"] or int(template_value(MD_TEMPLATE, "steps")),
            "seed": args.seed, "gpu_jobs": args.gpu_jobs, "md_backend": "CUDA",
            "sequence_dependent_parameters": True,
        })
    doc = {
        "seq2oxdna_version": __version__,
        "command": args.command,
        "input_file": os.path.basename(args.seqfile),
        "created": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "packages": _package_versions(),
        "nupack_version": nupack_ver or None,
        "nupack_dna_parameter_set": dna_set or None,
        "oxdna": None if args.command != "run" else {
            "binary": os.path.basename(oxdna_bin),
            "builds_reported": [{"release": rel, "git_commit": com} for rel, com in sorted(
                releases, key=lambda x: (x[0] or "", x[1] or ""))],
            "validated_git_commit": VALIDATED_OXDNA_COMMIT,
        },
        "parameters": params,
        "data_files_sha256": data_file_hashes(),
        "records": {r["name"]: {"structure_source": r.get("structure_source"),
                                "mc_seed": r.get("mc_seed"), "md_seed": r.get("md_seed"),
                                "status": r.get("status")}
                    for r in sorted(results, key=lambda x: x["index"])},
    }
    path = os.path.join(outdir, "run_parameters.json")
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")
    return path


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    validate_parameters(args)
    inputs_only = args.command == "prepare"

    seqfile = os.path.abspath(args.seqfile)
    if not os.path.exists(seqfile):
        sys.exit(f"ERROR: input file not found: {args.seqfile}")
    stem = safe_name(os.path.splitext(os.path.basename(seqfile))[0])
    outdir = (os.path.abspath(args.outdir) if args.outdir
              else os.path.join(os.path.dirname(seqfile), f"{stem}_oxdna"))

    try:
        records = read_named_sequences(seqfile, seq_col=args.seq_col, name_col=args.name_col)
    except (ValueError, ImportError) as exc:
        sys.exit(f"ERROR reading {args.seqfile}: {exc}")
    tasks, skipped = resolve_tasks(records, material_mode=args.material)

    print(f"Input:    {args.seqfile}")
    print(f"Output:   {outdir}")
    print(f"Parsed:   {len(tasks)} record(s)" + (f", skipped {len(skipped)}" if skipped else ""))
    for name, reason in skipped:
        print(f"  SKIP {name}: {reason}")

    if args.dry_run:
        print("\n--dry-run: parsed records (no folding, no files written) ---")
        for index, name, seq, material, structure in tasks:
            source = "given" if structure is not None else "nupack"
            print(f"  [{index}] {name:<24} {material:>3} {len(seq):>4} nt  {source:>6}  "
                  f"{seq[:50]}{'...' if len(seq) > 50 else ''}")
        return

    if not tasks:
        sys.exit("ERROR: no valid records found.")

    needs_fold = any(t[4] is None for t in tasks)
    if needs_fold:
        if not nupack_available():
            sys.exit(f"ERROR: {NUPACK_INSTALL_HINT}")
        validate_materials(tasks, args)
    nupack_ver = nupack_version() or ""
    dna_set = pinned_dna_parameter_set() if nupack_ver else ""

    oxdna_bin = None
    if not inputs_only:
        oxdna_bin = find_oxdna(args.oxdna_bin)
        if not oxdna_bin or not os.access(oxdna_bin, os.X_OK):
            sys.exit("ERROR: oxDNA executable not found. Pass --oxdna-bin, set OXDNA_BIN, or put "
                     "oxDNA on PATH (see scripts/install_oxdna.sh). `seq2oxdna prepare` needs no oxDNA.")

    prepare_outdir(outdir, args.overwrite)
    cfg = {
        "outdir": outdir, "inputs_only": inputs_only,
        "temp_c": args.temp_c, "concentration": args.concentration,
        "sodium": args.sodium, "magnesium": args.magnesium,
        "pair_stiffness": args.pair_stiffness, "pair_rest_length": args.pair_rest_length,
        "mc_temp_c": None, "md_temp_c": None, "mc_steps": None, "md_steps": None,
        "seed": None, "oxdna_bin": oxdna_bin, "md_slots": None,
    }
    if not inputs_only:
        cfg.update({
            "mc_temp_c": args.mc_temp_c if args.mc_temp_c is not None else args.temp_c,
            "md_temp_c": args.md_temp_c if args.md_temp_c is not None else args.temp_c,
            "mc_steps": args.mc_steps, "md_steps": args.md_steps, "seed": args.seed,
        })

    n_parallel = max(1, min(args.jobs, len(tasks)))
    print(f"NUPACK:   " + (f"{nupack_ver} (DNA parameter set {dna_set})" if nupack_ver
                           else "not installed (structures given, not needed)"))
    if oxdna_bin:
        print(f"oxDNA:    {oxdna_bin}")
    print(f"Workers:  {n_parallel} parallel"
          + ("  (prepare: inputs only)" if inputs_only
             else f"  (MC on CPU -> MD on CUDA, at most {args.gpu_jobs} on the GPU at once)"))
    print(f"Started:  {datetime.now()}\n", flush=True)

    if n_parallel == 1:
        results = [process_sequence((*t, cfg)) for t in tasks]
    else:
        # NUPACK initialises OpenMP state in the parent; fork would deadlock, so spawn.
        ctx = multiprocessing.get_context("spawn")
        with ctx.Manager() as manager:
            if not inputs_only:
                cfg["md_slots"] = manager.BoundedSemaphore(args.gpu_jobs)
            with ctx.Pool(processes=n_parallel) as pool:
                results = list(pool.imap_unordered(process_sequence, [(*t, cfg) for t in tasks]))
        cfg["md_slots"] = None

    print(f"\nAll records finished at {datetime.now()}", flush=True)
    summarise(results, outdir, inputs_only, nupack_ver, dna_set)
    write_run_parameters(outdir, args, cfg, results, nupack_ver, dna_set, oxdna_bin)

    commits = {r.get("oxdna_commit") for r in results if r.get("oxdna_commit")}
    if commits and commits != {VALIDATED_OXDNA_COMMIT}:
        print(f"NOTE: oxDNA reports git commit {', '.join(sorted(commits))}; seq2oxdna was "
              f"validated with {VALIDATED_OXDNA_COMMIT} (scripts/install_oxdna.sh builds it).")
    ok_status = "prepared" if inputs_only else "done"
    failed = [r["name"] for r in results if r["status"] != ok_status]
    if failed or skipped:
        sys.stdout.flush()
        sys.exit(f"{len(failed)} record(s) failed, {len(skipped)} skipped: see the log above "
                 f"and manifest.csv.")


if __name__ == "__main__":
    main()
