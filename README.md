# seq2oxdna

`seq2oxdna` turns a single-stranded DNA or RNA sequence into ready-to-run
[oxDNA](https://github.com/lorenzo-rovigatti/oxDNA) / oxRNA simulation files. The
secondary structure comes from [NUPACK](https://www.nupack.org) (minimum free energy) or
from a dot-bracket structure you supply. Optionally, it then relaxes the molecule with a
restrained Monte Carlo (MC) → molecular dynamics (MD) protocol.

The result is a **physically relaxed oxDNA/oxRNA configuration consistent with the given or
NUPACK-predicted secondary structure**. It is not a 3D structure prediction: the MD is
restrained to the input base pairs by construction, and tertiary contacts are not predicted.

```
sequence ──NUPACK MFE──┐
                       ├─> dot-bracket ─> 3D starting layout ─> oxDNA files ─> MC (CPU) ─> MD (CUDA)
dot-bracket (given) ───┘                                        (prepare)      └──────── run ────────┘
```

## What it does

1. **Structure.** Folds each sequence with NUPACK, or takes the structure you give.
   Material (DNA or RNA) is detected from T/U.
2. **Starting layout.** Builds a 3D starting configuration in which every backbone bond and
   base-pair distance is exact by construction: stems are helices (B-form twist for DNA,
   A-form for RNA), and each loop is a circle sized so its perimeter equals the strand's
   contour length. Branches are tilted into 3D; threaded loops and interpenetrating
   branches are then resolved by rotations that keep every bond and pair length, and any
   that remain are reported (`geom_overlaps`, `geom_threads` in `manifest.csv`).
3. **Restraints.** Writes a `forces.txt` with two reciprocal `mutual_trap` forces per base pair.
4. **Relaxation (`run` only).** MC on the CPU (100 000 steps), then MD on a CUDA GPU
   (1 000 000 steps, dt 0.003) with the traps on, using oxDNA2 (DNA) or oxRNA2 (RNA) with
   sequence-dependent parameters.

## Requirements

| Needed for | Requirement |
|---|---|
| everything | Python 3.11, numpy, scipy, pandas, openpyxl (installed below) |
| records **without** a structure | NUPACK 4.0.2.0 (free for non-commercial academic use; you download it yourself) |
| `seq2oxdna run` | oxDNA built at commit `dbdc7265`, **with CUDA** (MD runs on the GPU) |

A CPU-only machine can run `seq2oxdna prepare`. MD on the CPU is not supported (see
[Validation](#validation-and-limits)).

## Install

**1. The Python environment and seq2oxdna.** From a clone of this repository:

<!-- check_readme: never -->
```bash
conda env create -f environment.yml
conda activate seq2oxdna
```

```bash
python -m pip install .
```

**2. NUPACK** (only needed to fold sequences without a given structure). Register and
download NUPACK 4.0.2.0 from <https://www.nupack.org>, unzip it, and install the wheel that
matches your platform from its `package` folder:

<!-- check_readme: never -->
```bash
python -m pip install --no-index --find-links nupack-4.0.2.0/package nupack==4.0.2.0
```

NUPACK's licence does not allow redistribution, so it is not bundled with seq2oxdna.
seq2oxdna pins NUPACK's DNA parameter set to `dna04.1` (see
[Validation](#validation-and-limits)).

**3. oxDNA** (only needed for `seq2oxdna run`). Build the validated commit; on a machine
with an NVIDIA GPU and the CUDA toolkit, add `--cuda`:

```bash
bash scripts/install_oxdna.sh --jobs 4
export OXDNA_BIN="$PWD/oxDNA/build/bin/oxDNA"
```

seq2oxdna finds oxDNA through `--oxdna-bin`, then `$OXDNA_BIN`, then `oxDNA` on your `PATH`.
The build needs git, cmake and a C++ compiler (tested: gcc 11.4, cmake 3.22, CUDA 12.3).
oxDNA is optimised for the CPU it is built on, so build it on the machine (or cluster node
type) that will run it; a binary copied to a different CPU can stop with "illegal instruction".

## Quick start

The `examples/` folder has four sequences in `examples.fasta` — an 18-nt DNA hairpin from the
oxDNA distribution, the 256-nt DNA nanostructure R10D from
[Rajwar et al. 2026](https://doi.org/10.1002/adma.74852), yeast tRNA-Phe (PDB 1EHZ) and an
adenine riboswitch (PDB 1Y26) — and one record with its own structure, a designed 54-nt DNA
three-way junction, in `examples_given.db`. The dot-bracket record needs no NUPACK:

```bash
seq2oxdna prepare examples/examples_given.db --outdir given_out
```

Folding the sequences needs NUPACK:

<!-- check_readme: local -->
```bash
seq2oxdna prepare examples/examples.fasta --outdir examples_out
```

Full relaxation needs a CUDA build of oxDNA and a GPU:

<!-- check_readme: local -->
```bash
seq2oxdna run examples/examples.fasta --outdir examples_run
```

The inputs that `prepare` builds for the examples are committed in `examples/prepared/`,
so you can inspect them (for example in [oxView](https://sulcgroup.github.io/oxdna-viewer/))
without installing anything.

## Input

One file per invocation: FASTA, plain text (one sequence per line), CSV, TSV or Excel
(`.xlsx`), with or without names. Tables are parsed automatically: the sequence column is
found by its content, a header row is skipped, and `--seq-col` / `--name-col` override the
detection.

A record may carry its own structure, which then replaces NUPACK:

```
>my_hairpin
GCGTTGCTTCTCCAACGC
((((((......))))))
```

That is the ViennaRNA/RNAfold `.db` layout (a trailing energy such as `(-3.40)` is
ignored). In a table, add a column of dot-brackets. Files may mix records with and
without structures.

Accepted structures: nested `(`, `)` and `.` only; canonical pairs (DNA A–T, G–C; RNA also
G–U); hairpin loops of at least 3 nt; one strand per record. Anything else is skipped with
a message naming the position. `--material dna|rna` overrides the automatic detection.

## Output

`--outdir` (default: `<input name>_oxdna/` next to the input) gets one folder per record:

| File | Content |
|---|---|
| `sequence.top`, `sequence.dat` | oxDNA topology and starting configuration (`<name>.dat` is a copy) |
| `forces.txt` | mutual traps, both directions for every base pair |
| `<name>.db` | the structure used |
| `input_mc`, `input_md_gpu` | the rendered oxDNA inputs (`run`) |
| `mc_<name>.conf`, `md_<name>.conf` | configurations after MC and after MD (`run`) |
| `<name>_mc_traj.dat`, `<name>_md_traj.dat`, `energy_m[cd]_<name>.dat` | trajectories and energies (`run`) |

plus `manifest.csv` (one row per record: status, energies, structure, seeds, NUPACK
version) and `run_parameters.json` (all parameters, package versions, the oxDNA build that
ran, and checksums of the shipped templates). The output folder must be empty;
`--overwrite` replaces an earlier seq2oxdna output in it (its record folders, `logs/`,
`manifest.csv`, `run_parameters.json`) and leaves any other files alone.

## Options and defaults

| Option | Default | Meaning |
|---|---|---|
| `--temp-c` | 37 | NUPACK folding temperature, and MC/MD temperature unless overridden |
| `--mc-temp-c`, `--md-temp-c` | `--temp-c` | per-stage oxDNA temperatures (`run`) |
| `--sodium` | 1.0 | Na⁺ (M) for NUPACK and oxDNA |
| `--magnesium` | 0 | Mg²⁺ (M), NUPACK only |
| `--pair-stiffness`, `--pair-rest-length` | 1.0, 1.2 | mutual-trap strength and rest length (oxDNA units) |
| `--mc-steps`, `--md-steps` | 100 000, 1 000 000 | stage lengths (`run`) |
| `--seed` | oxDNA's choice | RNG seed for MC and MD; the seeds used are always recorded |
| `--jobs` | min(12, CPU cores) | records processed in parallel |
| `--gpu-jobs` | 1 | MD stages allowed on the GPU at the same time |

`seq2oxdna prepare --help` and `seq2oxdna run --help` list everything. The schedule used
in our earlier work (MC at 30 °C, MD at 20 °C) is `--mc-temp-c 30 --md-temp-c 20`.

## Validation and limits

- **Healthy results.** For a structured molecule the final MD potential energy is typically
  −0.8 to −1.5 oxDNA units per nucleotide. `manifest.csv` flags failed stages and alerts.
- **Cell-list retries are normal.** Small or compact molecules often overflow oxDNA's CUDA
  cell list; seq2oxdna then retries MD with a larger cell capacity and, if needed, without a
  cell list. Neither retry changes the physics.
- **Tested range.** Our validation runs covered single strands of 15–262 nt (DNA) and
  48–192 nt (RNA).
- **MD needs CUDA.** In a 16-seed-per-backend comparison, CPU MD gave a lower mean energy
  than CUDA MD for an 18-nt DNA hairpin (by 0.037 units per nucleotide; the 90 % confidence
  interval, 0.016–0.060, extends past our pre-set ±0.05 equivalence margin), so only CUDA MD
  is supported.
- **Reproducibility.** With `--seed`, input building and MC are bit-for-bit reproducible;
  CUDA MD is not (runs diverge within ~10⁴ steps), so compare MD results statistically.
- **NUPACK version.** NUPACK 4.0.2.0 changed its default DNA parameters (`dna04.2`), which
  changed 35 of the 123 folds we compared against 4.0.1.8. seq2oxdna uses `dna04.1`, which
  has the same free energies as the parameters our results were validated with.
- **Salt.** The default 1.0 M Na⁺ is outside the 0.1–0.5 M range over which oxDNA2's salt
  dependence was fitted. We keep it because it is the condition we validated and NUPACK's
  reference salt; other values are accepted, and one outside 0.1–0.5 M prints a note.
- **Not modelled:** pseudoknots, complexes of more than one strand, G-quadruplexes and other
  non-canonical pairs. A G-rich sequence can relax to a plausible energy in the wrong fold.
- **Platforms.** <!-- to be completed after the release tests --> Linux with an NVIDIA GPU
  (CUDA 12.3); `prepare` on macOS. Other platforms are untested.

### Run times

Measured on our validation machine (NVIDIA RTX A6000, Intel Xeon w9-3495X) with the examples:

| Step | Time (18–256 nt) |
|---|---|
| `seq2oxdna prepare` | about 1 s per input file, including NUPACK |
| MC, 100 000 steps, one CPU core per record | 4 s (18 nt) to 86 s (256 nt) |
| MD, 1 000 000 steps, one job on the GPU | 1.3–3.3 min per molecule, including cell-list retries |
| `seq2oxdna run examples/examples.fasta` | 9 min in total |

For molecules this small, running several MD jobs at once (`--gpu-jobs`) does not finish
sooner: with four or five sharing the GPU, each MD took about 20 min in our reference runs.

## Citing

If you use seq2oxdna, please cite this repository (see `CITATION.cff`) and the tools it
builds on:

- NUPACK 4: Fornace, Porubsky & Pierce (2020) *ACS Synth. Biol.* 9:2665–2678,
  doi:[10.1021/acssynbio.9b00523](https://doi.org/10.1021/acssynbio.9b00523)
- oxDNA: Poppleton et al. (2023) *J. Open Source Softw.* 8:4693,
  doi:[10.21105/joss.04693](https://doi.org/10.21105/joss.04693)
- oxDNA2: Snodin et al. (2015) *J. Chem. Phys.* 142:234901,
  doi:[10.1063/1.4921957](https://doi.org/10.1063/1.4921957)
- oxRNA: Šulc et al. (2014) *J. Chem. Phys.* 140:235102,
  doi:[10.1063/1.4881424](https://doi.org/10.1063/1.4881424); salt-dependent oxRNA2:
  Matek et al. (2015) *J. Chem. Phys.* 143:243122,
  doi:[10.1063/1.4933066](https://doi.org/10.1063/1.4933066)
- Sequence-dependent oxDNA: Šulc et al. (2012) *J. Chem. Phys.* 137:135101,
  doi:[10.1063/1.4754132](https://doi.org/10.1063/1.4754132)

Sources of the example structures are listed in `NOTICE`.

## Licence

GPL-3.0-or-later (see `LICENSE`). The sequence-dependent parameter files in
`src/seq2oxdna/data/` are copied unchanged from oxDNA and remain under oxDNA's licence,
GPL-3.0 (see `NOTICE`). NUPACK is not included and has its own licence.
