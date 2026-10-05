# Changelog

## 1.0.0 (2026-10-06)

First public release. The science code is extracted from the development pipeline the
validation was run with (`seq_to_oxdna_pipeline.py` + `nupack_oxdna_pipeline.py`,
source commit `fa3ebdf`). The layout engine, the input templates, the defaults and
the MC -> MD protocol are unchanged; every difference is listed below.

### Changed relative to the development pipeline (packaging only)

- Split into modules: `io` (input parsing), `fold` (materials, NUPACK, dot-bracket
  validation), `layout` (3D starting geometry), `writers` (oxDNA files), `build`
  (one record), `simulate` (MC/MD), `cli`.
- Each record's input files are written straight into its folder (no `_build/` staging
  folder).
- Two commands, `prepare` (inputs only; replaces `--inputs-only`) and `run`. The MC/MD
  options (`--mc-temp-c`, `--md-temp-c`, `--mc-steps`, `--md-steps`, `--seed`, `--gpu-jobs`,
  `--oxdna-bin`) exist only on `run`; `prepare` rejects them.
- `seq2oxdna --version` prints the package version.
- An input file that cannot be read gives a one-line `ERROR reading …` message instead of a
  Python traceback; the exit status is 1 either way. This applies to a file whose content
  cannot be parsed; a file that cannot be opened at all, such as one without read permission,
  still ends in a traceback, as before.
- NUPACK is imported lazily. Records with a given dot-bracket need no NUPACK; folding
  without NUPACK stops with an installation message. No automatic re-launch into
  another Python environment. Without NUPACK (possible only when every structure is given),
  the `nupack_version` column of `manifest.csv` is empty.
- `--dry-run` no longer builds a NUPACK model for each material in the input, so it needs
  no NUPACK. Temperature and salt are still range-checked first, as before.
- oxDNA is found via `--oxdna-bin`, then `$OXDNA_BIN`, then `oxDNA` on `PATH`
  (no built-in path). A relative path in `--oxdna-bin` or `$OXDNA_BIN` is resolved from the
  folder the command is run from, and a bare name is not searched on `PATH`. A leading `~` in
  either is expanded to the home folder. If no oxDNA executable is found, `run` stops before
  any record is processed; previously each record's inputs were still built and the record
  then failed.
- Templates and oxDNA's sequence-dependent parameter files ship as package data. Each
  run folder gets its own copy of the parameter file, referenced by a relative name. If a
  shipped file is missing (a broken installation), seq2oxdna stops with an error instead of
  marking each record failed.
- `--jobs` (default `min(12, CPU cores)`) sets parallel records; `--gpu-jobs` (default 1)
  caps concurrent MD stages on the GPU. Previously `--max-parallel` capped both.
- The output folder must be empty unless `--overwrite` is given. `--overwrite` acts only on
  a folder that holds a previous seq2oxdna output, and removes only what that run wrote
  (its record folders, `logs/`, `manifest.csv`, `run_parameters.json`).
- `run_parameters.json` records versions, parameters, seeds, oxDNA's reported build,
  and sha256 of every shipped data file, with relative paths only.
- Exit status is non-zero if any record is skipped or fails. A `run` record whose status is
  not `done` counts as failed, including `done — HIGH ENERGY` and `done (no energy)`.
- `--inputs-only` records have status `prepared` (was `inputs-only`).
- The oxDNA log header no longer contains absolute paths.
- Messages added: "MD needs an oxDNA built with CUDA" when oxDNA has no CUDA backend, and
  "rebuild oxDNA on this machine" when oxDNA dies with an illegal CPU instruction (SIGILL).
- The oxDNA release and git commit printed at start-up are recorded in `run_parameters.json`,
  with a note when the commit is not the validated `dbdc7265`.
- NUPACK's DNA parameter set is looked up only when a DNA record is folded; otherwise the
  `nupack_dna_parameter_set` manifest column is empty. `validate_materials` builds a NUPACK
  model only for materials that are folded.
- `--jobs`, `--gpu-jobs`, `--mc-steps` and `--md-steps` must be at least 1 (checked before any
  work starts). Advisory messages no longer cite internal notes.
- A record whose input build fails has an empty `interaction_type` in `manifest.csv`.

### Removed

- The G-quadruplex screen (`g4_flag` manifest column). G-quadruplexes are not modelled
  by NUPACK or oxDNA; the README says so.
- The spring-relaxed fallback layout for pseudoknots. It was never reached in any run;
  pseudoknots are rejected by the validator, and a non-nested structure now raises.
- `--md-backend cpu`: CPU MD was not equivalent to CUDA MD in validation, so MD
  requires CUDA.
- `--average-seq`, `--seq-dep`, `--seq-dep-file`: sequence-dependent parameters are
  always used, with the files shipped here.
- `--max-parallel` (see `--jobs` / `--gpu-jobs`).

### Fixed (in the development pipeline too, unless noted)

- When MD failed before writing any energy, the MC energy file was renamed to
  `energy_md_<name>.dat` and reported as the MD energy. The MC energy file is now moved
  aside before MD starts.
- A record named `logs`, `manifest.csv` or `run_parameters.json` took over that entry of the
  output folder (a crash at the end, or other records' logs mixed into its folder). Such
  records now get a suffix (`logs_2`), and names are compared case-insensitively, so records
  such as `hp` and `HP` no longer share a folder on a case-insensitive filesystem (macOS).
  Names that did not collide are unchanged.
- A file already sitting where a record's folder should go aborted the whole run; now only
  that record fails.
- With more than one record running in parallel, `--oxdna-bin` was ignored: the parallel
  workers ran `$OXDNA_BIN` or the development pipeline's built-in oxDNA path instead.
- (seq2oxdna only) `--dry-run` exited 0 even when records were skipped. It now also exits
  non-zero when the input holds no valid record, as a real run does.
