# Changelog

## 1.0.0 (unreleased)

First public release. The science code is extracted from the development pipeline the
validation was run with (`seq_to_oxdna_pipeline.py` + `nupack_oxdna_pipeline.py`,
source commit `fa3ebdf`). The layout engine, the input templates, the defaults and
the MC -> MD protocol are unchanged; every difference is listed below.

### Changed relative to the development pipeline (packaging only)

- Split into modules: `io` (input parsing), `fold` (materials, NUPACK, dot-bracket
  validation), `layout` (3D starting geometry), `writers` (oxDNA files), `build`
  (one record), `simulate` (MC/MD), `cli`.
- Two commands, `prepare` (inputs only; replaces `--inputs-only`) and `run`.
- NUPACK is imported lazily. Records with a given dot-bracket need no NUPACK; folding
  without NUPACK stops with an installation message. No automatic re-launch into
  another Python environment.
- oxDNA is found via `--oxdna-bin`, then `$OXDNA_BIN`, then `oxDNA` on `PATH`
  (no built-in path).
- Templates and oxDNA's sequence-dependent parameter files ship as package data. Each
  run folder gets its own copy of the parameter file, referenced by a relative name.
- `--jobs` (default `min(12, CPU cores)`) sets parallel records; `--gpu-jobs` (default 1)
  caps concurrent MD stages on the GPU. Previously `--max-parallel` capped both.
- The output folder must be empty unless `--overwrite` is given. `--overwrite` acts only on
  a folder that holds a previous seq2oxdna output, and removes only what that run wrote
  (its record folders, `logs/`, `manifest.csv`, `run_parameters.json`).
- `run_parameters.json` records versions, parameters, seeds, oxDNA's reported build,
  and sha256 of every shipped data file, with relative paths only.
- Exit status is non-zero if any record is skipped or fails.
- `--inputs-only` records have status `prepared` (was `inputs-only`).
- The oxDNA log header no longer contains absolute paths.

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
- (seq2oxdna only) `--dry-run` exited 0 even when records were skipped.
