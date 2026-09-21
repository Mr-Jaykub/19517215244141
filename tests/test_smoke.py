# SPDX-License-Identifier: GPL-3.0-or-later
"""Short runs of the real oxDNA binary.

`oxdna` tests need any oxDNA build ($OXDNA_BIN or on PATH): MC runs on the CPU for real.
On a CPU-only build MD fails with a clear "needs CUDA" message, which is also checked.
`cuda` tests need a CUDA build and a GPU (set SEQ2OXDNA_CUDA_TESTS=1).
"""

from __future__ import annotations

import csv
import math
import os

import pytest

from seq2oxdna.simulate import find_oxdna, process_sequence

from conftest import run_cli

HAIRPIN = (0, "hp", "GCGTTGCTTCTCCAACGC", "dna", "((((((......))))))")


def _cfg(outdir, **kw):
    cfg = dict(outdir=str(outdir), inputs_only=False, temp_c=37.0, concentration=1e-6,
               mc_temp_c=37.0, md_temp_c=37.0, mc_steps=3000, md_steps=2000, sodium=1.0,
               magnesium=0.0, pair_stiffness=1.0, pair_rest_length=1.2, seed=11,
               oxdna_bin=find_oxdna(), md_slots=None)
    cfg.update(kw)
    return cfg


def _energies(path):
    with open(path) as fh:
        return [float(line.split()[1]) for line in fh if line.strip()]


@pytest.mark.oxdna
def test_mc_stage_runs_and_is_healthy(spaced_tmp):
    result = process_sequence((*HAIRPIN, _cfg(spaced_tmp)))
    assert result["mc_ok"] is True and result["mc_seed"] == 11
    log = open(spaced_tmp / "logs" / "hp" / "input_mc.log").read()
    assert "ERROR" not in log and "nan" not in log.lower()
    energies = _energies(spaced_tmp / "hp" / "energy_mc_hp.dat")
    assert energies and all(math.isfinite(e) for e in energies) and energies[-1] < 0
    if result["md_ok"]:            # CUDA build
        assert result["status"] == "done" and result["final_pot_per_nuc"] < 0
    else:                          # CPU-only build
        md_log = open(spaced_tmp / "logs" / "hp" / "input_md_gpu.log").read()
        assert "Backend 'CUDA' not supported" in md_log
        assert result["final_pot_per_nuc"] is None


@pytest.mark.oxdna
def test_mc_is_reproducible_with_a_seed(tmp_path):
    for run in ("a", "b"):
        process_sequence((*HAIRPIN, _cfg(tmp_path / run)))
    a = open(tmp_path / "a" / "hp" / "hp_mc_traj.dat", "rb").read()
    b = open(tmp_path / "b" / "hp" / "hp_mc_traj.dat", "rb").read()
    assert a and a == b


@pytest.mark.cuda
@pytest.mark.nupack
def test_run_examples_on_cuda(examples):
    out = examples / "out"
    proc = run_cli("run", examples / "examples.fasta", "--outdir", out, "--mc-steps", "2000",
                   "--md-steps", "20000", "--seed", "5", "--gpu-jobs", "2")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    with open(out / "manifest.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 4
    for r in rows:
        assert r["status"] == "done" and r["n_alerts"] == "0", r
        assert -2.0 < float(r["final_pot_per_nuc"]) < 0, r
        assert r["md_backend"] == "cuda" and r["md_seed"] == "5"
