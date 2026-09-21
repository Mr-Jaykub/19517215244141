# SPDX-License-Identifier: GPL-3.0-or-later
"""Template rendering, oxDNA lookup, the MD retry escalation and the stage bookkeeping.

oxDNA itself is replaced by a fake here; test_smoke.py runs the real binary.
"""

from __future__ import annotations

import os

import pytest

from seq2oxdna import simulate
from seq2oxdna.simulate import (DATA_DIR, MC_TEMPLATE, MD_TEMPLATE, SEQ_DEP_FILES, find_oxdna,
                                process_sequence, render_input, run_oxdna, template_value)


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_render_input_none_deletes_key(tmp_path):
    src = _write(tmp_path, "tmpl", "backend = CUDA\nbackend_precision = mixed\nsteps = 10\n")
    dst = str(tmp_path / "out")
    render_input(src, dst, {"backend": "CPU", "backend_precision": None, "not_there": None, "seed": 7})
    assert open(dst).read() == "backend = CPU\nsteps = 10\nseed = 7\n"


def test_shipped_data_files():
    names = set(os.listdir(DATA_DIR))
    assert {MC_TEMPLATE, MD_TEMPLATE, *SEQ_DEP_FILES.values()} <= names
    assert template_value(MC_TEMPLATE, "sim_type") == "MC"
    assert template_value(MD_TEMPLATE, "backend") == "CUDA"
    assert template_value(MC_TEMPLATE, "steps") == "100000"
    assert template_value(MD_TEMPLATE, "steps") == "1000000"
    assert template_value(MD_TEMPLATE, "dt") == "0.003"


def test_find_oxdna_order(tmp_path, monkeypatch):
    fake = tmp_path / "bin" / "oxDNA"
    fake.parent.mkdir()
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake.parent))
    monkeypatch.delenv("OXDNA_BIN", raising=False)
    assert find_oxdna() == str(fake)
    monkeypatch.setenv("OXDNA_BIN", "/env/oxDNA")
    assert find_oxdna() == "/env/oxDNA"
    assert find_oxdna("/flag/oxDNA") == "/flag/oxDNA"
    monkeypatch.delenv("OXDNA_BIN")
    monkeypatch.setenv("PATH", str(tmp_path))
    assert find_oxdna() is None


def test_run_oxdna_missing_binary_fails_cleanly(tmp_path):
    ok, alerts, seed = run_oxdna(str(tmp_path / "no such oxDNA"), "t", "input_mc", str(tmp_path),
                                 str(tmp_path / "logs" / "x.log"))
    assert ok is False and alerts and seed is None


def _cfg(tmp_path, **kw):
    cfg = dict(outdir=str(tmp_path), inputs_only=False, temp_c=37.0, concentration=1e-6,
               mc_temp_c=37.0, md_temp_c=37.0, mc_steps=None, md_steps=None, sodium=1.0,
               magnesium=0.0, pair_stiffness=1.0, pair_rest_length=1.2, seed=101,
               oxdna_bin="oxDNA", md_slots=None)
    cfg.update(kw)
    return cfg


def _fake_oxdna(calls, md_failures, md_error, write_md_energy=True):
    """A stand-in for run_oxdna. MD fails the first `md_failures` times with `md_error`."""
    def fake(oxdna_bin, tag, input_file, folder, log_path):
        calls.append((input_file, os.path.basename(log_path)))
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        md_attempt = sum(1 for f, _ in calls if f == MD_TEMPLATE)
        fail = input_file == MD_TEMPLATE and md_attempt <= md_failures
        with open(log_path, "w") as fh:
            fh.write(md_error if fail else "INFO: RELEASE: v3.6.1\nINFO: GIT COMMIT: dbdc7265\n")
        if not fail and (input_file == MC_TEMPLATE or write_md_energy):
            with open(os.path.join(folder, "energy.dat"), "w") as fh:
                fh.write("0 -1.1 0.3 -0.8\n")
        return (not fail), (["ERROR"] if fail else []), 101
    return fake


def test_md_cell_list_overflow_escalates_to_cuda_list_no(tmp_path, monkeypatch):
    """Overflow on the first MD try and on the multiplier retry must end in CUDA_list = no."""
    calls = []
    monkeypatch.setattr(simulate, "run_oxdna", _fake_oxdna(
        calls, 2, "ERROR: A cell contains more than _max_n_per_cell (18) particles\n"))
    result = process_sequence((0, "hp", "GCGTTGCTTCTCCAACGC", "dna", "((((((......))))))", _cfg(tmp_path)))

    assert [log for _, log in calls] == ["input_mc.log", "input_md_gpu.log",
                                         "input_md_gpu_retry.log", "input_md_gpu_retry2.log"]
    assert result["md_ok"] is True and result["status"] == "done" and result["n_alerts"] == 0
    assert (result["oxdna_release"], result["oxdna_commit"]) == ("v3.6.1", "dbdc7265")
    rendered = open(tmp_path / "hp" / MD_TEMPLATE).read()
    assert "CUDA_list = no" in rendered and "max_density_multiplier = 50" in rendered


def test_rendered_inputs_use_local_seq_dep_copy(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(simulate, "run_oxdna", _fake_oxdna(calls, 0, ""))
    cfg = _cfg(tmp_path, mc_temp_c=30.0, md_temp_c=20.0, mc_steps=500, md_steps=700)
    result = process_sequence((0, "r", "GGGAAAUCC", "rna", "(((...)))", cfg))
    folder = tmp_path / "r"
    sd = SEQ_DEP_FILES["RNA2"]
    assert (folder / sd).read_bytes() == open(os.path.join(DATA_DIR, sd), "rb").read()
    for tmpl, temp, steps in ((MC_TEMPLATE, "30.0C", "500"), (MD_TEMPLATE, "20.0C", "700")):
        text = (folder / tmpl).read_text()
        for line in ("interaction_type = RNA2", f"T = {temp}", "salt_concentration = 1.0",
                     "use_average_seq = false", f"seq_dep_file = {sd}", "seed = 101",
                     f"steps = {steps}"):
            assert line + "\n" in text, (tmpl, line)
    assert "backend = CUDA" in (folder / MD_TEMPLATE).read_text()
    assert result["status"] == "done"


def test_failed_md_reports_no_energy(tmp_path, monkeypatch):
    """A failed MD must not inherit the MC energy file (a bug in the development pipeline)."""
    calls = []
    monkeypatch.setattr(simulate, "run_oxdna", _fake_oxdna(calls, 1, "ERROR: Backend 'CUDA' not supported\n"))
    result = process_sequence((0, "hp", "GCGTTGCTTCTCCAACGC", "dna", "((((((......))))))", _cfg(tmp_path)))
    assert result["status"] == "MD failed" and result["final_pot_per_nuc"] is None
    assert (tmp_path / "hp" / "energy_mc_hp.dat").exists()
    assert not (tmp_path / "hp" / "energy_md_hp.dat").exists()
    assert [log for _, log in calls] == ["input_mc.log", "input_md_gpu.log"]  # no pointless retry


def test_prepare_only_writes_no_simulation_files(tmp_path):
    result = process_sequence((0, "hp", "GCGTTGCTTCTCCAACGC", "dna", "((((((......))))))",
                               _cfg(tmp_path, inputs_only=True)))
    assert result["status"] == "prepared"
    assert sorted(os.listdir(tmp_path / "hp")) == ["forces.txt", "hp.dat", "hp.db",
                                                   "sequence.dat", "sequence.top"]


@pytest.mark.parametrize("bad", ["GGGAAACCC"])
def test_build_failure_is_reported_not_raised(tmp_path, bad):
    result = process_sequence((0, "x", bad, "dna", "((([.])))", _cfg(tmp_path, inputs_only=True)))
    assert result["status"].startswith("input-build failed")


def test_illegal_instruction_gets_a_rebuild_hint(tmp_path, capsys):
    """A binary built for another CPU dies with SIGILL (rc -4); say so instead of just failing."""
    fake = tmp_path / "oxDNA"
    fake.write_text("#!/bin/sh\nkill -ILL $$\n")
    fake.chmod(0o755)
    ok, _alerts, _seed = run_oxdna(str(fake), "t", "input_mc", str(tmp_path), str(tmp_path / "logs" / "x.log"))
    assert ok is False
    assert "illegal CPU instruction" in capsys.readouterr().out


def test_stray_file_at_record_path_fails_only_that_record(tmp_path):
    (tmp_path / "hp").write_text("not a folder")
    result = process_sequence((0, "hp", "GCGTTGCTTCTCCAACGC", "dna", "((((((......))))))",
                               _cfg(tmp_path, inputs_only=True)))
    assert result["status"].startswith("input-build failed")
