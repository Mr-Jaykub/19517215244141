# SPDX-License-Identifier: GPL-3.0-or-later
"""The command line, run as a user would (subprocess), with NUPACK hidden where it matters."""

from __future__ import annotations

import csv
import json
import os

import pytest

from conftest import run_cli


@pytest.fixture
def no_nupack_env(tmp_path):
    """Environment in which `import nupack` fails, whether or not NUPACK is installed."""
    blocker = tmp_path / "block_nupack" / "nupack"
    blocker.mkdir(parents=True)
    (blocker / "__init__.py").write_text("raise ImportError('NUPACK hidden for this test')\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(blocker.parent), env.get("PYTHONPATH", "")])
    return env


def _manifest(outdir):
    with open(os.path.join(outdir, "manifest.csv")) as fh:
        return list(csv.DictReader(fh))


def test_prepare_dot_bracket_without_nupack(examples, no_nupack_env):
    out = examples / "out given"
    proc = run_cli("prepare", examples / "examples_given.db", "--outdir", out, env=no_nupack_env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "not installed (structures given, not needed)" in proc.stdout
    rows = _manifest(out)
    assert [(r["name"], r["status"], r["structure_source"], r["nupack_version"]) for r in rows] == [
        ("three_way_dna_54_given", "prepared", "input", "")]
    params = json.loads((out / "run_parameters.json").read_text())
    assert params["command"] == "prepare" and params["nupack_version"] is None
    assert str(examples) not in (out / "run_parameters.json").read_text()  # no absolute paths


def test_sequence_input_without_nupack_fails_with_install_hint(examples, no_nupack_env):
    out = examples / "out"
    proc = run_cli("prepare", examples / "examples.fasta", "--outdir", out, env=no_nupack_env)
    assert proc.returncode != 0
    assert "www.nupack.org" in proc.stderr and "do not need NUPACK" in proc.stderr
    assert not out.exists()  # nothing written before failing


def test_dry_run_writes_nothing(examples):
    out = examples / "out"
    proc = run_cli("prepare", examples / "examples.fasta", "--outdir", out, "--dry-run")
    assert proc.returncode == 0, proc.stderr
    assert "trna_phe_rna_76" in proc.stdout and "rna" in proc.stdout
    assert not out.exists()


def test_output_folder_guard(examples, no_nupack_env):
    out = examples / "out"
    args = ("prepare", examples / "examples_given.db", "--outdir", out)
    assert run_cli(*args, env=no_nupack_env).returncode == 0
    (out / "stale_marker").write_text("from an earlier run")
    proc = run_cli(*args, env=no_nupack_env)
    assert proc.returncode != 0 and "not empty" in proc.stderr
    assert (out / "stale_marker").exists()
    proc = run_cli(*args, "--overwrite", env=no_nupack_env)
    assert proc.returncode == 0, proc.stderr
    assert not (out / "stale_marker").exists()  # replaced, not mixed


def test_overwrite_refuses_foreign_folder(examples, no_nupack_env):
    foreign = examples / "my_data"
    foreign.mkdir()
    (foreign / "precious.txt").write_text("keep me")
    proc = run_cli("prepare", examples / "examples_given.db", "--outdir", foreign, "--overwrite",
                   env=no_nupack_env)
    assert proc.returncode != 0 and "does not look like a seq2oxdna output" in proc.stderr
    assert (foreign / "precious.txt").read_text() == "keep me"


def test_default_output_folder_next_to_input(examples, no_nupack_env):
    proc = run_cli("prepare", examples / "examples_given.db", env=no_nupack_env)
    assert proc.returncode == 0, proc.stderr
    assert (examples / "examples_given_oxdna" / "three_way_dna_54_given" / "sequence.top").exists()


def test_skipped_record_gives_nonzero_exit(tmp_path, no_nupack_env):
    path = tmp_path / "in.db"
    path.write_text(">ok\nGGGAAACCC\n(((...)))\n>pk\nGGGAAACCC\n((([.])))\n")
    proc = run_cli("prepare", path, "--outdir", tmp_path / "out", env=no_nupack_env)
    assert proc.returncode != 0
    assert "SKIP pk: pseudoknots are not supported" in proc.stdout
    assert [r["name"] for r in _manifest(tmp_path / "out")] == ["ok"]


def test_run_without_oxdna_fails_before_writing(examples, no_nupack_env):
    env = dict(no_nupack_env, PATH=os.path.dirname(os.sys.executable))
    env.pop("OXDNA_BIN", None)
    out = examples / "out"
    proc = run_cli("run", examples / "examples_given.db", "--outdir", out, env=env)
    assert proc.returncode != 0 and "oxDNA executable not found" in proc.stderr
    assert not out.exists()


@pytest.mark.parametrize("args,message", [
    (("--sodium", "1.5"), "outside NUPACK's accepted range"),
    (("--jobs", "0"), "must be at least 1"),
])
def test_bad_parameters_fail_fast(examples, args, message):
    proc = run_cli("prepare", examples / "examples_given.db", *args)
    assert proc.returncode != 0 and message in proc.stderr


def test_md_steps_flag_only_on_run(examples):
    proc = run_cli("prepare", examples / "examples_given.db", "--md-steps", "10")
    assert proc.returncode != 0 and "unrecognized arguments" in proc.stderr


def test_version():
    proc = run_cli("--version")
    assert proc.returncode == 0 and proc.stdout.startswith("seq2oxdna ")
