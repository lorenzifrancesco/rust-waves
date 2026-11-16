"""Launcher test: the axial trap written by write_from_experiment.

The kernels add (x / l_harm_x)^2 / 2 (units hbar omega_perp, l_perp), so the launcher must
write l_harm_x = omega_perp / omega_x. Run from the repository root:

    python -m pytest src/processing/launch/test_rw.py
"""
import os
import sys

import pytest
import toml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
from rw import write_from_experiment  # noqa: E402


@pytest.fixture
def at_root(monkeypatch):
    monkeypatch.chdir(ROOT)  # write_from_experiment reads input/_default.toml


@pytest.mark.parametrize("exp", ["exp_ss_pre.toml", "exp_fig2a_pre.toml"])
def test_axial_trap_frequency(at_root, tmp_path, exp):
    ex = toml.load(os.path.join(ROOT, "input", exp))
    out = tmp_path / "params.toml"
    write_from_experiment(os.path.join(ROOT, "input", exp), str(out), "t", load_gs=False)
    l_harm_x = toml.load(out)["physics"]["l_harm_x"]
    # trap frequency seen by the kernel, (omega_perp / l_harm_x)
    omega_sim = ex["omega_perp"] / l_harm_x
    assert omega_sim == pytest.approx(ex["omega_x"], rel=1e-12)


def test_free_axis(at_root, tmp_path):
    out = tmp_path / "params.toml"
    write_from_experiment(os.path.join(ROOT, "input", "exp_ss_pre.toml"), str(out), "t",
                          load_gs=False, free_x=True)
    assert toml.load(out)["physics"]["l_harm_x"] == 1e300
