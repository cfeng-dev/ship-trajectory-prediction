"""Project paths stay independent of the launching working directory."""

import importlib
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "relative_path",
    [
        "data/processed/BR24Dataset20131107_with_reference.csv",
        "results/evaluation/bayesian_ctrv_smc.csv",
        "figures/model_comparison/comparison.png",
        "external/IMM/br24_imm.py",
        "src/stan/models/bayesian_ctrv.stan",
    ],
)
def test_project_path_resolves_every_project_area_from_any_working_directory(
    tmp_path, monkeypatch, relative_path
):
    paths = importlib.import_module("bayestraj.paths")
    monkeypatch.chdir(tmp_path)

    resolved = paths.project_path(relative_path)

    assert resolved == (ROOT / relative_path).resolve()
    assert isinstance(resolved, Path)
    assert resolved.is_absolute()
    assert list(tmp_path.iterdir()) == []


def test_project_path_accepts_path_objects_and_normalizes_parent_components():
    paths = importlib.import_module("bayestraj.paths")

    assert paths.project_path(Path("data/processed/../raw")) == ROOT / "data/raw"
    assert paths.project_path() == ROOT


@pytest.mark.parametrize(
    ("script", "name", "relative_path"),
    [
        (
            "scripts/model_evaluation/bayesian_ctrv.py",
            "DATA_FILE",
            "data/processed/BR24Dataset20131107_with_reference.csv",
        ),
        (
            "scripts/model_comparison/compare_bayesian_ctrv_imm.py",
            "BAYESIAN_SMC_CSV",
            "results/evaluation/bayesian_ctrv_smc.csv",
        ),
        (
            "scripts/model_evaluation/imm.py",
            "IMM_PYTHON",
            "external/IMM/br24_imm.py",
        ),
        (
            "scripts/model_comparison/compare_bayesian_ctrv_imm.py",
            "IMM_CSV",
            "results/evaluation/imm.csv",
        ),
        (
            "examples/plot_simulated_trajectory.py",
            "DATA_FILE",
            "data/simulated/example_simulated_trajectory.csv",
        ),
    ],
)
def test_entrypoint_paths_remain_unchanged_from_another_working_directory(
    tmp_path, monkeypatch, script, name, relative_path
):
    monkeypatch.chdir(tmp_path)

    namespace = runpy.run_path(ROOT / script)

    assert namespace[name] == ROOT / relative_path
