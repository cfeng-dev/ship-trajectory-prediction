"""Tests that the simulation package works without sibling project packages."""

import ast
import shutil
import subprocess
import sys
from pathlib import Path

SIMULATION_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "ship_trajectory_prediction"
    / "simulation"
)


def test_simulation_has_no_imports_from_sibling_project_packages():
    """Simulation source may use relative or third-party imports only."""
    for source_file in SIMULATION_SOURCE.glob("*.py"):
        syntax_tree = ast.parse(source_file.read_text(encoding="utf-8"))
        for node in ast.walk(syntax_tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("ship_trajectory_prediction")
            if isinstance(node, ast.Import):
                assert all(
                    not alias.name.startswith("ship_trajectory_prediction")
                    for alias in node.names
                )


def test_copied_simulation_package_can_generate_trajectory(tmp_path):
    """A copied simulation directory should import and run as its own package."""
    copied_package = tmp_path / "simulation"
    shutil.copytree(
        SIMULATION_SOURCE,
        copied_package,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    smoke_test = "\n".join(
        [
            "import sys",
            "sys.modules['tkinter'] = None",
            "from simulation.cli import main",
            "from simulation.paths import default_simulation_data_directory",
            "from simulation.synthetic_ctrv import simulate_synthetic_ctrv_data",
            "data = simulate_synthetic_ctrv_data(count=3)",
            "assert len(data) == 3",
            "assert default_simulation_data_directory().name == 'simulated'",
            "assert callable(main)",
        ]
    )

    subprocess.run(
        [sys.executable, "-c", smoke_test],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
