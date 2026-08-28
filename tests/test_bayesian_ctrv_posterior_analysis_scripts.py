"""Tests for the six standalone Bayesian CTRV posterior-analysis scripts."""

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PARAMETERS = {
    "plot_initial_speed_prior_update.py": "initial_speed",
    "plot_initial_heading_prior_update.py": "initial_heading",
    "plot_initial_turn_rate_prior_update.py": "initial_turn_rate",
    "plot_position_noise_prior_update.py": "position_observation_noise",
    "plot_speed_process_noise_prior_update.py": "speed_process_noise",
    "plot_turn_rate_process_noise_prior_update.py": "turn_rate_process_noise",
}


def _load_script(filename):
    script_path = PROJECT_ROOT / "experiments" / "posterior_analysis" / filename
    if not script_path.is_file():
        pytest.fail(f"Posterior-analysis script is missing: {script_path}")
    module_name = f"{script_path.stem}_for_tests"
    specification = importlib.util.spec_from_file_location(module_name, script_path)
    if specification is None or specification.loader is None:
        pytest.fail(f"Cannot load posterior-analysis script: {script_path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("filename", "parameter_name"), SCRIPT_PARAMETERS.items())
def test_each_script_opens_only_its_configured_parameter(
    filename,
    parameter_name,
    monkeypatch,
):
    script = _load_script(filename)
    calls = []
    sentinel = object()

    def fake_run(**options):
        calls.append(options)
        return sentinel

    monkeypatch.setattr(
        script.prior_posterior,
        "run_bayesian_ctrv_prior_posterior_analysis",
        fake_run,
    )

    result = script.main(["--no-show"])

    assert result is sentinel
    assert script.PARAMETER_NAME == parameter_name
    assert script.START_INDEX == 0
    assert script.CREDIBLE_INTERVAL == pytest.approx(0.9)
    assert script.SHOW_LEGEND is True
    assert len(calls) == 1
    assert calls[0]["parameter_name"] == parameter_name
    assert calls[0]["show"] is False
    assert calls[0]["show_legend"] is True
    assert calls[0]["inference_method"] in {"vi", "mcmc"}
    assert "observation_counts" not in calls[0]
