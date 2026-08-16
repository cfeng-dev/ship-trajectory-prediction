"""Tests for shared trajectory experiment console reporting."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ship_trajectory_prediction.validation.reporting import (
    format_aligned_rows,
    format_evaluation_report,
    format_prediction_table,
    posterior_parameter_summary,
    posterior_variable_samples,
    print_prediction_setup,
    print_variational_diagnostics,
    variational_elbo_history,
)


class FakeWindow:
    """Minimal trajectory window used by reporting tests."""

    observation_count = 20
    prediction_count = 5


def test_print_prediction_setup_reports_and_aligns_all_rows(capsys):
    """Shared and model-specific values should use one aligned layout."""
    title = "Bayesian Constant-Turn-Rate Trajectory Prediction"

    print_prediction_setup(
        title,
        data_file="trajectory.csv",
        run_id=1,
        window=FakeWindow(),
        extra_rows=[("Speed prior median", "12.50 m/s")],
    )

    lines = capsys.readouterr().out.splitlines()
    value_lines = lines[3:]

    assert lines[:3] == ["=" * 72, title, "=" * 72]
    assert "Data file" in value_lines[0]
    assert "trajectory.csv" in value_lines[0]
    assert "Observed positions" in value_lines[2]
    assert "20" in value_lines[2]
    assert "Speed prior median" in value_lines[-1]
    assert "12.50 m/s" in value_lines[-1]
    assert len({line.index(":") for line in value_lines}) == 1


def test_compact_prediction_table_and_evaluation_report_fit_terminal():
    """Single-window prediction reports should use the rolling-style layout."""
    table = format_prediction_table(
        pd.DataFrame(
            {
                "horizon_seconds": [10.0],
                "x_actual": [12.3456],
                "y_actual": [-3.4567],
                "x_predicted": [11.9876],
                "y_predicted": [-2.9876],
                "position_error_m": [0.5901],
            }
        ),
        columns=[
            "horizon_seconds",
            "x_actual",
            "y_actual",
            "x_predicted",
            "y_predicted",
            "position_error_m",
        ],
    )
    report = format_evaluation_report(
        [("ADE", "0.59 m"), ("Computation time", "0.004 s")],
        table,
    )

    lines = report.splitlines()
    metric_lines = [line for line in lines if " : " in line]
    assert lines[:3] == [
        "=" * 72,
        "Complete prediction evaluation",
        "=" * 72,
    ]
    assert "Horizon[s]" in report
    assert "x_actual" not in report
    assert len({line.index(":") for line in metric_lines}) == 1
    assert max(map(len, lines)) <= 80


def test_format_aligned_rows_accepts_no_rows():
    """An optional empty diagnostic section should format cleanly."""
    assert format_aligned_rows([]) == ""


def test_print_prediction_setup_expands_separator_for_long_title(capsys):
    """A long title should not extend beyond its separator line."""
    title = "Long trajectory prediction experiment " * 2

    print_prediction_setup(
        title,
        data_file="trajectory.csv",
        run_id=1,
        window=FakeWindow(),
    )

    lines = capsys.readouterr().out.splitlines()

    assert lines[0] == "=" * len(title)
    assert lines[2] == lines[0]


def test_posterior_variable_samples_requests_all_variational_draws():
    """CmdStanVB requires mean=False to expose uncertainty instead of its mean."""
    fit = FakeVariationalFit(curvature=np.array([-0.1, 0.0, 0.2]))

    samples = posterior_variable_samples(fit, "curvature")

    assert samples == pytest.approx([-0.1, 0.0, 0.2])
    assert fit.requested_mean is False


def test_posterior_parameter_summary_uses_variational_draws():
    """VI parameter tables should report quantiles without MCMC-only fields."""
    fit = FakeVariationalFit(curvature=np.array([-1.0, 0.0, 1.0]))

    summary = posterior_parameter_summary(fit, ["curvature"], credible_interval=0.8)

    assert summary.loc["curvature", "Mean"] == pytest.approx(0.0)
    assert summary.loc["curvature", "50%"] == pytest.approx(0.0)
    assert summary.loc["curvature", "10%"] == pytest.approx(-0.8)
    assert summary.loc["curvature", "90%"] == pytest.approx(0.8)


def test_variational_elbo_history_and_console_report(tmp_path, capsys):
    """The CmdStan progress table should become reusable convergence data."""
    stdout_file = tmp_path / "variational-stdout.txt"
    stdout_file.write_text(
        """Begin stochastic gradient ascent.
  iter             ELBO   delta_ELBO_mean   delta_ELBO_med   notes
   100          -20.000             1.000            1.000
   200          -10.000             0.500            0.500
COMPLETED.
""",
        encoding="utf-8",
    )
    fit = FakeVariationalFit(curvature=np.array([0.0]))
    fit.runset = SimpleNamespace(stdout_files=[str(stdout_file)])

    history = variational_elbo_history(fit)
    print_variational_diagnostics(fit)

    assert history["iteration"].tolist() == [100, 200]
    assert history["elbo"].tolist() == [-20.0, -10.0]
    output = capsys.readouterr().out
    diagnostic_lines = [line for line in output.splitlines() if " : " in line]
    assert "Final iteration" in output
    assert "200" in output
    assert "Final ELBO" in output
    assert "-10.000" in output
    assert len({line.index(":") for line in diagnostic_lines}) == 1


class FakeVariationalFit:
    """Small CmdStanVB-like result exposing variational draws."""

    variational_sample = np.empty((0, 0))

    def __init__(self, **variables):
        self.variables = variables
        self.requested_mean = None

    def stan_variable(self, name, *, mean=None):
        """Record whether the caller requested all approximate draws."""
        self.requested_mean = mean
        return self.variables[name]
