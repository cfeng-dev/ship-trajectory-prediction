"""Tests for shared trajectory experiment console reporting."""

from ship_trajectory_prediction.evaluation.reporting import print_prediction_setup


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

    assert lines[:3] == ["=" * 60, title, "=" * 60]
    assert "Data file" in value_lines[0]
    assert "trajectory.csv" in value_lines[0]
    assert "Observed positions" in value_lines[2]
    assert "20" in value_lines[2]
    assert "Speed prior median" in value_lines[-1]
    assert "12.50 m/s" in value_lines[-1]
    assert len({line.index(":") for line in value_lines}) == 1


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
