"""A single export API preserves a fixed schema and method diagnostics."""

import csv

import pandas as pd
import pytest

import bayestraj.validation.bayesian_ctrv_evaluation as evaluation


@pytest.mark.parametrize(
    ("method", "diagnostics"),
    [
        (
            "mcmc",
            {
                "mcmc_rhat_max": 1.003,
                "mcmc_diagnostics_ok": True,
                "mcmc_divergences": 2,
                "mcmc_ess_bulk_min": 500.0,
                "mcmc_ess_tail_min": 300.0,
                "mcmc_treedepth_hits": 4,
            },
        ),
        (
            "vi",
            {
                "converged": True,
                "vi_termination": "elbo_converged",
                "vi_elbo_delta_mean": 0.08,
                "vi_elbo_delta_median": 0.009,
            },
        ),
        (
            "sequential_vi",
            {
                "sequential_vi_bootstrap_runtime_seconds": 0.5,
                "converged": False,
                "vi_termination": "elbo_converged",
                "vi_elbo_delta_mean": 0.004,
                "vi_elbo_delta_median": 0.003,
            },
        ),
        (
            "smc",
            {
                "particle_effective_sample_size": 50.0,
                "particle_effective_sample_size_fraction": 50.0 / 64,
                "particle_count": 64,
            },
        ),
    ],
)
def test_export_keeps_one_ordered_schema_and_blank_inapplicable_diagnostics(
    tmp_path, method, diagnostics
):
    values = {
        "inference_method": method,
        "time": pd.Timestamp("1970-01-01 13:37:58.5+00:00"),
        "forecast_origin_seconds": 49075.8,
        "horizon_seconds": 2.7,
        "fit_runtime_seconds": 1.25,
        "position_error_m": 2.5,
        "x_mean": 1.0,
        "y_mean": 2.0,
        "point_estimate": "mean",
        "x_mean_route": 1.0,
        "y_mean_route": 2.0,
        **diagnostics,
    }
    table = pd.DataFrame([values])[list(reversed(values))]
    original = table.copy(deep=True)
    path = tmp_path / f"{method}.csv"

    assert evaluation.export_evaluation_csv(table, path)

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == list(evaluation.PREDICTION_CSV_COLUMNS)
        assert "time" not in reader.fieldnames
        assert "window_runtime_seconds" not in reader.fieldnames
        row = next(reader)
    assert row["inference_method"] == method
    assert float(row["position_error_m"]) == 2.5
    assert row["point_estimate"] == "mean"
    assert float(row["forecast_origin_seconds"]) == 49075.8
    assert float(row["horizon_seconds"]) == 2.7
    assert float(row["fit_runtime_seconds"]) == 1.25
    for name, value in (
        ("x_mean", 1.0),
        ("y_mean", 2.0),
        ("x_mean_route", 1.0),
        ("y_mean_route", 2.0),
    ):
        assert float(row[name]) == value
    for name, value in diagnostics.items():
        if isinstance(value, (bool, str)):
            assert row[name] == str(value)
        else:
            assert float(row[name]) == value
    if method != "mcmc":
        assert row["mcmc_rhat_max"] == ""
        for name in (
            "mcmc_divergences",
            "mcmc_ess_bulk_min",
            "mcmc_ess_tail_min",
            "mcmc_treedepth_hits",
        ):
            assert row[name] == ""
    if method not in {"vi", "sequential_vi"}:
        assert row["vi_termination"] == ""
        assert row["vi_elbo_delta_mean"] == ""
        assert row["vi_elbo_delta_median"] == ""
    assert {name for name in row if name.startswith("vi_")} == {
        "vi_termination",
        "vi_elbo_delta_mean",
        "vi_elbo_delta_median",
    }
    if method != "smc":
        assert row["particle_effective_sample_size"] == ""
        assert row["particle_effective_sample_size_fraction"] == ""
        assert row["particle_count"] == ""
    assert "evaluation_start_observation_count" not in row
    assert len(reader.fieldnames) == len(set(reader.fieldnames))
    pd.testing.assert_frame_equal(table, original)


def test_summary_uses_the_same_api_without_the_prediction_schema(tmp_path):
    table = pd.DataFrame({"observation_count": [2, 3], "ade_m": [1.0, 2.0]})
    path = tmp_path / "summary.csv"

    assert evaluation.export_evaluation_csv(table, path, summary=True)

    pd.testing.assert_frame_equal(pd.read_csv(path), table)


def test_export_rejects_unknown_columns_instead_of_discarding_them(tmp_path, capsys):
    path = tmp_path / "metrics.csv"
    original = b"previous,results\n1,2\n"
    path.write_bytes(original)
    table = pd.DataFrame({"inference_method": ["smc"], "new_diagnostic": [0.5]})

    assert not evaluation.export_evaluation_csv(table, path)

    assert path.read_bytes() == original
    assert "new_diagnostic" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == [path]


def test_association_export_preserves_misses_and_integer_detection_ids(tmp_path):
    table = pd.DataFrame(
        [
            {
                "time": 10.0,
                "cycle_counter": 1,
                "detection_id": 7,
                "x": 110.0,
                "y": -20.0,
            },
            {
                "time": 11.0,
                "cycle_counter": 2,
                "detection_id": None,
                "x": None,
                "y": None,
            },
        ]
    )
    original = table.copy(deep=True)
    path = tmp_path / "associations.csv"
    assert hasattr(evaluation, "export_associations_csv")

    assert evaluation.export_associations_csv(table, path)

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["time", "cycle_counter", "detection_id", "x", "y"]
        accepted, missed = list(reader)
    assert accepted["detection_id"] == "7"
    assert float(accepted["x"]) == 110.0
    assert missed["detection_id"] == missed["x"] == missed["y"] == ""
    pd.testing.assert_frame_equal(table, original)


@pytest.mark.parametrize(
    ("metrics", "associations"),
    [
        ("bayesian_ctrv_mcmc_evaluation.csv", "bayesian_ctrv_mcmc_associations.csv"),
        ("bayesian_ctrv_vi_02.csv", "bayesian_ctrv_vi_02_associations.csv"),
        ("metrics.csv", "metrics_associations.csv"),
    ],
)
def test_association_output_path_keeps_method_and_window_identity(
    tmp_path, metrics, associations
):
    assert hasattr(evaluation, "association_csv_path")
    assert (
        evaluation.association_csv_path(tmp_path / metrics) == tmp_path / associations
    )
