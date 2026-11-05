"""A single export API preserves a fixed schema and method diagnostics."""

import csv

import pandas as pd
import pytest

import bayestraj.validation.bayesian_ctrv_evaluation as evaluation


@pytest.mark.parametrize(
    ("method", "diagnostics"),
    [
        ("mcmc", {"mcmc_rhat_max": 1.003, "mcmc_diagnostics_ok": True}),
        ("vi", {"converged": True}),
        ("sequential_vi", {"sequential_vi_bootstrap_runtime_seconds": 0.5}),
        ("smc", {"particle_effective_sample_size": 50.0, "particle_count": 64}),
    ],
)
def test_export_keeps_one_ordered_schema_and_blank_inapplicable_diagnostics(
    tmp_path, method, diagnostics
):
    values = {"inference_method": method, "position_error_m": 2.5, **diagnostics}
    table = pd.DataFrame([values])[list(reversed(values))]
    original = table.copy(deep=True)
    path = tmp_path / f"{method}.csv"

    assert evaluation.export_evaluation_csv(table, path)

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == list(evaluation.PREDICTION_CSV_COLUMNS)
        row = next(reader)
    assert row["inference_method"] == method
    assert float(row["position_error_m"]) == 2.5
    for name, value in diagnostics.items():
        if isinstance(value, bool):
            assert row[name] == str(value)
        else:
            assert float(row[name]) == value
    if method != "mcmc":
        assert row["mcmc_rhat_max"] == ""
    if method != "smc":
        assert row["particle_effective_sample_size"] == ""
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
