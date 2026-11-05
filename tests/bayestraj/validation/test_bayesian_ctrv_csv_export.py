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
