"""BR24 export scripts produce complete reference-enriched datasets."""

import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parents[1]


def test_br24_tracking_input_export_writes_to_processed_data():
    module = runpy.run_path(
        PROJECT_ROOT / "scripts" / "data_preparation" / "export_br24_dataset20131107.py"
    )

    assert (
        str(module["OUTPUT_CSV"])
        .replace("\\", "/")
        .endswith("data/processed/BR24Dataset20131107.csv")
    )


@pytest.fixture(params=["dataset20131107", "imm_tracked_trajectory"])
def export_run(request, tmp_path, monkeypatch):
    module = runpy.run_path(
        PROJECT_ROOT / "scripts/data_preparation" / f"export_br24_{request.param}.py"
    )
    radar_csv = tmp_path / "radar.csv"
    sensor_csv = tmp_path / "sensor.csv"
    reference_csv = tmp_path / "reference.csv"
    output_csv = tmp_path / "processed" / module["OUTPUT_CSV"].name
    pd.DataFrame(
        {
            "Time": [0.5, 1.5, 2.5],
            "CycleCounter": [1, 2, 3],
            "DetectionID": [1, 2, 3],
            "Range": [10.0, 11.0, 12.0],
            "Azimuth": 0.0,
            "StdRange": 0.5,
            "StdAzimuth": 0.01,
            "XRelative": [10.0, 11.0, 12.0],
            "YRelative": 0.0,
        }
    ).to_csv(radar_csv, index=False)
    pd.DataFrame(
        {
            "Time": [0.5, 1.5, 2.5],
            "CycleCounter": [1, 2, 3],
            "X": 100.0,
            "Y": -20.0,
            "Yaw": 0.0,
        }
    ).to_csv(sensor_csv, index=False)
    pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0, 3.0],
            "x": [0.0, 10.0, 20.0, 30.0],
            "y": [0.0, 2.0, 4.0, 6.0],
            "yaw": 0.1,
            "velocity": 2.0,
        }
    ).to_csv(reference_csv, index=False)

    def export_associated(*, radar_csv, sensor_csv, output_csv):
        # Only the optional association backend is replaced; reference export is real.
        assert radar_csv == tmp_path / "radar.csv"
        assert sensor_csv == tmp_path / "sensor.csv"
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "time": [0.5, 1.5, 2.5],
                "cycle_counter": [1, 2, 3],
                "detection_id": [1, 2, 3],
                "x": [110.0, 111.0, 112.0],
                "y": -20.0,
                "cov_xx": 2.0,
                "cov_xy": 0.3,
                "cov_yy": 1.0,
            }
        ).to_csv(output_csv, index=False)
        return output_csv

    monkeypatch.setattr(
        module["br24"], "export_br24_target_radar_trajectory", export_associated
    )
    for name, value in {
        "RADAR_CSV": radar_csv,
        "SENSOR_CSV": sensor_csv,
        "REFERENCE_CSV": reference_csv,
        "OUTPUT_CSV": output_csv,
    }.items():
        monkeypatch.setitem(module["main"].__globals__, name, value)
    return module["main"], output_csv, reference_csv


def test_br24_export_includes_aligned_reference_in_one_run(export_run):
    main, output_csv, reference_csv = export_run
    main()

    exported = pd.read_csv(output_csv)
    np.testing.assert_array_equal(exported["time"], [0.5, 1.5, 2.5])
    np.testing.assert_allclose(exported["ground_truth_x"], [5.0, 15.0, 25.0])
    np.testing.assert_allclose(exported["ground_truth_y"], [1.0, 3.0, 5.0])
    np.testing.assert_allclose(exported["ground_truth_yaw"], 0.1)
    np.testing.assert_allclose(exported["ground_truth_velocity"], 2.0)
    assert list(output_csv.parent.iterdir()) == [output_csv]

    reference = pd.read_csv(reference_csv)
    reference["x"] += 1000.0
    reference.to_csv(reference_csv, index=False)
    main()
    updated = pd.read_csv(output_csv)
    measurement_columns = exported.columns[
        ~exported.columns.str.startswith("ground_truth_")
    ]
    pd.testing.assert_frame_equal(
        updated[measurement_columns], exported[measurement_columns]
    )
    np.testing.assert_allclose(updated["ground_truth_x"], [1005.0, 1015.0, 1025.0])
    assert list(output_csv.parent.iterdir()) == [output_csv]


@pytest.mark.parametrize("existing", [False, True])
def test_invalid_reference_never_leaves_an_unlabelled_output(export_run, existing):
    main, output_csv, reference_csv = export_run
    if existing:
        main()
        previous = output_csv.read_bytes()
    reference = pd.read_csv(reference_csv)
    reference["time"] += 100.0
    reference.to_csv(reference_csv, index=False)

    with pytest.raises(ValueError, match="outside the ground-truth time range"):
        main()

    if existing:
        assert output_csv.read_bytes() == previous
        assert list(output_csv.parent.iterdir()) == [output_csv]
    else:
        assert not output_csv.exists()
        assert list(output_csv.parent.iterdir()) == []
