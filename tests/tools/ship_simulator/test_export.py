"""Simulator exports retain observed positions and evaluation-only truth."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from ship_simulator import io as simulation_io
from ship_simulator.time_format import format_elapsed_time


@pytest.mark.parametrize(
    ("seconds", "expected"),
    ((20.0, "20.0 s"), (80.0, "1 min 20 s"), (3_735.0, "1 h 02 min 15 s")),
)
def test_status_time_uses_adaptive_elapsed_format(seconds, expected):
    assert format_elapsed_time(seconds) == expected


def test_csv_export_preserves_numeric_source_time_and_local_positions(tmp_path):
    simulator = SimpleNamespace(
        x_all=[0.0, 10.0],
        y_all=[0.0, -5.0],
        t_all=[0.0, 0.0004],
        v_all=[2.2, 2.4],
        theta_all=[np.pi + 0.2, -np.pi - 0.2],
        omega_all=[0.1, -0.2],
    )
    data = simulation_io.create_simulation_dataframe(simulator, interval_seconds=0.0004)
    assert data.columns.tolist() == [
        "time",
        "x",
        "y",
        "ground_truth_x",
        "ground_truth_y",
        "ground_truth_velocity",
        "ground_truth_yaw",
        "ground_truth_turn_rate",
    ]
    np.testing.assert_array_equal(data["ground_truth_x"], data["x"])
    np.testing.assert_array_equal(data["ground_truth_y"], data["y"])
    np.testing.assert_array_equal(data["ground_truth_velocity"], simulator.v_all)
    np.testing.assert_allclose(data["ground_truth_yaw"], [-np.pi + 0.2, np.pi - 0.2])
    np.testing.assert_array_equal(data["ground_truth_turn_rate"], simulator.omega_all)
    result = simulation_io.save_trajectory_data(data, tmp_path / "run.csv")
    saved = pd.read_csv(result.output_path, float_precision="round_trip")
    pd.testing.assert_frame_equal(saved, data)
    assert result.appended is False


def test_default_export_selects_original_two_second_samples_without_partial_tail():
    from ship_simulator.core import ShipSimulator

    simulator = ShipSimulator(v=5.0, dt=0.05)
    for _ in range(76):
        simulator.step(omega=0.1, motor_running=True)
    data = simulation_io.create_simulation_dataframe(simulator)
    np.testing.assert_allclose(data["time"], [0.0, 2.0], rtol=0.0, atol=1e-12)
    for column, history in (("x", simulator.x_all), ("y", simulator.y_all)):
        np.testing.assert_array_equal(data[column], np.asarray(history)[[0, 40]])
    previous = data.copy()
    for _ in range(85):
        simulator.step(omega=-0.05, motor_running=True)
    data = simulation_io.create_simulation_dataframe(simulator)
    np.testing.assert_allclose(data["time"], [0.0, 2.0, 4.0, 6.0, 8.0], atol=1e-12)
    pd.testing.assert_frame_equal(data.iloc[:2], previous)
    for column, history in (("x", simulator.x_all), ("y", simulator.y_all)):
        np.testing.assert_array_equal(data[column], np.asarray(history)[::40])
        np.testing.assert_array_equal(data[f"ground_truth_{column}"], data[column])
    np.testing.assert_array_equal(
        data["ground_truth_velocity"], np.asarray(simulator.v_all)[::40]
    )
    np.testing.assert_array_equal(
        data["ground_truth_turn_rate"], np.asarray(simulator.omega_all)[::40]
    )


@pytest.mark.parametrize("interval", [0.0, -2.0, np.nan, np.inf])
def test_export_rejects_invalid_sampling_intervals(interval):
    simulator = SimpleNamespace(t_all=[0.0, 0.05], x_all=[0.0, 1.0], y_all=[0.0, 0.0])
    with pytest.raises(ValueError, match="interval"):
        simulation_io.create_simulation_dataframe(simulator, interval_seconds=interval)


def test_export_rejects_grid_without_matching_source_samples():
    times = np.arange(101) * 0.03
    simulator = SimpleNamespace(
        t_all=times,
        x_all=times,
        y_all=times,
        v_all=np.ones_like(times),
        theta_all=np.zeros_like(times),
        omega_all=np.zeros_like(times),
    )
    with pytest.raises(ValueError, match="align"):
        simulation_io.create_simulation_dataframe(simulator)


@pytest.mark.parametrize("terminal_newline", [False, True])
def test_repeated_save_continues_one_trajectory_without_duplicate_times(
    tmp_path, terminal_newline
):
    path = tmp_path / "run.csv"
    first = pd.DataFrame({"time": [0.0, 0.05], "x": [0.0, 0.25], "y": [0.0, 0.0]})
    later = pd.DataFrame({"time": [0.17, 0.23], "x": [0.85, 1.15], "y": [0.0, 0.1]})
    simulation_io.save_trajectory_data(first, path)
    if not terminal_newline:
        path.write_bytes(path.read_bytes().rstrip(b"\r\n"))
    previous = path.read_bytes()
    result = simulation_io.save_trajectory_data(later, path, append=True)
    assert result.appended is True
    assert path.read_bytes().startswith(previous)
    saved = pd.read_csv(path)
    pd.testing.assert_frame_equal(saved, pd.concat([first, later], ignore_index=True))
    assert saved.columns.tolist() == ["time", "x", "y"]


@pytest.mark.parametrize("reference_first", [False, True])
def test_append_rejects_mixing_plain_and_reference_schemas(tmp_path, reference_first):
    path = tmp_path / "run.csv"
    plain = pd.DataFrame({"time": [0.0], "x": [0.0], "y": [0.0]})
    labeled = plain.assign(
        ground_truth_x=0.0,
        ground_truth_y=0.0,
        ground_truth_velocity=2.2,
        ground_truth_yaw=0.0,
        ground_truth_turn_rate=0.1,
    )
    first, later = (labeled, plain) if reference_first else (plain, labeled)
    simulation_io.save_trajectory_data(first, path)
    previous = path.read_bytes()
    later = later.assign(time=2.0)
    with pytest.raises(ValueError, match="columns|schema"):
        simulation_io.save_trajectory_data(later, path, append=True)
    assert path.read_bytes() == previous


@pytest.mark.parametrize(
    "problem", ["new_run", "overlap", "legacy_schema", "missing_file"]
)
def test_save_rejects_mixing_runs_or_schemas_without_changing_file(tmp_path, problem):
    path = tmp_path / "run.csv"
    data = pd.DataFrame({"time": [0.0, 0.1], "x": [0.0, 0.5], "y": [0.0, 0.0]})
    if problem == "legacy_schema":
        pd.DataFrame({"time": ["2026-01-01"], "run_id": [0]}).to_csv(path, index=False)
    elif problem != "missing_file":
        simulation_io.save_trajectory_data(data, path)
    previous = path.read_bytes() if path.exists() else None
    with pytest.raises((ValueError, FileExistsError)):
        simulation_io.save_trajectory_data(data, path, append=problem != "new_run")
    assert (path.read_bytes() if path.exists() else None) == previous


@pytest.mark.parametrize("times", [[0.0, 0.0], [1.0, 0.0], [0.0, np.nan]])
def test_invalid_simulated_times_are_rejected_before_writing(tmp_path, times):
    path = tmp_path / "run.csv"
    data = pd.DataFrame({"time": times, "x": [0.0, 1.0], "y": [0.0, 0.0]})
    with pytest.raises(ValueError):
        simulation_io.save_trajectory_data(data, path)
    assert not path.exists()


def test_simulator_csv_loads_directly_into_bayesian_predictor(tmp_path):
    from bayesian_predictor import dashboard
    from bayesian_predictor.session import prepare_analysis
    from bayesian_predictor.settings import default_form_values, parse_settings
    from ship_simulator.config import GUIConfig
    from ship_simulator.core import ShipSimulator

    config = GUIConfig()
    simulator = ShipSimulator(v=config.initial_speed, dt=config.simulation_dt)
    for _ in range(201):
        simulator.step(omega=0.1, motor_running=True)
    path = tmp_path / "run.csv"
    simulation_io.save_trajectory_data(
        simulation_io.create_simulation_dataframe(simulator), path
    )
    values = default_form_values()
    values["data"].update(
        data_file=str(path), inference_method="smc", prediction_count="1"
    )
    values["smc"].update(particle_count="32", posterior_draw_count="16")
    trajectory, _, _, load = prepare_analysis(parse_settings(values).analysis)
    np.testing.assert_allclose(trajectory.observed_x, simulator.x_all[::40])
    np.testing.assert_allclose(trajectory.observed_y, simulator.y_all[::40])
    np.testing.assert_allclose(trajectory.reference_time_seconds, [0, 2, 4, 6, 8, 10])
    np.testing.assert_allclose(trajectory.reference_x, simulator.x_all[::40])
    np.testing.assert_allclose(trajectory.reference_y, simulator.y_all[::40])
    np.testing.assert_allclose(trajectory.reference_speed_mps, simulator.v_all[::40])
    np.testing.assert_allclose(
        trajectory.reference_heading_degrees, np.rad2deg(simulator.theta_all[::40])
    )
    np.testing.assert_allclose(
        trajectory.reference_turn_rate_degrees_per_second,
        np.rad2deg(simulator.omega_all[::40]),
    )
    assert not trajectory.reference_speed_is_derived
    assert not trajectory.reference_heading_is_derived
    assert not trajectory.reference_turn_rate_is_derived
    update = load(3)
    assert update.forecast is not None

    # Deliberately change only labels to detect reference leakage into inference.
    labeled = pd.read_csv(path)
    labeled["ground_truth_x"] += 1000.0
    labeled["ground_truth_y"] -= 500.0
    labeled["ground_truth_velocity"] += 1.0
    labeled["ground_truth_yaw"] += 0.2
    labeled["ground_truth_turn_rate"] += 0.1
    labeled.to_csv(path, index=False)
    changed_trajectory, _, _, changed_load = prepare_analysis(
        parse_settings(values).analysis
    )
    changed_update = changed_load(3)
    np.testing.assert_allclose(
        changed_trajectory.reference_x, trajectory.reference_x + 1000.0
    )
    np.testing.assert_array_equal(changed_trajectory.observed_x, trajectory.observed_x)
    np.testing.assert_array_equal(changed_trajectory.observed_y, trajectory.observed_y)
    for name in update.samples_by_parameter:
        np.testing.assert_array_equal(
            changed_update.samples_by_parameter[name], update.samples_by_parameter[name]
        )
    np.testing.assert_array_equal(
        changed_update.forecast.sample_positions, update.forecast.sample_positions
    )
    before = dashboard.analysis_metrics_at(trajectory, {3: update}, 3)
    after = dashboard.analysis_metrics_at(changed_trajectory, {3: changed_update}, 3)
    assert after.ade_m > before.ade_m + 100.0


def test_save_dialog_continues_current_run_and_requires_new_file_after_reset(
    tmp_path, monkeypatch
):
    pytest.importorskip("tkinter")
    from ship_simulator import dialogs
    from ship_simulator.core import ShipSimulator

    simulator = ShipSimulator(v=5.0, dt=0.05)
    for _ in range(2):
        simulator.step(0.0, True)
    path = tmp_path / "run.csv"
    messages = []
    monkeypatch.setattr(simulation_io, "DATA_DIR", tmp_path)
    monkeypatch.setattr(
        dialogs.filedialog, "asksaveasfilename", lambda **kwargs: str(path)
    )
    for name in ("showinfo", "showwarning", "showerror"):
        monkeypatch.setattr(
            dialogs.messagebox, name, lambda *args, **kwargs: messages.append(args)
        )
    gui = SimpleNamespace(
        simulator=simulator,
        pause_simulation=lambda: None,
        default_csv_filename="run.csv",
        saved_run_states={},
        csv_export_interval_seconds=2.0,
    )
    dialogs.save_csv(gui)
    assert len(pd.read_csv(path)) == 1
    previous = path.read_bytes()
    dialogs.save_csv(gui)
    assert path.read_bytes() == previous
    assert messages[-1][0] == "No New Data"
    for _ in range(38):
        simulator.step(0.0, True)
    dialogs.save_csv(gui)
    assert path.read_bytes() == previous
    assert messages[-1][0] == "No New Data"
    simulator.step(0.0, True)
    dialogs.save_csv(gui)
    saved = pd.read_csv(path)
    np.testing.assert_allclose(saved["time"], [0.0, 2.0])
    assert gui.saved_run_states[path.resolve()] == 2
    previous = path.read_bytes()
    simulator.reset()
    gui.saved_run_states.clear()
    simulator.step(0.0, True)
    dialogs.save_csv(gui)
    assert messages[-1][0] == "Save Failed"
    assert path.read_bytes() == previous
    path = tmp_path / "next_run.csv"
    dialogs.save_csv(gui)
    assert len(pd.read_csv(path)) == 1
