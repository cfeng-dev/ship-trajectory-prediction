"""BR24 radar association matching the local CV/CTRV/CA IMM reference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

ASSOCIATION_GATE_NIS = float(chi2.ppf(0.99, df=2))
_AZIMUTH_STANDARD_DEVIATION_INFLATION_RAD = np.deg2rad(0.5)
_EGO_POSITION_VARIANCE = 2.0 * 0.25**2 / 2.5
_EGO_POSITION_COVARIANCE = np.eye(2) * _EGO_POSITION_VARIANCE
_TRANSITION_PROBABILITIES = np.array(
    [[0.95, 0.025, 0.025], [0.025, 0.95, 0.025], [0.1, 0.1, 0.8]],
    dtype=float,
)
_H_CV = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
_H_CTRV = np.array(
    [[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]]
)
_H_CA = np.array(
    [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]
)
_OUTPUT_COLUMNS = (
    "time",
    "cycle_counter",
    "detection_id",
    "x",
    "y",
    "cov_xx",
    "cov_xy",
    "cov_yy",
)


@dataclass(frozen=True, slots=True)
class BR24Association:
    """Nearest radar detection accepted by the Reuter 99% NIS gate."""

    detection_index: int
    nis: float


@dataclass(slots=True)
class _IMMState:
    """CV, CTRV, and CA states required for Reuter's next association step."""

    cv_state: np.ndarray
    cv_covariance: np.ndarray
    ctrv_state: np.ndarray
    ctrv_covariance: np.ndarray
    ca_state: np.ndarray
    ca_covariance: np.ndarray
    mode_probabilities: np.ndarray


def polar_to_cartesian(range_m, azimuth_rad) -> np.ndarray:
    """Form the sensor-relative radar positions used as MATLAB ``Z_Det``."""
    ranges, azimuths = np.broadcast_arrays(
        np.asarray(range_m, dtype=float),
        np.asarray(azimuth_rad, dtype=float),
    )
    if not np.all(np.isfinite(ranges)) or not np.all(np.isfinite(azimuths)):
        raise ValueError("Radar range and azimuth must be finite.")
    if np.any(ranges < 0.0):
        raise ValueError("Radar range must be non-negative.")
    return np.column_stack((ranges * np.cos(azimuths), ranges * np.sin(azimuths)))


def position_measurement_noise(
    *,
    predicted_position: np.ndarray,
    range_variance: float,
    azimuth_variance: float,
) -> np.ndarray:
    """Return Reuter's Cartesian ``positionMeasurementNoise`` covariance."""
    position = np.asarray(predicted_position, dtype=float)
    if position.shape != (2,) or not np.all(np.isfinite(position)):
        raise ValueError("predicted_position must be a finite two-dimensional vector.")
    range_variance = _non_negative_finite(range_variance, "range_variance")
    azimuth_variance = _non_negative_finite(azimuth_variance, "azimuth_variance")
    squared_range = float(position @ position)
    if squared_range <= 0.0:
        raise ValueError("predicted_position must have a non-zero range.")

    cos_twice_azimuth = (position[0] ** 2 - position[1] ** 2) / squared_range
    sin_twice_azimuth = 2.0 * position[0] * position[1] / squared_range
    radial_minus_tangential = range_variance - squared_range * azimuth_variance
    diagonal_base = 0.5 * (range_variance + squared_range * azimuth_variance)
    covariance = np.array(
        [
            [
                diagonal_base + 0.5 * cos_twice_azimuth * radial_minus_tangential,
                0.5 * sin_twice_azimuth * radial_minus_tangential,
            ],
            [
                0.5 * sin_twice_azimuth * radial_minus_tangential,
                diagonal_base - 0.5 * cos_twice_azimuth * radial_minus_tangential,
            ],
        ]
    )
    return 0.5 * (covariance + covariance.T)


def associate_br24_detections(
    *,
    predicted_position: np.ndarray,
    innovation_covariance: np.ndarray,
    detections: np.ndarray,
) -> BR24Association | None:
    """Apply Reuter's nearest-neighbour Mahalanobis selection and strict gate."""
    predicted_position = np.asarray(predicted_position, dtype=float)
    covariance = np.asarray(innovation_covariance, dtype=float)
    detections = np.asarray(detections, dtype=float)
    if predicted_position.shape != (2,) or not np.all(np.isfinite(predicted_position)):
        raise ValueError("predicted_position must be a finite two-dimensional vector.")
    if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
        raise ValueError("innovation_covariance must be a finite 2x2 matrix.")
    if detections.ndim != 2 or detections.shape[1:] != (2,):
        raise ValueError("detections must have shape (count, 2).")
    if detections.size == 0:
        return None
    if not np.all(np.isfinite(detections)):
        raise ValueError("detections must be finite.")

    innovations = detections - predicted_position
    try:
        solved_innovations = np.linalg.solve(covariance, innovations.T).T
    except np.linalg.LinAlgError as error:
        raise ValueError("innovation_covariance must be invertible.") from error
    nis = np.einsum("ni,ni->n", innovations, solved_innovations)
    detection_index = int(np.argmin(nis))
    nearest_nis = float(nis[detection_index])
    if nearest_nis >= ASSOCIATION_GATE_NIS:
        return None
    return BR24Association(detection_index=detection_index, nis=nearest_nis)


def export_br24_target_radar_trajectory(
    *,
    radar_csv: str | Path,
    sensor_csv: str | Path,
    output_csv: str | Path,
) -> Path:
    """Associate BR24 radar detections with Reuter's IMM procedure and export them."""
    radar = _read_radar_csv(radar_csv)
    sensor = _read_sensor_csv(sensor_csv)
    radar_by_cycle = {
        int(cycle): group.sort_values("detection_id").reset_index(drop=True)
        for cycle, group in radar.groupby("cycle_counter", sort=False)
    }
    first_sensor = sensor.iloc[0]
    first_cycle = int(first_sensor["cycle_counter"])
    first_detections = radar_by_cycle.get(first_cycle)
    if first_detections is None or first_detections.empty:
        raise ValueError("The first sensor cycle requires one radar detection.")

    initial_position = polar_to_cartesian(
        first_detections.loc[0, "range"],
        first_detections.loc[0, "azimuth"],
    )[0]
    state = _initialize_imm_state(initial_position)
    exported_rows: list[dict[str, float | int]] = []

    previous_sensor = first_sensor
    for _, current_sensor in sensor.iloc[1:].iterrows():
        time_seconds = float(current_sensor["time"])
        previous_time = float(previous_sensor["time"])
        interval_seconds = time_seconds - previous_time
        if interval_seconds <= 0.0:
            raise ValueError("Sensor timestamps must be strictly increasing.")
        ego_delta = np.array(
            [
                float(current_sensor["x"]) - float(previous_sensor["x"]),
                float(current_sensor["y"]) - float(previous_sensor["y"]),
            ]
        )
        predicted, mixing_probabilities, predicted_mode_probabilities = _predict_imm(
            state,
            interval_seconds,
            ego_delta,
        )
        cycle_counter = int(current_sensor["cycle_counter"])
        detections = radar_by_cycle.get(cycle_counter)
        state, record = _associate_and_update_imm(
            predicted=predicted,
            mixing_probabilities=mixing_probabilities,
            predicted_mode_probabilities=predicted_mode_probabilities,
            detections=detections,
            time_seconds=time_seconds,
            cycle_counter=cycle_counter,
            sensor_position=np.array(
                [float(current_sensor["x"]), float(current_sensor["y"])]
            ),
        )
        if record is not None:
            exported_rows.append(record)
        previous_sensor = current_sensor

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(exported_rows, columns=_OUTPUT_COLUMNS).to_csv(output_path, index=False)
    return output_path


def _read_radar_csv(csv_path: str | Path) -> pd.DataFrame:
    radar = _read_csv_with_columns(
        csv_path,
        required=(
            "time",
            "cyclecounter",
            "detectionid",
            "range",
            "azimuth",
            "stdrange",
            "stdazimuth",
        ),
    )
    radar = radar.rename(
        columns={
            "cyclecounter": "cycle_counter",
            "detectionid": "detection_id",
            "stdrange": "std_range",
            "stdazimuth": "std_azimuth",
        }
    )
    _validate_finite_columns(
        radar,
        ("time", "cycle_counter", "detection_id", "range", "azimuth", "std_range", "std_azimuth"),
        source="Radar",
    )
    _validate_non_negative_columns(radar, ("range", "std_range", "std_azimuth"), source="Radar")
    _validate_integral_column(radar, "cycle_counter", source="Radar")
    _validate_integral_column(radar, "detection_id", source="Radar")
    if radar.duplicated(("cycle_counter", "detection_id")).any():
        raise ValueError("Radar DetectionID values must be unique within each cycle.")
    return radar.sort_values(["cycle_counter", "detection_id"]).reset_index(drop=True)


def _read_sensor_csv(csv_path: str | Path) -> pd.DataFrame:
    sensor = _read_csv_with_columns(
        csv_path,
        required=("time", "cyclecounter", "x", "y"),
    )
    sensor = sensor.rename(columns={"cyclecounter": "cycle_counter"})
    _validate_finite_columns(
        sensor,
        ("time", "cycle_counter", "x", "y"),
        source="Sensor",
    )
    _validate_integral_column(sensor, "cycle_counter", source="Sensor")
    if sensor["cycle_counter"].duplicated().any():
        raise ValueError("Sensor CycleCounter values must be unique.")
    sensor = sensor.sort_values("cycle_counter").reset_index(drop=True)
    if sensor.empty:
        raise ValueError("Sensor CSV must contain at least one cycle.")
    if np.any(np.diff(sensor["time"].to_numpy(dtype=float)) <= 0.0):
        raise ValueError("Sensor timestamps must be strictly increasing.")
    return sensor


def _read_csv_with_columns(csv_path: str | Path, *, required: tuple[str, ...]) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    source = pd.read_csv(csv_path)
    normalized_names = {column.lower(): column for column in source.columns}
    missing = [column for column in required if column not in normalized_names]
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")
    selected = source.loc[:, [normalized_names[column] for column in required]].copy()
    selected.columns = list(required)
    return selected.apply(pd.to_numeric, errors="coerce")


def _validate_finite_columns(data, columns, *, source: str) -> None:
    if not np.all(np.isfinite(data.loc[:, columns].to_numpy(dtype=float))):
        raise ValueError(f"{source} CSV values must be finite.")


def _validate_non_negative_columns(data, columns, *, source: str) -> None:
    if (data.loc[:, columns].to_numpy(dtype=float) < 0.0).any():
        raise ValueError(f"{source} range and standard deviations must be non-negative.")


def _validate_integral_column(data, column, *, source: str) -> None:
    values = data[column].to_numpy(dtype=float)
    if not np.array_equal(values, np.floor(values)):
        raise ValueError(f"{source} {column} values must be integers.")
    data[column] = values.astype(int)


def _initialize_imm_state(initial_position: np.ndarray) -> _IMMState:
    cv_state = np.array([initial_position[0], 0.0, initial_position[1], 0.0])
    ctrv_state = np.array([initial_position[0], initial_position[1], 0.0, 0.0, 0.0])
    ca_state = np.array([initial_position[0], 0.0, 0.0, initial_position[1], 0.0, 0.0])
    return _IMMState(
        cv_state=cv_state,
        cv_covariance=np.diag(np.array([5.0, 3.0, 5.0, 3.0]) ** 2),
        ctrv_state=ctrv_state,
        ctrv_covariance=np.diag(np.array([5.0, 5.0, 25.0, 3.0, 3.0]) ** 2)
        * np.diag([1.0, 1.0, (np.pi / 180.0) ** 2, (np.pi / 180.0) ** 2, 1.0]),
        ca_state=ca_state,
        ca_covariance=np.diag(np.array([5.0, 3.0, 1.0, 5.0, 3.0, 1.0]) ** 2),
        mode_probabilities=np.array([1.0, 0.0, 0.0]),
    )


def _predict_imm(state: _IMMState, interval_seconds: float, ego_delta: np.ndarray):
    mixing_probabilities, predicted_mode_probabilities = _mixing_probability(
        state.mode_probabilities
    )
    mixed_cv = _mix_to_cv(state, mixing_probabilities[:, 0])
    mixed_ctrv = _mix_to_ctrv(state, mixing_probabilities[:, 1])
    mixed_ca = _mix_to_ca(state, mixing_probabilities[:, 2])
    cv = _ego_compensate(
        _predict_cv(*mixed_cv, interval_seconds, 0.01**2, 0.01**2),
        _H_CV,
        ego_delta,
    )
    ctrv = _ego_compensate(
        _predict_ctrv(*mixed_ctrv, interval_seconds, 0.03**2, (3.0 * np.pi / 180.0) ** 2),
        _H_CTRV,
        ego_delta,
    )
    ca = _ego_compensate(
        _predict_ca(*mixed_ca, interval_seconds, 10.0, 0.10**2, 0.10**2),
        _H_CA,
        ego_delta,
    )
    return (cv, ctrv, ca), mixing_probabilities, predicted_mode_probabilities


def _associate_and_update_imm(
    *,
    predicted,
    mixing_probabilities: np.ndarray,
    predicted_mode_probabilities: np.ndarray,
    detections: pd.DataFrame | None,
    time_seconds: float,
    cycle_counter: int,
    sensor_position: np.ndarray,
) -> tuple[_IMMState, dict[str, float | int] | None]:
    cv, ctrv, ca = predicted
    if detections is None or detections.empty:
        return _IMMState(*cv, *ctrv, *ca, predicted_mode_probabilities), None

    std_range = _cycle_standard_deviation(detections, "std_range")
    std_azimuth = _cycle_standard_deviation(detections, "std_azimuth")
    mixed_cv_state, mixed_cv_covariance = _mix_states(
        *_to_cv(*cv),
        *_to_cv(*ctrv),
        *_to_cv(*ca),
        mixing_probabilities[:, 0],
    )
    predicted_position = _H_CV @ mixed_cv_state
    measurement_covariance = position_measurement_noise(
        predicted_position=predicted_position,
        range_variance=std_range**2,
        azimuth_variance=(std_azimuth + _AZIMUTH_STANDARD_DEVIATION_INFLATION_RAD)
        ** 2,
    )
    innovation_covariance = _H_CV @ mixed_cv_covariance @ _H_CV.T + measurement_covariance
    detection_positions = polar_to_cartesian(
        detections["range"].to_numpy(dtype=float),
        detections["azimuth"].to_numpy(dtype=float),
    )
    association = associate_br24_detections(
        predicted_position=predicted_position,
        innovation_covariance=innovation_covariance,
        detections=detection_positions,
    )
    if association is None:
        return _IMMState(*cv, *ctrv, *ca, predicted_mode_probabilities), None

    measurement = detection_positions[association.detection_index]
    cv_updated, cv_innovation_covariance = _update(*cv, measurement, _H_CV, measurement_covariance)
    ctrv_updated, ctrv_innovation_covariance = _update(
        *ctrv, measurement, _H_CTRV, measurement_covariance
    )
    ca_updated, ca_innovation_covariance = _update(*ca, measurement, _H_CA, measurement_covariance)
    likelihoods = np.array(
        [
            _normal_density(measurement, _H_CV @ cv[0], cv_innovation_covariance),
            _normal_density(measurement, _H_CTRV @ ctrv[0], ctrv_innovation_covariance),
            _normal_density(measurement, _H_CA @ ca[0], ca_innovation_covariance),
        ]
    )
    mode_probabilities = _update_mode_probabilities(likelihoods, predicted_mode_probabilities)
    ctrv_updated = _normalize_ctrv(*ctrv_updated)
    detection = detections.iloc[association.detection_index]
    record = {
        "time": time_seconds,
        "cycle_counter": cycle_counter,
        "detection_id": int(detection["detection_id"]),
        "x": float(sensor_position[0] + measurement[0]),
        "y": float(sensor_position[1] + measurement[1]),
        "cov_xx": float(measurement_covariance[0, 0]),
        "cov_xy": float(measurement_covariance[0, 1]),
        "cov_yy": float(measurement_covariance[1, 1]),
    }
    return _IMMState(
        *cv_updated,
        *ctrv_updated,
        *ca_updated,
        mode_probabilities,
    ), record


def _cycle_standard_deviation(detections: pd.DataFrame, column: str) -> float:
    values = detections[column].to_numpy(dtype=float)
    if not np.allclose(values, values[0], rtol=0.0, atol=0.0):
        raise ValueError(
            f"Radar {column} must be constant within a cycle to match the MATLAB IMM."
        )
    return float(values[0])


def _mixing_probability(mode_probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transition_masses = _TRANSITION_PROBABILITIES * mode_probabilities[:, None]
    predicted_mode_probabilities = np.sum(transition_masses, axis=0)
    return transition_masses / predicted_mode_probabilities, predicted_mode_probabilities


def _mix_to_cv(state: _IMMState, weights: np.ndarray):
    return _mix_states(
        state.cv_state,
        state.cv_covariance,
        *_to_cv(state.ctrv_state, state.ctrv_covariance),
        *_to_cv(state.ca_state, state.ca_covariance),
        weights,
    )


def _mix_to_ctrv(state: _IMMState, weights: np.ndarray):
    mixed = _mix_states(
        *_to_ctrv(state.cv_state, state.cv_covariance),
        state.ctrv_state,
        state.ctrv_covariance,
        *_to_ctrv(state.ca_state, state.ca_covariance),
        weights,
    )
    return _normalize_ctrv(*mixed)


def _mix_to_ca(state: _IMMState, weights: np.ndarray):
    return _mix_states(
        *_to_ca(state.cv_state, state.cv_covariance),
        *_to_ca(state.ctrv_state, state.ctrv_covariance),
        state.ca_state,
        state.ca_covariance,
        weights,
    )


def _mix_states(xa, pa, xb, pb, xc, pc, weights):
    mixed_state = weights[0] * xa + weights[1] * xb + weights[2] * xc
    mixed_covariance = (
        weights[0] * (pa + np.outer(xa - mixed_state, xa - mixed_state))
        + weights[1] * (pb + np.outer(xb - mixed_state, xb - mixed_state))
        + weights[2] * (pc + np.outer(xc - mixed_state, xc - mixed_state))
    )
    return mixed_state, 0.5 * (mixed_covariance + mixed_covariance.T)


def _to_cv(state: np.ndarray, covariance: np.ndarray):
    if state.shape == (4,):
        return state, covariance
    if state.shape == (5,):
        cosine = np.cos(state[2])
        sine = np.sin(state[2])
        converted = np.array([state[0], cosine * state[4], state[1], sine * state[4]])
        jacobian = np.zeros((4, 5))
        jacobian[0, 0] = jacobian[2, 1] = 1.0
        jacobian[1, 4] = cosine
        jacobian[3, 4] = sine
        jacobian[1, 2] = -sine * state[4]
        jacobian[3, 2] = cosine * state[4]
        return converted, _symmetric(jacobian @ covariance @ jacobian.T)
    matrix = np.zeros((4, 6))
    matrix[0, 0] = matrix[1, 1] = matrix[2, 3] = matrix[3, 4] = 1.0
    return matrix @ state, matrix @ covariance @ matrix.T


def _to_ctrv(state: np.ndarray, covariance: np.ndarray):
    if state.shape == (5,):
        return state, covariance
    if state.shape == (4,):
        vx, vy = state[1], state[3]
        velocity_squared = vx**2 + vy**2 + 1e-12
        velocity = np.sqrt(velocity_squared)
        converted = np.array([state[0], state[2], np.arctan2(vy, vx), 0.0, velocity])
        jacobian = np.zeros((5, 4))
        jacobian[0, 0] = jacobian[1, 2] = 1.0
        jacobian[2, 1] = -vy / velocity_squared
        jacobian[2, 3] = vx / velocity_squared
        jacobian[4, 1] = vx / velocity
        jacobian[4, 3] = vy / velocity
        converted_covariance = jacobian @ covariance @ jacobian.T
        return converted, _symmetric(converted_covariance)

    vx, ax, vy, ay = state[1], state[2], state[4], state[5]
    velocity = np.sqrt(vx**2 + vy**2) + 1e-12
    converted = np.array(
        [
            state[0],
            state[3],
            np.arctan2(vy, vx),
            (vx * ay - vy * ax) / velocity**2,
            velocity,
        ]
    )
    jacobian = np.zeros((5, 6))
    jacobian[0, 0] = jacobian[1, 3] = 1.0
    jacobian[2, 1] = -vy / velocity**2
    jacobian[2, 4] = vx / velocity**2
    jacobian[3, 1] = (ay * (vy**2 - vx**2) + 2.0 * ax * vx * vy) / velocity**4
    jacobian[3, 2] = -vy / velocity**2
    jacobian[3, 4] = (ax * (vy**2 - vx**2) - 2.0 * ay * vx * vy) / velocity**4
    jacobian[3, 5] = vx / velocity**2
    jacobian[4, 1] = vx / velocity
    jacobian[4, 4] = vy / velocity
    return converted, _symmetric(jacobian @ covariance @ jacobian.T)


def _to_ca(state: np.ndarray, covariance: np.ndarray):
    if state.shape == (6,):
        return state, covariance
    if state.shape == (4,):
        matrix = np.zeros((6, 4))
        matrix[0, 0] = matrix[1, 1] = matrix[3, 2] = matrix[4, 3] = 1.0
        return matrix @ state, matrix @ covariance @ matrix.T

    cosine = np.cos(state[2])
    sine = np.sin(state[2])
    converted = np.array(
        [
            state[0],
            cosine * state[4],
            -sine * state[3] * state[4],
            state[1],
            sine * state[4],
            cosine * state[3] * state[4],
        ]
    )
    jacobian = np.zeros((6, 5))
    jacobian[0, 0] = jacobian[3, 1] = 1.0
    jacobian[1, 2] = -sine * state[4]
    jacobian[1, 4] = cosine
    jacobian[2, 2] = -cosine * state[3] * state[4]
    jacobian[2, 3] = -sine * state[4]
    jacobian[2, 4] = -sine * state[3]
    jacobian[4, 2] = cosine * state[4]
    jacobian[4, 4] = sine
    jacobian[5, 2] = -sine * state[3] * state[4]
    jacobian[5, 3] = cosine * state[4]
    jacobian[5, 4] = cosine * state[3]
    return converted, _symmetric(jacobian @ covariance @ jacobian.T)


def _predict_cv(state, covariance, interval_seconds, variance_acceleration_x, variance_acceleration_y):
    transition = np.array(
        [[1.0, interval_seconds], [0.0, 1.0]], dtype=float
    )
    transition = np.block(
        [[transition, np.zeros((2, 2))], [np.zeros((2, 2)), transition]]
    )
    gain = np.array([interval_seconds**2 / 2.0, interval_seconds])
    noise = np.outer(gain, gain)
    process_noise = np.block(
        [
            [variance_acceleration_x * noise, np.zeros((2, 2))],
            [np.zeros((2, 2)), variance_acceleration_y * noise],
        ]
    )
    return transition @ state, transition @ covariance @ transition.T + process_noise


def _predict_ctrv(state, covariance, interval_seconds, variance_velocity, variance_yaw):
    velocity = state[4]
    cosine = np.cos(state[2])
    sine = np.sin(state[2])
    predicted_state = np.array(
        [
            state[0] + cosine * velocity * interval_seconds,
            state[1] + sine * velocity * interval_seconds,
            state[2] + state[3] * interval_seconds,
            state[3],
            velocity,
        ]
    )
    transition = np.eye(5)
    transition[0, 2] = -sine * velocity * interval_seconds
    transition[0, 4] = cosine * interval_seconds
    transition[1, 2] = cosine * velocity * interval_seconds
    transition[1, 4] = sine * interval_seconds
    transition[2, 3] = interval_seconds
    process_noise = np.zeros((5, 5))
    process_noise[2, 2] = 0.33 * interval_seconds**3 * variance_yaw
    process_noise[2, 3] = process_noise[3, 2] = 0.5 * interval_seconds**2 * variance_yaw
    process_noise[3, 3] = interval_seconds * variance_yaw
    process_noise[4, 4] = interval_seconds * variance_velocity
    return _normalize_ctrv(
        predicted_state,
        _symmetric(transition @ covariance @ transition.T + process_noise),
    )


def _predict_ca(state, covariance, interval_seconds, tau_acceleration, variance_acceleration_x, variance_acceleration_y):
    transition_axis = np.array(
        [
            [1.0, interval_seconds, 0.5 * interval_seconds**2],
            [0.0, 1.0, interval_seconds],
            [0.0, 0.0, np.exp(-interval_seconds / tau_acceleration)],
        ]
    )
    transition = np.block(
        [[transition_axis, np.zeros((3, 3))], [np.zeros((3, 3)), transition_axis]]
    )
    gain = np.array([0.5 * interval_seconds**2, interval_seconds, 1.0])
    noise = np.outer(gain, gain)
    process_noise = np.block(
        [
            [variance_acceleration_x * noise, np.zeros((3, 3))],
            [np.zeros((3, 3)), variance_acceleration_y * noise],
        ]
    )
    return transition @ state, transition @ covariance @ transition.T + process_noise


def _ego_compensate(prediction, observation_matrix, ego_delta):
    state, covariance = prediction
    matrix = -observation_matrix.T
    return (
        state + matrix @ ego_delta,
        covariance + matrix @ _EGO_POSITION_COVARIANCE @ matrix.T,
    )


def _update(state, covariance, measurement, observation_matrix, measurement_covariance):
    innovation = measurement - observation_matrix @ state
    innovation_covariance = (
        observation_matrix @ covariance @ observation_matrix.T + measurement_covariance
    )
    gain = np.linalg.solve(innovation_covariance, observation_matrix @ covariance).T
    updated_state = state + gain @ innovation
    updated_covariance = covariance - gain @ observation_matrix @ covariance
    return (updated_state, _symmetric(updated_covariance)), innovation_covariance


def _normal_density(value, mean, covariance) -> float:
    difference = value - mean
    determinant = float(np.linalg.det(covariance))
    if determinant <= 0.0:
        raise ValueError("Innovation covariance must be positive definite.")
    quadratic = float(difference @ np.linalg.solve(covariance, difference))
    return float(np.exp(-0.5 * quadratic) / (2.0 * np.pi * np.sqrt(determinant)))


def _update_mode_probabilities(likelihoods, predicted_mode_probabilities):
    weighted_likelihoods = likelihoods * predicted_mode_probabilities
    normalization = float(np.sum(weighted_likelihoods))
    if normalization <= 0.0 or not np.isfinite(normalization):
        raise ValueError("IMM mode likelihoods could not be normalized.")
    return weighted_likelihoods / normalization


def _normalize_ctrv(state, covariance):
    state = np.asarray(state, dtype=float).copy()
    covariance = np.asarray(covariance, dtype=float).copy()
    if state[4] < 0.0:
        matrix = np.eye(5)
        matrix[4, 4] = -1.0
        state = matrix @ state
        state[2] += np.pi
        covariance = matrix @ covariance @ matrix.T
    if state[2] > np.pi:
        state[2] -= 2.0 * np.pi
    elif state[2] < -np.pi:
        state[2] += 2.0 * np.pi
    return state, covariance


def _symmetric(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _non_negative_finite(value, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return value
