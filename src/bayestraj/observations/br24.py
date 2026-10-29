"""BR24 radar association matching the local CV/CTRV/CA IMM reference."""

from __future__ import annotations

import runpy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

from bayestraj.paths import project_path

from .coordinates import (
    calculate_motion_from_positions,
    calculate_signed_turn_rate_from_yaw,
)

ASSOCIATION_GATE_NIS = float(chi2.ppf(0.99, df=2))
_AZIMUTH_STANDARD_DEVIATION_INFLATION_RAD = np.deg2rad(0.5)
_IMM_IMPLEMENTATION_PATH = project_path("external/IMM/br24_imm.py")
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
_TRACKING_INPUT_COLUMNS = (
    "time",
    "cycle_counter",
    "sensor_x",
    "sensor_y",
    "sensor_yaw",
    "detection_id",
    "range",
    "azimuth",
    "std_range",
    "std_azimuth",
    "x_relative",
    "y_relative",
)
_TRACKING_REFERENCE_COLUMNS = (
    "ground_truth_x",
    "ground_truth_y",
    "ground_truth_yaw",
    "ground_truth_velocity",
)


def is_br24_tracking_input(csv_path: str | Path) -> bool:
    """Return whether a CSV has the complete flattened BR24 tracking schema."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    columns = pd.read_csv(csv_path, nrows=0).columns
    return set(_TRACKING_INPUT_COLUMNS).issubset(columns)


@dataclass(frozen=True, slots=True)
class BR24Association:
    """Nearest radar detection accepted by the Reuter 99% NIS gate."""

    detection_index: int
    nis: float


@dataclass(frozen=True, slots=True)
class BR24Detection:
    """One raw BR24 detection belonging to a sensor cycle."""

    detection_id: int
    range_m: float
    azimuth_rad: float
    std_range_m: float
    std_azimuth_rad: float


@dataclass(frozen=True, slots=True)
class BR24RadarCycle:
    """One timestamped sensor state and every radar detection at that time."""

    time_seconds: float
    cycle_counter: int
    sensor_position: np.ndarray
    sensor_yaw_rad: float
    detections: tuple[BR24Detection, ...]


@dataclass(frozen=True, slots=True)
class BR24TrackingReference:
    """Evaluation-only target labels synchronized to BR24 radar cycles."""

    cycle_counters: np.ndarray
    time_seconds: np.ndarray
    x_meters: np.ndarray
    y_meters: np.ndarray
    yaw_rad: np.ndarray | None = None
    velocity_mps: np.ndarray | None = None
    turn_rate_rad_s: np.ndarray | None = None

    def __post_init__(self) -> None:
        arrays = {
            name: np.asarray(getattr(self, name), dtype=float).copy()
            for name in (
                "cycle_counters",
                "time_seconds",
                "x_meters",
                "y_meters",
            )
        }
        for name in ("yaw_rad", "velocity_mps", "turn_rate_rad_s"):
            if getattr(self, name) is not None:
                arrays[name] = np.asarray(getattr(self, name), dtype=float).copy()
        if (
            any(values.ndim != 1 for values in arrays.values())
            or len({values.shape for values in arrays.values()}) != 1
        ):
            raise ValueError("BR24 reference values must be matching vectors.")
        if not arrays["time_seconds"].size or any(
            not np.all(np.isfinite(values)) for values in arrays.values()
        ):
            raise ValueError("BR24 reference values must be finite and non-empty.")
        if not np.array_equal(
            arrays["cycle_counters"], np.floor(arrays["cycle_counters"])
        ):
            raise ValueError("BR24 reference cycle counters must be integers.")
        arrays["cycle_counters"] = arrays["cycle_counters"].astype(int)
        if np.any(np.diff(arrays["cycle_counters"]) <= 0) or np.any(
            np.diff(arrays["time_seconds"]) <= 0.0
        ):
            raise ValueError("BR24 reference cycles must be strictly time-ordered.")
        for name, values in arrays.items():
            values.setflags(write=False)
            object.__setattr__(self, name, values)

    @property
    def speed_is_derived(self) -> bool:
        return self.velocity_mps is None

    @property
    def heading_is_derived(self) -> bool:
        return self.yaw_rad is None

    @property
    def resolved_velocity_mps(self) -> np.ndarray:
        if self.velocity_mps is not None:
            return self.velocity_mps
        speed, _ = calculate_motion_from_positions(
            self.x_meters, self.y_meters, self.time_seconds
        )
        speed.setflags(write=False)
        return speed

    @property
    def resolved_yaw_rad(self) -> np.ndarray:
        """Use native yaw when supplied; otherwise return the movement course."""
        if self.yaw_rad is not None:
            return self.yaw_rad
        _, course = calculate_motion_from_positions(
            self.x_meters, self.y_meters, self.time_seconds
        )
        course.setflags(write=False)
        return course

    @property
    def turn_rate_is_derived(self) -> bool:
        return self.turn_rate_rad_s is None

    @property
    def resolved_turn_rate_rad_s(self) -> np.ndarray:
        """Use supplied evaluation labels or derive causal rates from reference yaw."""
        if self.turn_rate_rad_s is not None:
            return self.turn_rate_rad_s
        rates = calculate_signed_turn_rate_from_yaw(
            self.resolved_yaw_rad, self.time_seconds
        )
        rates.setflags(write=False)
        return rates


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


def br24_measurement_position(
    *,
    sensor_position: np.ndarray,
    range_m: float,
    azimuth_rad: float,
) -> np.ndarray:
    """Return a BR24 detection in the world frame used by the reference export."""
    sensor_position = np.asarray(sensor_position, dtype=float)
    if sensor_position.shape != (2,) or not np.all(np.isfinite(sensor_position)):
        raise ValueError("sensor_position must be a finite two-dimensional vector.")
    return sensor_position + polar_to_cartesian(range_m, azimuth_rad)[0]


def br24_measurement_covariance(
    *,
    range_m: float,
    azimuth_rad: float,
    std_range_m: float,
    std_azimuth_rad: float,
) -> np.ndarray:
    """Return the Cartesian covariance of one raw BR24 radar detection."""
    relative_position = polar_to_cartesian(range_m, azimuth_rad)[0]
    return position_measurement_noise(
        predicted_position=relative_position,
        range_variance=_non_negative_finite(std_range_m, "std_range_m") ** 2,
        azimuth_variance=(
            _non_negative_finite(std_azimuth_rad, "std_azimuth_rad")
            + _AZIMUTH_STANDARD_DEVIATION_INFLATION_RAD
        )
        ** 2,
    )


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
    """Export IMM-associated detections using the optional local implementation."""
    if not _IMM_IMPLEMENTATION_PATH.is_file():
        raise FileNotFoundError(
            "Optional IMM implementation not found: external/IMM/br24_imm.py. "
            "Restore the local external IMM files to use this export."
        )
    imm = runpy.run_path(str(_IMM_IMPLEMENTATION_PATH))
    return imm["export_br24_target_radar_trajectory"](
        radar_csv=radar_csv,
        sensor_csv=sensor_csv,
        output_csv=output_csv,
    )


def export_br24_tracking_input(
    *,
    radar_csv: str | Path,
    sensor_csv: str | Path,
    output_csv: str | Path,
) -> Path:
    """Flatten BR24 sensor cycles and their raw radar detections for tracking."""
    radar = _read_radar_csv(radar_csv)
    sensor = _read_sensor_csv(sensor_csv)
    combined = sensor.merge(
        radar,
        how="left",
        on="cycle_counter",
        validate="one_to_many",
        suffixes=("_sensor", ""),
    )
    has_detection = combined["detection_id"].notna()
    if not np.array_equal(
        combined.loc[has_detection, "time_sensor"].to_numpy(dtype=float),
        combined.loc[has_detection, "time"].to_numpy(dtype=float),
    ):
        raise ValueError("Radar detection times must match their sensor cycle times.")

    output = combined.loc[
        :,
        [
            "time_sensor",
            "cycle_counter",
            "x",
            "y",
            "yaw",
            "detection_id",
            "range",
            "azimuth",
            "std_range",
            "std_azimuth",
            "x_relative",
            "y_relative",
        ],
    ].rename(
        columns={
            "time_sensor": "time",
            "x": "sensor_x",
            "y": "sensor_y",
            "yaw": "sensor_yaw",
        }
    )
    output = output.sort_values(
        ["cycle_counter", "detection_id"],
        na_position="last",
    ).reset_index(drop=True)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output_path


def read_br24_tracking_input(csv_path: str | Path) -> tuple[BR24RadarCycle, ...]:
    """Read flattened BR24 input as complete time-ordered radar cycles."""
    data = _read_csv_with_columns(csv_path, required=_TRACKING_INPUT_COLUMNS)
    _validate_finite_columns(
        data,
        ("time", "cycle_counter", "sensor_x", "sensor_y", "sensor_yaw"),
        source="BR24 tracking input",
    )
    _validate_integral_column(
        data,
        "cycle_counter",
        source="BR24 tracking input",
    )
    cycles: list[BR24RadarCycle] = []
    for cycle_counter, group in data.groupby("cycle_counter", sort=True):
        sensor_values = group.loc[
            :, ("time", "sensor_x", "sensor_y", "sensor_yaw")
        ].to_numpy(dtype=float)
        if not np.allclose(sensor_values, sensor_values[0], rtol=0.0, atol=0.0):
            raise ValueError("Sensor values must be identical within each BR24 cycle.")
        detection_columns = (
            "detection_id",
            "range",
            "azimuth",
            "std_range",
            "std_azimuth",
            "x_relative",
            "y_relative",
        )
        detection_rows = group.loc[:, detection_columns]
        has_detection = detection_rows["detection_id"].notna()
        if detection_rows.loc[~has_detection].notna().any(axis=None):
            raise ValueError(
                "Empty BR24 detection rows must contain only missing values."
            )
        detections = detection_rows.loc[has_detection].copy()
        if not detections.empty:
            _validate_finite_columns(
                detections,
                detection_columns,
                source="BR24 tracking input detection",
            )
            _validate_integral_column(
                detections,
                "detection_id",
                source="BR24 tracking input detection",
            )
            _validate_non_negative_columns(
                detections,
                ("range", "std_range", "std_azimuth"),
                source="BR24 tracking input detection",
            )
            if detections["detection_id"].duplicated().any():
                raise ValueError(
                    "DetectionID values must be unique within each BR24 cycle."
                )
        cycles.append(
            BR24RadarCycle(
                time_seconds=float(sensor_values[0, 0]),
                cycle_counter=int(cycle_counter),
                sensor_position=sensor_values[0, 1:3].copy(),
                sensor_yaw_rad=float(sensor_values[0, 3]),
                detections=tuple(
                    BR24Detection(
                        detection_id=int(row.detection_id),
                        range_m=float(row.range),
                        azimuth_rad=float(row.azimuth),
                        std_range_m=float(row.std_range),
                        std_azimuth_rad=float(row.std_azimuth),
                    )
                    for row in detections.sort_values("detection_id").itertuples(
                        index=False
                    )
                ),
            )
        )
    if not cycles:
        raise ValueError("BR24 tracking input must contain at least one sensor cycle.")
    times = np.array([cycle.time_seconds for cycle in cycles])
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("BR24 tracking-input cycle times must be strictly increasing.")
    return tuple(cycles)


def read_br24_tracking_reference(
    csv_path: str | Path,
) -> BR24TrackingReference | None:
    """Read optional evaluation labels without exposing them to the tracker.

    ``None`` means that the flattened BR24 input has no reference labels.  When
    present, x/y are required. Yaw [rad], velocity [m/s], and turn rate [rad/s]
    are independent optional columns; absent motion values are derived only for
    display/evaluation. Supplied values must be finite and identical within a cycle.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    normalized_columns = {
        str(column).lower() for column in pd.read_csv(csv_path, nrows=0).columns
    }
    supported_columns = (*_TRACKING_REFERENCE_COLUMNS, "ground_truth_turn_rate")
    available_columns = set(supported_columns) & normalized_columns
    if not available_columns:
        return None
    required_columns = ("ground_truth_x", "ground_truth_y")
    if not set(required_columns).issubset(available_columns):
        missing = sorted(set(required_columns) - available_columns)
        raise ValueError(f"Missing BR24 reference columns: {missing}")
    selected_columns = tuple(
        name for name in supported_columns if name in available_columns
    )
    data = _read_csv_with_columns(
        csv_path,
        required=(
            "time",
            "cycle_counter",
            *selected_columns,
        ),
    )
    _validate_finite_columns(data, data.columns, source="BR24 tracking reference")
    _validate_integral_column(
        data,
        "cycle_counter",
        source="BR24 tracking reference",
    )
    cycle_references = []
    reference_columns = ("time", *selected_columns)
    for cycle_counter, group in data.groupby("cycle_counter", sort=True):
        values = group.loc[:, reference_columns].to_numpy(dtype=float)
        if not np.allclose(values, values[0], rtol=0.0, atol=0.0):
            raise ValueError(
                "Reference values must be identical within each BR24 cycle."
            )
        cycle_references.append((int(cycle_counter), *values[0]))
    reference = np.asarray(cycle_references, dtype=float)
    values_by_column = {
        name: reference[:, index + 1] for index, name in enumerate(reference_columns)
    }
    return BR24TrackingReference(
        cycle_counters=reference[:, 0],
        time_seconds=reference[:, 1],
        x_meters=reference[:, 2],
        y_meters=reference[:, 3],
        yaw_rad=values_by_column.get("ground_truth_yaw"),
        velocity_mps=values_by_column.get("ground_truth_velocity"),
        turn_rate_rad_s=values_by_column.get("ground_truth_turn_rate"),
    )


def export_br24_target_radar_with_ground_truth(
    *,
    radar_csv: str | Path,
    reference_csv: str | Path,
    output_csv: str | Path,
) -> Path:
    """Add evaluation-only interpolated target reference values to radar positions."""
    radar = _read_csv_with_columns(radar_csv, required=_OUTPUT_COLUMNS)
    _validate_finite_columns(radar, _OUTPUT_COLUMNS, source="Radar trajectory")
    _validate_strictly_increasing_time(radar, source="Radar trajectory")
    ground_truth = _interpolate_ground_truth(
        reference_csv=reference_csv,
        target_times=radar["time"].to_numpy(dtype=float),
    )
    output = radar.copy()
    for name, values in ground_truth.items():
        output[f"ground_truth_{name}"] = values

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output_path


def export_br24_tracking_input_with_reference(
    *,
    tracking_csv: str | Path,
    reference_csv: str | Path,
    output_csv: str | Path,
) -> Path:
    """Attach evaluation-only reference labels to a flattened BR24 input CSV.

    The original tracking columns are preserved exactly.  Ground-truth columns are
    derived only after the raw radar cycles have been validated and are ignored by
    :func:`read_br24_tracking_input`.
    """
    cycles = read_br24_tracking_input(tracking_csv)
    tracking = pd.read_csv(tracking_csv, float_precision="round_trip")
    cycle_counters = np.asarray([cycle.cycle_counter for cycle in cycles], dtype=int)
    ground_truth = _interpolate_ground_truth(
        reference_csv=reference_csv,
        target_times=np.asarray([cycle.time_seconds for cycle in cycles], dtype=float),
    )
    output = tracking.copy()
    for name, values in ground_truth.items():
        values_by_cycle = dict(zip(cycle_counters, values, strict=True))
        output[f"ground_truth_{name}"] = output["cycle_counter"].map(values_by_cycle)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output_path


def _interpolate_ground_truth(
    *,
    reference_csv: str | Path,
    target_times: np.ndarray,
) -> dict[str, np.ndarray]:
    """Synchronize complete ground-truth states to validated radar-cycle times."""
    reference = _read_csv_with_columns(
        reference_csv,
        required=("time", "x", "y", "yaw", "velocity"),
    )
    _validate_finite_columns(reference, ("time",), source="Ground-truth trajectory")
    _validate_strictly_increasing_time(
        reference,
        source="Ground-truth trajectory",
        minimum_sample_count=2,
    )

    reference_value_columns = ("x", "y", "yaw", "velocity")
    complete_reference_rows = np.isfinite(
        reference.loc[:, reference_value_columns].to_numpy(dtype=float)
    ).all(axis=1)
    complete_reference_indices = np.flatnonzero(complete_reference_rows)
    reference = reference.loc[complete_reference_rows].copy()
    _validate_strictly_increasing_time(
        reference,
        source="Complete ground-truth trajectory",
        minimum_sample_count=2,
    )

    target_times = np.asarray(target_times, dtype=float)
    reference_times = reference["time"].to_numpy(dtype=float)
    if target_times.size and (
        target_times[0] < reference_times[0] or target_times[-1] > reference_times[-1]
    ):
        raise ValueError("Radar times fall outside the ground-truth time range.")
    if _radar_times_cross_incomplete_ground_truth(
        target_times,
        reference_times,
        complete_reference_indices,
    ):
        raise ValueError("Radar times fall within an incomplete ground-truth interval.")

    unwrapped_yaw = np.unwrap(reference["yaw"].to_numpy(dtype=float))
    return {
        "x": np.interp(
            target_times, reference_times, reference["x"].to_numpy(dtype=float)
        ),
        "y": np.interp(
            target_times, reference_times, reference["y"].to_numpy(dtype=float)
        ),
        "yaw": _wrap_angle(np.interp(target_times, reference_times, unwrapped_yaw)),
        "velocity": np.interp(
            target_times,
            reference_times,
            reference["velocity"].to_numpy(dtype=float),
        ),
    }


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
            "xrelative",
            "yrelative",
        ),
    )
    radar = radar.rename(
        columns={
            "cyclecounter": "cycle_counter",
            "detectionid": "detection_id",
            "stdrange": "std_range",
            "stdazimuth": "std_azimuth",
            "xrelative": "x_relative",
            "yrelative": "y_relative",
        }
    )
    _validate_finite_columns(
        radar,
        (
            "time",
            "cycle_counter",
            "detection_id",
            "range",
            "azimuth",
            "std_range",
            "std_azimuth",
            "x_relative",
            "y_relative",
        ),
        source="Radar",
    )
    _validate_non_negative_columns(
        radar, ("range", "std_range", "std_azimuth"), source="Radar"
    )
    _validate_integral_column(radar, "cycle_counter", source="Radar")
    _validate_integral_column(radar, "detection_id", source="Radar")
    if radar.duplicated(("cycle_counter", "detection_id")).any():
        raise ValueError("Radar DetectionID values must be unique within each cycle.")
    return radar.sort_values(["cycle_counter", "detection_id"]).reset_index(drop=True)


def _read_sensor_csv(csv_path: str | Path) -> pd.DataFrame:
    sensor = _read_csv_with_columns(
        csv_path,
        required=("time", "cyclecounter", "x", "y", "yaw"),
    )
    sensor = sensor.rename(columns={"cyclecounter": "cycle_counter"})
    _validate_finite_columns(
        sensor,
        ("time", "cycle_counter", "x", "y", "yaw"),
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


def _read_csv_with_columns(
    csv_path: str | Path, *, required: tuple[str, ...]
) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    source = pd.read_csv(csv_path, float_precision="round_trip")
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
        raise ValueError(
            f"{source} range and standard deviations must be non-negative."
        )


def _validate_integral_column(data, column, *, source: str) -> None:
    values = data[column].to_numpy(dtype=float)
    if not np.array_equal(values, np.floor(values)):
        raise ValueError(f"{source} {column} values must be integers.")
    data[column] = values.astype(int)


def _validate_strictly_increasing_time(
    data,
    *,
    source: str,
    minimum_sample_count: int = 0,
) -> None:
    if len(data) < minimum_sample_count:
        raise ValueError(
            f"{source} must contain at least {minimum_sample_count} samples."
        )
    if np.any(np.diff(data["time"].to_numpy(dtype=float)) <= 0.0):
        raise ValueError(f"{source} times must be strictly increasing.")


def _wrap_angle(angle_rad: np.ndarray) -> np.ndarray:
    """Wrap angles to the half-open interval [-pi, pi)."""
    return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi


def _radar_times_cross_incomplete_ground_truth(
    radar_times: np.ndarray,
    complete_reference_times: np.ndarray,
    complete_reference_indices: np.ndarray,
) -> bool:
    """Return whether an interpolation interval contains incomplete reference rows."""
    missing_between_complete_rows = np.diff(complete_reference_indices) > 1
    if not missing_between_complete_rows.any():
        return False
    interval_starts = complete_reference_times[:-1][missing_between_complete_rows]
    interval_ends = complete_reference_times[1:][missing_between_complete_rows]
    return bool(
        np.any(
            (radar_times[:, np.newaxis] > interval_starts)
            & (radar_times[:, np.newaxis] < interval_ends)
        )
    )


def _non_negative_finite(value, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return value
