"""Diagnose provisional priors for the latent Bayesian position model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import bayestraj.observations.coordinates as coordinates
import bayestraj.observations.io as observations_io
import bayestraj.observations.paths as paths

DATA_FILE = paths.data_path(
    "raw/processed_ship_data_2026-01-10T00-00-00+01-00_2026-02-02T00-00-00+01-00_10.csv"
)
RUN_ID_RANGE = range(0, 100)
MAX_TIME_GAP_SECONDS = 15.0
MIN_DISPLACEMENT_METERS = 1.0
ROBUST_MAD_SCALE_FACTOR = 1.4826
HALF_NORMAL_MEDIAN_FACTOR = 0.6744897501960817


@dataclass(frozen=True, slots=True)
class PositionModelPriorCalibration:
    """Robust prior candidates from measurement-contaminated displacements."""

    sample_count: int
    log_displacement_scale_prior_scale: float
    rotation_angle_prior_scale: float
    observed_displacement_residual_empirical_scale_m: float
    provisional_sigma_motion_residual_prior_scale_m: float


def calibrate_position_model_priors(
    data: pd.DataFrame,
) -> PositionModelPriorCalibration:
    """Return provisional scales from consecutive observed position changes."""
    log_scale_changes = []
    rotation_changes = []
    displacement_residual_components = []

    for _, run_data in data.groupby("run_id", sort=True):
        run_data = run_data.sort_values("time").reset_index(drop=True)
        timestamps = pd.to_datetime(run_data["time"], utc=True)
        time_seconds = (timestamps - timestamps.iloc[0]).dt.total_seconds().to_numpy()
        x_meters, y_meters = coordinates.gps_to_local_coordinates(
            run_data["gps_longitude"].to_numpy(dtype=float),
            run_data["gps_latitude"].to_numpy(dtype=float),
            unit="m",
        )
        displacement = np.column_stack((np.diff(x_meters), np.diff(y_meters)))
        time_steps = np.diff(time_seconds)
        norms = np.linalg.norm(displacement, axis=1)
        valid_displacement = (
            np.all(np.isfinite(displacement), axis=1)
            & np.isfinite(time_steps)
            & (time_steps > 0)
            & (time_steps <= MAX_TIME_GAP_SECONDS)
            & (norms >= MIN_DISPLACEMENT_METERS)
        )

        valid_transition = valid_displacement[:-1] & valid_displacement[1:]
        if not np.any(valid_transition):
            continue
        previous = displacement[:-1][valid_transition]
        current = displacement[1:][valid_transition]
        previous_norm = norms[:-1][valid_transition]
        current_norm = norms[1:][valid_transition]
        log_scale_changes.extend(np.log(current_norm / previous_norm))

        previous_angle = np.arctan2(previous[:, 1], previous[:, 0])
        current_angle = np.arctan2(current[:, 1], current[:, 0])
        angle_change = current_angle - previous_angle
        rotation_changes.extend(np.arctan2(np.sin(angle_change), np.cos(angle_change)))
        displacement_residual_components.extend((current - previous).ravel())

    log_scale = _robust_scale(log_scale_changes)
    rotation_scale = _robust_scale(rotation_changes)
    residual_empirical_scale = _robust_scale(displacement_residual_components)
    return PositionModelPriorCalibration(
        sample_count=len(log_scale_changes),
        log_displacement_scale_prior_scale=log_scale,
        rotation_angle_prior_scale=rotation_scale,
        observed_displacement_residual_empirical_scale_m=residual_empirical_scale,
        provisional_sigma_motion_residual_prior_scale_m=(
            residual_empirical_scale / HALF_NORMAL_MEDIAN_FACTOR
        ),
    )


def main() -> PositionModelPriorCalibration:
    """Print reproducible candidates without claiming motion-noise calibration."""
    data = observations_io.read_ship_data(DATA_FILE, run_id=RUN_ID_RANGE)
    data = data.loc[:, ["time", "run_id", "gps_latitude", "gps_longitude"]].copy()
    calibration = calibrate_position_model_priors(data)
    print("Bayesian latent-position model prior diagnostics")
    print("=" * 48)
    print(f"Transition samples              : {calibration.sample_count}")
    print(
        "Log displacement scale prior  : "
        f"{calibration.log_displacement_scale_prior_scale:.6f}"
    )
    print(
        f"Rotation-angle prior [rad]    : {calibration.rotation_angle_prior_scale:.6f}"
    )
    print(
        "Residual empirical scale [m] : "
        f"{calibration.observed_displacement_residual_empirical_scale_m:.6f}"
    )
    print(
        "Provisional motion scale [m] : "
        f"{calibration.provisional_sigma_motion_residual_prior_scale_m:.6f}"
    )
    print("The observed residual combines measurement noise and motion mismatch.")
    print("It is not a calibrated motion-only residual for the latent-position model.")
    return calibration


def _robust_scale(values) -> float:
    """Return the Gaussian-consistent MAD scale of finite scalar values."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Calibration values must contain finite samples.")
    median = float(np.median(values))
    median_absolute_deviation = float(np.median(np.abs(values - median)))
    scale = ROBUST_MAD_SCALE_FACTOR * median_absolute_deviation
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Calibration scale must be finite and positive.")
    return scale


if __name__ == "__main__":
    main()
