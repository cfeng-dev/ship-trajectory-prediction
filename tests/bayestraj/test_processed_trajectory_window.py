"""Local processed trajectories prepare inference windows without GPS metadata."""

import numpy as np
import pandas as pd

import bayestraj.observations.window as observation_window


def test_processed_window_uses_selected_elapsed_positions_without_gps_origin():
    """A time/x/y window preserves local positions and restarts time at zero."""
    data = pd.DataFrame(
        {
            "time": [0.0, 10.0, 20.0, 30.0, 40.0],
            "x": [0.0, 2.0, 5.0, 9.0, 14.0],
            "y": [0.0, -1.0, -1.0, 0.0, 2.0],
        }
    )

    prepare = getattr(observation_window, "prepare_processed_trajectory_window", None)

    assert callable(prepare)
    window = prepare(
        data,
        observation_count=3,
        prediction_count=1,
        start_index=1,
    )
    np.testing.assert_allclose(window.time_seconds, [0.0, 10.0, 20.0, 30.0])
    np.testing.assert_allclose(window.x_meters, [2.0, 5.0, 9.0, 14.0])
    np.testing.assert_allclose(window.y_meters, [-1.0, -1.0, 0.0, 2.0])
    assert window.reference_longitude is None
    assert window.reference_latitude is None
    assert np.isnan(window.gps_speed_mps).all()
