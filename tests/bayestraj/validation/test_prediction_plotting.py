"""Tests for posterior-predictive trajectory plot sampling."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

import bayestraj.validation.prediction_plotting as prediction_plotting


@pytest.mark.parametrize(
    ("renderer_name", "per_step", "label"),
    [
        ("_draw_prediction_regions", True, "Covariance ellipse"),
        ("_draw_prediction_density_contours", True, "Per-step density contour"),
        ("_draw_prediction_density_regions", True, "Per-step density region"),
        ("_draw_horizon_prediction_density_contours", False, "Horizon density contour"),
        ("_draw_horizon_prediction_density_regions", False, "Horizon density region"),
    ],
)
@pytest.mark.parametrize("levels", [(0.99,), (0.5, 0.9, 0.99)])
def test_prediction_renderers_support_an_optional_outer_99_percent_region(
    renderer_name, per_step, label, levels
):
    positions = np.random.default_rng(42).normal(size=(400, 2, 2))
    positions[:, 1] += [3, 1]
    draws = (positions[:, :, 0], positions[:, :, 1])
    figure, axis = plt.subplots()
    try:
        renderer = getattr(prediction_plotting, renderer_name)
        args = (draws, [0, 1, 2]) if per_step else (draws,)
        options = {"annotate_time": False} if per_step else {}
        handles = renderer(axis, *args, region_probabilities=levels, **options)
        assert [handle.get_label() for handle in handles] == [
            f"{label} ({100 * level:g} %)" for level in levels
        ]
        if renderer_name == "_draw_prediction_regions":
            regions = {
                level: next(
                    patch
                    for patch in axis.patches
                    if patch.get_gid() == f"posterior-predictive-region-{level:g}-t0"
                )
                for level in levels
            }
            assert regions[0.99].width > 0
            if 0.9 in regions:
                assert regions[0.99].width * regions[0.99].height == pytest.approx(
                    2 * regions[0.9].width * regions[0.9].height
                )
        elif "contours" in renderer_name:
            assert axis.collections
            assert all(
                len(collection.levels) == len(levels) for collection in axis.collections
            )
        else:
            bounds = {}
            for level in levels:
                token = f"-{level:g}"
                region = next(
                    collection
                    for collection in axis.collections
                    if token + "-" in (collection.get_gid() or "")
                    or (collection.get_gid() or "").endswith(token)
                )
                bounds[level] = region.get_paths()[0].get_extents()
            assert bounds[0.99].width > 0
            if 0.9 in bounds:
                assert bounds[0.99].width > bounds[0.9].width
                assert bounds[0.99].height > bounds[0.9].height
    finally:
        plt.close(figure)


def test_sample_trajectory_indices_honors_requests_above_default_count():
    """Allow an experiment to show more than the default number of paths."""
    indices = prediction_plotting._sample_trajectory_indices(120, 100, seed=42)

    assert len(indices) == 100
