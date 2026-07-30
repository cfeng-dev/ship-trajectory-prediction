"""Evaluation and visualization of probabilistic trajectory predictions."""

from ship_trajectory_prediction.evaluation.metrics import (
    PositionEvaluation,
    evaluate_position_predictions,
    format_position_evaluation,
    print_position_evaluation,
)
from ship_trajectory_prediction.evaluation.plotting import plot_prediction
from ship_trajectory_prediction.evaluation.posterior import (
    plot_scalar_posterior,
    plot_scalar_posterior_comparison,
    plot_state_credible_band,
    plot_state_posterior_at_time,
    save_bayesian_ctrv_posterior_plots,
)
from ship_trajectory_prediction.evaluation.reporting import print_prediction_setup
from ship_trajectory_prediction.evaluation.rolling import (
    RollingPositionSummary,
    RollingWindowSpec,
    build_rolling_window_specs,
    summarize_rolling_predictions,
)

__all__ = [
    "PositionEvaluation",
    "RollingPositionSummary",
    "RollingWindowSpec",
    "build_rolling_window_specs",
    "evaluate_position_predictions",
    "format_position_evaluation",
    "plot_prediction",
    "plot_scalar_posterior",
    "plot_scalar_posterior_comparison",
    "plot_state_credible_band",
    "plot_state_posterior_at_time",
    "print_prediction_setup",
    "print_position_evaluation",
    "save_bayesian_ctrv_posterior_plots",
    "summarize_rolling_predictions",
]
