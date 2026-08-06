"""Evaluation and visualization of probabilistic trajectory predictions."""

from ship_trajectory_prediction.evaluation.metrics import (
    PositionEvaluation,
    evaluate_position_predictions,
    format_position_evaluation,
    print_position_evaluation,
)
from ship_trajectory_prediction.evaluation.motion_priors import (
    MotionPriorSamples,
    PriorSuggestions,
    collect_motion_prior_samples,
    plot_motion_prior_distributions,
    print_motion_prior_report,
    suggest_prior_scales,
)
from ship_trajectory_prediction.evaluation.posterior_plotting import (
    plot_scalar_posterior,
    plot_scalar_posterior_comparison,
    plot_scalar_prior_to_posterior_update,
    plot_state_credible_band,
    plot_state_posterior_at_time,
    save_bayesian_ctrv_posterior_plots,
    show_bayesian_ctrv_posterior_plots,
    show_bayesian_ctrv_prior_update_plots,
)
from ship_trajectory_prediction.evaluation.prediction_plotting import (
    plot_operational_prediction,
    plot_prediction,
    plot_trajectory_paths,
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
    "MotionPriorSamples",
    "PriorSuggestions",
    "RollingPositionSummary",
    "RollingWindowSpec",
    "build_rolling_window_specs",
    "collect_motion_prior_samples",
    "evaluate_position_predictions",
    "format_position_evaluation",
    "plot_operational_prediction",
    "plot_prediction",
    "plot_trajectory_paths",
    "plot_motion_prior_distributions",
    "plot_scalar_posterior",
    "plot_scalar_posterior_comparison",
    "plot_scalar_prior_to_posterior_update",
    "plot_state_credible_band",
    "plot_state_posterior_at_time",
    "print_prediction_setup",
    "print_motion_prior_report",
    "print_position_evaluation",
    "save_bayesian_ctrv_posterior_plots",
    "show_bayesian_ctrv_posterior_plots",
    "show_bayesian_ctrv_prior_update_plots",
    "suggest_prior_scales",
    "summarize_rolling_predictions",
]
