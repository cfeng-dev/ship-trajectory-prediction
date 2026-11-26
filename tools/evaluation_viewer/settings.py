"""Validated display options, independent of Tk and model fitting."""

from dataclasses import dataclass
from math import isfinite

from bayestraj.validation.metrics import STUDENT_T_DEGREES_OF_FREEDOM

METRICS = {
    "ade_m": "ADE",
    "fde_m": "FDE",
    "rmse_m": "RMSE",
    "energy_score_m": "Energy Score",
    "mean_negative_log_predictive_density": "NLPD (Gaussian)",
    "mean_student_t_negative_log_predictive_density": f"NLPD (Student-t, ν={STUDENT_T_DEGREES_OF_FREEDOM})",
    "fit_runtime_seconds": "Runtime",
}


@dataclass(frozen=True)
class ViewerOptions:
    top_metric: str = "ade_m"
    bottom_metric: str = "mean_negative_log_predictive_density"
    follow_metric_time: bool = False
    metric_window_width: float = 60.0
    follow_ship: bool = True
    ship_view_span: float = 100.0
    show_legend: bool = True
    language: str = "en"

    def __post_init__(self):
        if self.top_metric not in METRICS or self.bottom_metric not in METRICS:
            raise ValueError("Select a supported metric for each diagram.")
        if self.language not in {"en", "de"}:
            raise ValueError("Select English or German.")
        for name in ("metric_window_width", "ship_view_span"):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and greater than zero.")
            object.__setattr__(self, name, value)
