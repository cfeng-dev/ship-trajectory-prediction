"""Evaluate parametric Bayesian CTRV forecasts across rolling windows."""

import matplotlib.pyplot as plt

import bayestraj.forecasting.bayesian_ctrv as config
import bayestraj.inference.configuration as inference
import bayestraj.models.bayesian_ctrv as bayesian_model
import bayestraj.observations.paths as paths
import bayestraj.observations.plotting as observation_plotting
import bayestraj.validation.bayesian_ctrv_workflow as workflow
import bayestraj.validation.cli as cli

DATA_FILE = paths.data_path(
    "raw/processed_ship_data_2026-01-10T00-00-00+01-00_2026-02-02T00-00-00+01-00_10.csv"
)

# Rolling inference:
# - Batch: "vi_sliding", "vi_expanding", "mcmc_sliding", or "mcmc_expanding".
# - Online: "rbpf" or "smc".
EXPERIMENT = config.RollingExperimentConfig(
    run_id=102,
    observation_count=5,
    prediction_count=3,
    position_noise_std_m=5.0,
    position_noise_seed=2026,
    stride=1,  # Start a new three-step forecast after every observed position.
    inference_method="rbpf",
    inference_seed=42,
)
PRIORS = bayesian_model.BayesianCTRVPriors(
    speed_prior_upper_mps=20.0,
    speed_prior_tail_probability=0.05,
    turn_rate_prior_abs_rate_deg_s=4.5,
    turn_rate_prior_tail_probability=0.05,
    sigma_position_observation_prior_upper_m=20.0,
    sigma_position_observation_prior_tail_probability=0.05,
    sigma_speed_process_prior_upper_mps=5.0,
    sigma_speed_process_prior_tail_probability=0.05,
    sigma_turn_rate_process_prior_upper_deg_s=4.5,
    sigma_turn_rate_process_prior_tail_probability=0.05,
)
VI_CONFIG = inference.create_default_vi_config()
MCMC_CONFIG = inference.create_default_mcmc_config()
RBPF_CONFIG = inference.create_default_ctrv_rbpf_config()
SMC_CONFIG = inference.create_default_ctrv_smc_config()
CREDIBLE_INTERVAL = 0.9
MAX_WINDOWS = None
PLOT_EACH_WINDOW = False
SAMPLE_TRAJECTORIES_PER_FORECAST = 15
SHOW_TIME_LABELS = False  # Avoid repeated labels across all rolling windows.
SHOW_FIT_RUNTIME_PLOT = True
RUNTIME_PLOT_STYLE = observation_plotting.ShipDataPlotStyle()


def main(argv=None):
    """Run the configured parametric Bayesian CTRV rolling evaluation."""
    options = cli.parse_bayesian_ctrv_evaluation_arguments(
        description=__doc__,
        experiment=EXPERIMENT,
        priors=PRIORS,
        vi_config=VI_CONFIG,
        max_windows=MAX_WINDOWS,
        plot_each_window=PLOT_EACH_WINDOW,
        argv=argv,
    )
    predictions, summary = workflow.run_bayesian_ctrv_evaluation(
        data_file=DATA_FILE,
        experiment=EXPERIMENT,
        priors=PRIORS,
        vi_config=VI_CONFIG,
        mcmc_config=MCMC_CONFIG,
        rbpf_config=RBPF_CONFIG,
        smc_config=SMC_CONFIG,
        fullrank_grad_samples=inference.DEFAULT_FULLRANK_GRAD_SAMPLES,
        credible_interval=CREDIBLE_INTERVAL,
        sample_trajectories_per_forecast=SAMPLE_TRAJECTORIES_PER_FORECAST,
        options=options,
        show_time_labels=SHOW_TIME_LABELS,
    )
    if SHOW_FIT_RUNTIME_PLOT:
        plot_fit_runtime(predictions, inference_selection=options.inference_method)
    return predictions, summary


def plot_fit_runtime(predictions, *, inference_selection=None):
    """Plot inference time per rolling window against available history size."""
    required_columns = {
        "window_index",
        "observation_count",
        "fit_runtime_seconds",
        "inference_method",
    }
    missing_columns = sorted(required_columns.difference(predictions.columns))
    if missing_columns:
        raise ValueError(f"Missing fit-runtime columns: {missing_columns}")
    windows = predictions.groupby("window_index", sort=True).first()
    inference_methods = windows["inference_method"].dropna().unique()
    if len(inference_methods) != 1:
        raise ValueError(
            "Runtime plot requires exactly one inference_method per configuration."
        )
    inference_title = _format_inference_title(
        inference_selection or str(inference_methods[0])
    )
    figure, axis = plt.subplots(figsize=RUNTIME_PLOT_STYLE.speed_figure_size)
    axis.plot(
        windows["observation_count"].tolist(),
        windows["fit_runtime_seconds"].tolist(),
        marker="o",
        markersize=3,
        color=RUNTIME_PLOT_STYLE.derived_data_color,
        label="Inferenzzeit pro Vorhersagefenster",
    )
    axis.set_xlabel(
        "Anzahl bisher beobachteter Positionen",
        fontsize=RUNTIME_PLOT_STYLE.axis_label_font_size,
    )
    axis.set_ylabel(
        "Inferenzzeit pro Vorhersagefenster [s]",
        fontsize=RUNTIME_PLOT_STYLE.axis_label_font_size,
    )
    axis.set_title(
        f"Bayesian-CTRV ({inference_title})",
        pad=RUNTIME_PLOT_STYLE.title_pad,
        fontsize=RUNTIME_PLOT_STYLE.title_font_size,
        fontweight=RUNTIME_PLOT_STYLE.title_font_weight,
    )
    axis.tick_params(axis="both", labelsize=RUNTIME_PLOT_STYLE.axis_tick_font_size)
    axis.legend(loc=RUNTIME_PLOT_STYLE.legend_location)
    figure.tight_layout()
    plt.show()
    return figure, axis


def _format_inference_title(inference_selection):
    """Return a concise title label for the selected inference configuration."""
    inference_method, separator, window_mode = inference_selection.partition("_")
    title = inference_method.upper()
    if separator and window_mode in {"sliding", "expanding"}:
        return f"{title}, {window_mode.title()} Window"
    return title


if __name__ == "__main__":
    main()
