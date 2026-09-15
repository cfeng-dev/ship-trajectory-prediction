"""Run the reproducible final Master's-thesis CTRV benchmark."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import bayestraj.forecasting.bayesian_ctrv as bayesian_forecasting
import bayestraj.forecasting.deterministic_ctrv as deterministic_forecasting
import bayestraj.inference.configuration as inference
import bayestraj.models.bayesian_ctrv as bayesian_model
import bayestraj.observations.paths as paths
import bayestraj.validation.bayesian_ctrv_workflow as bayesian_workflow
import bayestraj.validation.benchmark as benchmark
import bayestraj.validation.cli as validation_cli
import bayestraj.validation.deterministic_ctrv_workflow as deterministic_workflow

DATA_FILE = paths.data_path(
    "raw/processed_ship_data_2026-01-10T00-00-00+01-00_2026-02-02T00-00-00+01-00_10.csv"
)

# Edit these explicit defaults to define the final thesis test set. The command
# line can override every expensive benchmark dimension for development runs.
RUN_IDS = (102,)
OBSERVATION_COUNT = 5
PREDICTION_COUNTS = (1, 3)
POSITION_NOISE_STD_M = (0.0, 5.0)
METHODS = ("deterministic", "rbpf", "smc")
INFERENCE_SEED = 42
POSITION_NOISE_SEED = 2026
MAX_WINDOWS = None
OUTPUT_DIRECTORY = Path("results/thesis_benchmark")

# MCMC is intentionally excluded from METHODS by default. Add
# "mcmc_expanding" there and use these optional restrictions for reference runs.
MCMC_RUN_IDS = ()
MCMC_PREDICTION_COUNTS = ()
MCMC_WINDOW_INDICES = ()

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


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse compact overrides for the main benchmark dimensions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-ids", default=_csv(RUN_IDS))
    parser.add_argument("--methods", default=_csv(METHODS))
    parser.add_argument("--prediction-counts", default=_csv(PREDICTION_COUNTS))
    parser.add_argument("--noise-levels", default=_csv(POSITION_NOISE_STD_M))
    parser.add_argument("--max-windows", type=int, default=MAX_WINDOWS)
    parser.add_argument("--output-directory", type=Path, default=OUTPUT_DIRECTORY)
    parser.add_argument("--mcmc-run-ids", default=_csv(MCMC_RUN_IDS))
    parser.add_argument(
        "--mcmc-prediction-counts",
        default=_csv(MCMC_PREDICTION_COUNTS),
    )
    parser.add_argument("--mcmc-window-indices", default=_csv(MCMC_WINDOW_INDICES))
    return parser.parse_args(argv)


def build_config(arguments: argparse.Namespace) -> benchmark.BenchmarkConfig:
    """Convert command-line values into one validated benchmark configuration."""
    return benchmark.BenchmarkConfig(
        run_ids=_parse_int_csv(arguments.run_ids, "run_ids"),
        observation_count=OBSERVATION_COUNT,
        prediction_counts=_parse_int_csv(
            arguments.prediction_counts,
            "prediction_counts",
        ),
        position_noise_std_m=_parse_float_csv(arguments.noise_levels, "noise_levels"),
        methods=_parse_string_csv(arguments.methods, "methods"),
        position_noise_seed=POSITION_NOISE_SEED,
        inference_seed=INFERENCE_SEED,
        max_windows=arguments.max_windows,
        mcmc_run_ids=_parse_optional_int_csv(arguments.mcmc_run_ids, "mcmc_run_ids"),
        mcmc_prediction_counts=_parse_optional_int_csv(
            arguments.mcmc_prediction_counts,
            "mcmc_prediction_counts",
        ),
        mcmc_window_indices=_parse_optional_int_csv(
            arguments.mcmc_window_indices,
            "mcmc_window_indices",
        ),
    )


def main(argv: Sequence[str] | None = None) -> benchmark.BenchmarkResult:
    """Run all configured cases and write predictions, run records, and aggregates."""
    arguments = parse_arguments(argv)
    config = build_config(arguments)
    result = benchmark.execute_benchmark(
        config,
        output_directory=arguments.output_directory,
        case_runner=_run_case,
    )
    print(f"Wrote benchmark CSV files to {arguments.output_directory}")
    print(
        f"Successful configurations: {(result.runs['status'] == 'success').sum()}/"
        f"{len(result.runs)}"
    )
    return result


def _run_case(case: benchmark.BenchmarkCase):
    """Delegate one case to the existing deterministic or Bayesian workflow."""
    if case.method == "deterministic":
        experiment = deterministic_forecasting.DeterministicRollingExperimentConfig(
            run_id=case.run_id,
            window_mode="expanding",
            observation_count=case.observation_count,
            prediction_count=case.prediction_count,
            position_noise_std_m=case.position_noise_std_m,
            position_noise_seed=case.position_noise_seed,
            stride=1,
        )
        options = validation_cli.DeterministicCTRVEvaluationOptions(
            window_mode="expanding",
            observation_count=case.observation_count,
            prediction_count=case.prediction_count,
            stride=1,
            position_noise_std_m=case.position_noise_std_m,
            position_noise_seed=case.position_noise_seed,
            max_windows=case.max_windows,
            show_plot=False,
        )
        predictions, _ = deterministic_workflow.run_deterministic_ctrv_evaluation(
            data_file=DATA_FILE,
            experiment=experiment,
            options=options,
            show_time_labels=False,
        )
        return predictions

    experiment = bayesian_forecasting.RollingExperimentConfig(
        run_id=case.run_id,
        observation_count=case.observation_count,
        prediction_count=case.prediction_count,
        position_noise_std_m=case.position_noise_std_m,
        position_noise_seed=case.position_noise_seed,
        stride=1,
        inference_method=case.method,
        inference_seed=case.inference_seed,
    )
    options = validation_cli.BayesianCTRVEvaluationOptions(
        observation_count=case.observation_count,
        prediction_count=case.prediction_count,
        stride=1,
        inference_method=case.method,
        vi_algorithm=VI_CONFIG["algorithm"],
        turn_rate_prior_abs_rate_deg_s=PRIORS.turn_rate_prior_abs_rate_deg_s,
        inference_seed=case.inference_seed,
        position_noise_std_m=case.position_noise_std_m,
        position_noise_seed=case.position_noise_seed,
        require_converged=VI_CONFIG["require_converged"],
        max_windows=case.max_windows,
        plot_each_window=False,
    )
    predictions, _ = bayesian_workflow.run_bayesian_ctrv_evaluation(
        data_file=DATA_FILE,
        experiment=experiment,
        priors=PRIORS,
        vi_config=VI_CONFIG,
        mcmc_config=MCMC_CONFIG,
        rbpf_config=RBPF_CONFIG,
        smc_config=SMC_CONFIG,
        fullrank_grad_samples=inference.DEFAULT_FULLRANK_GRAD_SAMPLES,
        credible_interval=0.9,
        sample_trajectories_per_forecast=0,
        options=options,
        show_time_labels=False,
        selected_window_indices=case.selected_window_indices,
        show_plot=False,
    )
    return predictions


def _csv(values: Sequence[object]) -> str:
    return ",".join(map(str, values))


def _parse_string_csv(value: str, name: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise ValueError(f"{name} must contain at least one comma-separated value.")
    return result


def _parse_int_csv(value: str, name: str) -> tuple[int, ...]:
    result = _parse_optional_int_csv(value, name)
    if not result:
        raise ValueError(f"{name} must contain at least one comma-separated integer.")
    return result


def _parse_optional_int_csv(value: str, name: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    try:
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError(f"{name} must contain comma-separated integers.") from error


def _parse_float_csv(value: str, name: str) -> tuple[float, ...]:
    if not value.strip():
        raise ValueError(f"{name} must contain at least one comma-separated number.")
    try:
        return tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError(f"{name} must contain comma-separated numbers.") from error


if __name__ == "__main__":
    main()
