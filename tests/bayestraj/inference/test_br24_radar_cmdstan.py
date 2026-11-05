"""Optional real BR24 inference checks with an explicit missing radar cycle."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bayestraj.forecasting.bayesian_ctrv import RollingExperimentConfig
from bayestraj.inference.br24_radar_tracker import BR24BayesianRadarTracker
from bayestraj.inference.ctrv_sequential_vi import SequentialVIConfig
from bayestraj.inference.ctrv_smc import SequentialMonteCarloCTRVConfig
from bayestraj.models.bayesian_ctrv import BayesianCTRVPriors
from bayestraj.validation.bayesian_ctrv_workflow import run_bayesian_ctrv_evaluation
from bayestraj.validation.cli import BayesianCTRVEvaluationOptions

RUN_INTEGRATION = os.environ.get("RUN_CMDSTAN_INTEGRATION") == "1"
ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.skipif(not RUN_INTEGRATION, reason="Set RUN_CMDSTAN_INTEGRATION=1.")
@pytest.mark.parametrize("method", ["smc", "vi", "mcmc", "sequential_vi"])
def test_real_br24_methods_fit_and_forecast_across_a_missing_cycle(
    tmp_path, monkeypatch, method
):
    observation_scores = []
    process_cycle = BR24BayesianRadarTracker.process_cycle

    def process_and_capture(tracker, cycle):
        step = process_cycle(tracker, cycle)
        observation_scores.append(
            (step.detection_id, tracker.filter.last_log_predictive_density)
        )
        return step

    monkeypatch.setattr(BR24BayesianRadarTracker, "process_cycle", process_and_capture)
    data = pd.read_csv(ROOT / "data/processed/BR24Dataset20131107_with_reference.csv")
    cycle_ids = data["cycle_counter"].drop_duplicates().iloc[:7]
    data = data[data["cycle_counter"].isin(cycle_ids)].copy()
    data.loc[
        data["cycle_counter"] == cycle_ids.iloc[1],
        [
            "detection_id",
            "range",
            "azimuth",
            "std_range",
            "std_azimuth",
            "x_relative",
            "y_relative",
        ],
    ] = np.nan
    input_csv = tmp_path / "radar_with_miss.csv"
    data.to_csv(input_csv, index=False)
    options = BayesianCTRVEvaluationOptions(
        observation_count=5,
        prediction_count=2,
        stride=1,
        inference_method=method,
        vi_algorithm="fullrank",
        turn_rate_prior_abs_rate_deg_s=10.0,
        inference_seed=42,
        position_noise_std_m=0.0,
        position_noise_seed=2026,
        require_converged=False,
        max_windows=3,
        plot_each_window=False,
    )
    vi_config = dict(
        algorithm="fullrank",
        iter=2_000,
        grad_samples=20,
        elbo_samples=50,
        eta=1.0,
        adapt_iter=50,
        tol_rel_obj=0.05,
        eval_elbo=100,
        draws=64,
        require_converged=False,
        show_console=False,
    )

    predictions, summary = run_bayesian_ctrv_evaluation(
        data_file=input_csv,
        experiment=RollingExperimentConfig(
            observation_count=5,
            prediction_count=2,
            stride=1,
            inference_method=method,
            position_noise_std_m=0.0,
            position_noise_seed=2026,
            inference_seed=42,
        ),
        priors=BayesianCTRVPriors(),
        vi_config=vi_config,
        mcmc_config=dict(
            chains=1,
            parallel_chains=1,
            iter_warmup=100,
            iter_sampling=100,
            adapt_delta=0.9,
            max_treedepth=10,
            show_console=False,
        ),
        smc_config=SequentialMonteCarloCTRVConfig(
            particle_count=64, posterior_draw_count=64
        ),
        sequential_vi_config=SequentialVIConfig(n_bootstrap=3, **vi_config),
        fullrank_grad_samples=20,
        credible_interval=0.9,
        sample_trajectories_per_forecast=0,
        options=options,
        show_plot=False,
    )

    assert summary.inference_method == method
    assert summary.window_count == 3
    assert len(predictions) == 6
    assert set(predictions["inference_method"]) == {method}
    assert predictions["radar_processed_cycle_count"].tolist() == [2, 2, 3, 3, 4, 4]
    assert observation_scores[0] == (None, None)
    assert all(
        np.isfinite(score)
        for detection_id, score in observation_scores
        if detection_id is not None
    )
    assert not predictions.iloc[-2:]["is_prior_predictive"].any()
    assert np.all(
        np.isfinite(
            predictions[
                [
                    "x_median",
                    "y_median",
                    "energy_score_m",
                    "negative_log_predictive_density",
                ]
            ].to_numpy(dtype=float)
        )
    )
    assert set(tmp_path.iterdir()) == {input_csv}
