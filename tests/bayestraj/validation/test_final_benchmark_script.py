"""Tests for the final thesis benchmark command-line overrides."""

from experiments.model_evaluation import final_benchmark


def test_cli_overrides_small_benchmark_dimensions_without_editing_source():
    arguments = final_benchmark.parse_arguments(
        [
            "--run-ids",
            "11,12",
            "--methods",
            "deterministic,smc",
            "--prediction-counts",
            "1,3",
            "--noise-levels",
            "0,5",
            "--max-windows",
            "2",
        ]
    )

    config = final_benchmark.build_config(arguments)

    assert config.run_ids == (11, 12)
    assert config.methods == ("deterministic", "smc")
    assert config.prediction_counts == (1, 3)
    assert config.position_noise_std_m == (0.0, 5.0)
    assert config.max_windows == 2
