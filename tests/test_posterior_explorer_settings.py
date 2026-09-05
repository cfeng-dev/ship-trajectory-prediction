"""GUI settings are validated without creating a window or starting inference."""

import pytest
from posterior_explorer import settings
from posterior_explorer.cli import main


def test_command_help_does_not_open_a_window(capsys):
    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0
    assert "posterior explorer" in capsys.readouterr().out


@pytest.fixture
def form(tmp_path):
    values = settings.default_form_values()
    data_file = tmp_path / "trajectory.csv"
    data_file.touch()
    values["data"]["data_file"] = str(data_file)
    return values


@pytest.mark.parametrize("method", ["rbpf", "smc", "vi", "mcmc"])
def test_settings_preserve_selected_method_and_units(form, method):
    form["data"]["inference_method"] = method
    form["priors"]["sigma_turn_rate_process_prior_upper_deg_s"] = "7.5"
    result = settings.parse_settings(form)
    assert result.analysis.experiment.inference_method == method
    assert result.analysis.experiment.maximum_observation_count is None
    assert result.analysis.priors.sigma_turn_rate_process_prior_upper_deg_s == 7.5
    assert result.playback_interval_ms > 0


@pytest.mark.parametrize(
    ("group", "field", "value"),
    [
        ("data", "start_index", "-1"),
        ("data", "run_id", "1.2"),
        ("data", "maximum_observation_count", "2"),
        ("data", "position_noise_std_m", "nan"),
        ("data", "playback_interval_ms", "0"),
        ("priors", "speed_prior_tail_probability", "1"),
        ("rbpf", "posterior_draw_count", "1"),
        ("rbpf", "resample_ess_fraction", "1.1"),
        ("rbpf", "rejuvenation_scale", "inf"),
    ],
)
def test_invalid_settings_are_rejected(form, group, field, value):
    form[group][field] = value
    with pytest.raises(ValueError):
        settings.parse_settings(form)


def test_inactive_method_fields_do_not_block_analysis(form):
    form["vi"]["iter"] = "unfinished edit"
    settings.parse_settings(form)
    form["data"]["inference_method"] = "vi"
    with pytest.raises(ValueError, match="iter"):
        settings.parse_settings(form)


def test_batch_options_are_forwarded_without_hidden_overrides(form):
    form["data"]["inference_method"] = "vi"
    form["vi"]["algorithm"] = "fullrank"
    form["vi"]["grad_samples"] = "7"
    form["vi"]["require_converged"] = True
    result = settings.parse_settings(form)
    assert result.analysis.vi_config["algorithm"] == "fullrank"
    assert result.analysis.vi_config["grad_samples"] == 7
    assert result.analysis.vi_config["require_converged"] is True


def test_defaults_are_independent_and_missing_file_is_rejected(form):
    values = settings.default_form_values()
    values["rbpf"]["particle_count"] = "2"
    assert settings.default_form_values()["rbpf"]["particle_count"] == "4000"
    form["data"]["data_file"] += ".missing"
    with pytest.raises(ValueError, match="CSV"):
        settings.parse_settings(form)
