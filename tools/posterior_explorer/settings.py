"""Form defaults and validation, independent of Tk and inference execution."""

from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path

from bayestraj.inference.configuration import (
    create_default_ctrv_rbpf_config,
    create_default_ctrv_smc_config,
    create_default_mcmc_config,
    create_default_vi_config,
)
from bayestraj.models.bayesian_ctrv import BayesianCTRVPriors
from bayestraj.observations.paths import data_path
from bayestraj.validation.bayesian_ctrv_posterior_dashboard import (
    DEFAULT_PLAYBACK_INTERVAL_MS,
    PosteriorDashboardConfig,
)
from bayestraj.validation.posterior_session import AnalysisSettings

METHODS = ("rbpf", "smc", "vi", "mcmc")
MAIN_DATA_FIELDS = (
    "data_file",
    "run_id",
    "inference_method",
    "start_index",
    "maximum_observation_count",
)
DATA_OPTION_FIELDS = (
    "position_noise_std_m",
    "position_noise_seed",
    "inference_seed",
    "playback_interval_ms",
    "show_legend",
)
LABELS = {
    "data_file": "CSV-Datei",
    "run_id": "Run-ID",
    "inference_method": "Inferenzmethode",
    "start_index": "Startindex (ab 0)",
    "maximum_observation_count": "Max. Beobachtungen (leer = alle)",
    "position_noise_std_m": "Zusätzliches Positionsrauschen [m]",
    "position_noise_seed": "Seed für Positionsrauschen",
    "inference_seed": "Seed für Inferenz",
    "playback_interval_ms": "Wiedergabeintervall [ms]",
    "show_legend": "Legende anzeigen",
    "speed_prior_upper_mps": "Geschwindigkeit: obere Schwelle [m/s]",
    "speed_prior_tail_probability": "Geschwindigkeit: Tail-Wahrscheinlichkeit",
    "turn_rate_prior_abs_heading_change_deg": "Drehrate: Kursänderung [°]",
    "turn_rate_prior_reference_interval_seconds": "Drehrate: Referenzintervall [s]",
    "turn_rate_prior_tail_probability": "Drehrate: Tail-Wahrscheinlichkeit",
    "sigma_position_observation_prior_upper_m": "Beobachtungsrauschen: Schwelle [m]",
    "sigma_position_observation_prior_tail_probability": "Beobachtung: Tail-Wahrscheinlichkeit",
    "sigma_speed_process_prior_upper_mps": "Geschwindigkeitsprozess: Schwelle [m/s]",
    "sigma_speed_process_prior_tail_probability": "Geschwindigkeitsprozess: Tail-Wahrsch.",
    "sigma_turn_rate_process_prior_upper_deg_s": "Drehratenprozess: Schwelle [°/s]",
    "sigma_turn_rate_process_prior_tail_probability": "Drehratenprozess: Tail-Wahrscheinlichkeit",
    "particle_count": "Partikelanzahl",
    "posterior_draw_count": "Posterior-Draws",
    "resample_ess_fraction": "Resampling: ESS-Anteil",
    "rejuvenation_scale": "Rejuvenation-Skala",
    "require_converged": "VI-Konvergenz verlangen",
}


@dataclass(frozen=True)
class ExplorerSettings:
    """Analysis snapshot and independent presentation preferences."""

    analysis: AnalysisSettings
    playback_interval_ms: int
    show_legend: bool


def _defaults():
    return {
        "data": {
            "data_file": str(
                data_path(
                    "raw/processed_ship_data_2026-01-10T00-00-00+01-00_"
                    "2026-02-02T00-00-00+01-00_10.csv"
                )
            ),
            "run_id": 102,
            "inference_method": "rbpf",
            "start_index": 0,
            "maximum_observation_count": "",
            "position_noise_std_m": 5.0,
            "position_noise_seed": 2026,
            "inference_seed": 42,
            "playback_interval_ms": DEFAULT_PLAYBACK_INTERVAL_MS,
            "show_legend": True,
        },
        "priors": asdict(BayesianCTRVPriors()),
        "rbpf": asdict(create_default_ctrv_rbpf_config()),
        "smc": asdict(create_default_ctrv_smc_config()),
        "vi": create_default_vi_config(),
        "mcmc": create_default_mcmc_config(),
    }


def default_form_values():
    """Return independent strings/booleans suitable for editable Tk variables."""
    return {
        group: {
            key: value if isinstance(value, bool) else str(value)
            for key, value in fields.items()
        }
        for group, fields in _defaults().items()
    }


def _parse_group(values, defaults):
    parsed = {}
    for key, default in defaults.items():
        value = values[key]
        try:
            if isinstance(default, bool):
                if not isinstance(value, bool):
                    raise ValueError("expected a boolean")
            elif isinstance(default, int):
                value = int(str(value).strip())
            elif isinstance(default, float):
                value = float(value)
                if not isfinite(value):
                    raise ValueError("expected a finite number")
            else:
                value = str(value).strip()
        except (ValueError, TypeError) as error:
            raise ValueError(f"Ungültiger Wert für {LABELS.get(key, key)}.") from error
        parsed[key] = value
    return parsed


def _minimum(fields, key, minimum):
    if fields[key] < minimum:
        raise ValueError(f"{LABELS.get(key, key)} muss mindestens {minimum} sein.")


def _validate_batch(method, fields):
    if method == "vi":
        if fields["algorithm"] not in ("meanfield", "fullrank"):
            raise ValueError("VI algorithm muss meanfield oder fullrank sein.")
        for key in ("iter", "grad_samples", "elbo_samples", "eval_elbo"):
            _minimum(fields, key, 1)
        _minimum(fields, "draws", 2)
        _minimum(fields, "adapt_iter", 0)
        for key in ("eta", "tol_rel_obj"):
            if fields[key] <= 0:
                raise ValueError(f"{key} muss größer als 0 sein.")
    elif method == "mcmc":
        for key in ("chains", "parallel_chains", "iter_sampling", "max_treedepth"):
            _minimum(fields, key, 1)
        _minimum(fields, "iter_warmup", 0)
        if fields["chains"] * fields["iter_sampling"] < 2:
            raise ValueError("Mindestens zwei MCMC-Draws sind erforderlich.")
        if not 0 < fields["adapt_delta"] < 1:
            raise ValueError("adapt_delta muss zwischen 0 und 1 liegen.")
        if fields["parallel_chains"] > fields["chains"]:
            raise ValueError("parallel_chains darf chains nicht überschreiten.")


def _validate_data_options(fields):
    for key in ("position_noise_std_m", "position_noise_seed", "inference_seed"):
        _minimum(fields, key, 0)
    _minimum(fields, "playback_interval_ms", 1)


def validate_dialog_values(group, values):
    """Validate one dialog independently, without needing a valid CSV selection."""
    defaults = _defaults()
    if group not in defaults:
        raise ValueError(f"Unbekannter Einstellungsbereich: {group}")
    fields = defaults[group]
    if group == "data":
        fields = {key: fields[key] for key in DATA_OPTION_FIELDS}
    parsed = _parse_group(values, fields)
    if group == "data":
        _validate_data_options(parsed)
    elif group == "priors":
        BayesianCTRVPriors(**parsed)
    elif group == "rbpf":
        type(create_default_ctrv_rbpf_config())(**parsed)
    elif group == "smc":
        type(create_default_ctrv_smc_config())(**parsed)
    else:
        _validate_batch(group, parsed)
    return parsed


def parse_settings(values) -> ExplorerSettings:
    """Validate edits before replacing a working analysis; ignore inactive methods."""
    defaults = _defaults()
    data = _parse_group(values["data"], defaults["data"])
    data_file = Path(data.pop("data_file"))
    if not data_file.is_file():
        raise ValueError("Bitte eine vorhandene CSV-Datei auswählen.")
    method = data["inference_method"]
    if method not in METHODS:
        raise ValueError("Inferenzmethode muss rbpf, smc, vi oder mcmc sein.")
    maximum = data["maximum_observation_count"]
    try:
        data["maximum_observation_count"] = int(maximum) if maximum else None
    except ValueError as error:
        raise ValueError("Max. Beobachtungen muss eine ganze Zahl sein.") from error
    if maximum:
        _minimum(data, "maximum_observation_count", 3)
    for key in ("run_id", "start_index"):
        _minimum(data, key, 0)
    _validate_data_options(data)
    interval = data.pop("playback_interval_ms")
    legend = data.pop("show_legend")
    priors = BayesianCTRVPriors(**_parse_group(values["priors"], defaults["priors"]))
    defaults[method] = _parse_group(values[method], defaults[method])
    _validate_batch(method, defaults[method])
    rbpf = type(create_default_ctrv_rbpf_config())(**defaults["rbpf"])
    smc = type(create_default_ctrv_smc_config())(**defaults["smc"])
    return ExplorerSettings(
        AnalysisSettings(
            data_file=data_file,
            experiment=PosteriorDashboardConfig(**data),
            priors=priors,
            vi_config=defaults["vi"],
            mcmc_config=defaults["mcmc"],
            rbpf_config=rbpf,
            smc_config=smc,
        ),
        playback_interval_ms=interval,
        show_legend=legend,
    )
