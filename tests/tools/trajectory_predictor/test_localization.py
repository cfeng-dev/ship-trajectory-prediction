"""Language changes affect presentation without changing analysis settings."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from trajectory_predictor import localization
from trajectory_predictor.dashboard import normalize_prediction_region_style
from trajectory_predictor.settings import (
    METHOD_DISPLAY_LABELS,
    default_form_values,
    normalize_inference_method,
    parse_settings,
)


@pytest.mark.parametrize(
    ("english", "german"),
    [
        ("Settings", "Einstellungen"),
        ("Start analysis", "Analyse starten"),
        ("Forecast steps (0 = off)", "Prognoseschritte (0 = aus)"),
        ("Forecast (median)", "Prognose (Median)"),
        (
            "Possible future trajectories (n = 20)",
            "Mögliche zukünftige Trajektorien (n = 20)",
        ),
        (
            "Bayesian CTRV posterior update — N = 4",
            "Bayessches CTRV: Posterior-Update — N = 4",
        ),
        ("1.40°/s (derived)", "1.40°/s (abgeleitet)"),
    ],
)
def test_translation_round_trip_preserves_numbers_and_returns_to_english(
    english, german
):
    assert localization.translate(english, "de") == german
    assert localization.translate(german, "en") == english
    assert localization.translate(german, "de") == german


def test_language_validation_and_unknown_technical_details():
    with pytest.raises(ValueError, match="en.*de"):
        localization.normalize_language("fr")
    path = r"D:\data\reference.csv"
    assert localization.translate(path, "de") == path
    assert localization.translate("3.50 m/s", "de") == "3.50 m/s"


def test_localized_choices_parse_to_the_same_analysis_configuration(tmp_path):
    data_file = tmp_path / "trajectory.csv"
    data_file.write_text("time,x,y\n0,0,0\n1,1,1\n2,2,2\n", encoding="utf-8")
    original = default_form_values()
    original["data"]["data_file"] = str(data_file)
    localized = {group: dict(values) for group, values in original.items()}
    localized["data"]["inference_method"] = localization.translate(
        METHOD_DISPLAY_LABELS["sequential_vi"], "de"
    )
    localized["data"]["prediction_region_style"] = localization.translate(
        "Horizon density region", "de"
    )
    original["data"]["inference_method"] = "sequential_vi"
    assert parse_settings(localized) == parse_settings(original)
    assert (
        normalize_inference_method(localized["data"]["inference_method"])
        == "sequential_vi"
    )
    assert (
        normalize_prediction_region_style(localized["data"]["prediction_region_style"])
        == "funnel"
    )


def test_figure_translation_preserves_data_limits_and_reverses():
    figure, axis = plt.subplots()
    (line,) = axis.plot([1, 2], [3, 4], label="Reference trajectory")
    axis.legend()
    axis.set_title("Speed posterior")
    axis.set_xlabel("Easting x [m]")
    axis.set_ylabel("Density")
    limits = (axis.get_xlim(), axis.get_ylim())
    try:
        localization.localize_figure(figure, "de")
        assert axis.get_title() == "Posterior der Geschwindigkeit"
        assert axis.get_ylabel() == "Dichte"
        assert axis.get_legend().get_texts()[0].get_text() == "Referenztrajektorie"
        assert (axis.get_xlim(), axis.get_ylim()) == limits
        assert list(line.get_ydata()) == [3, 4]
        localization.localize_figure(figure, "en")
        assert axis.get_title() == "Speed posterior"
        assert axis.get_xlabel() == "Easting x [m]"
        assert axis.get_legend().get_texts()[0].get_text() == "Reference trajectory"
    finally:
        plt.close(figure)


def test_help_has_german_translations_for_every_description():
    from trajectory_predictor.help import POSTERIOR_HELP_SECTIONS

    for title, rows in POSTERIOR_HELP_SECTIONS:
        assert localization.translate(title, "de") != title
        for _, description in rows:
            assert localization.translate(description, "de") != description


def test_all_catalog_labels_round_trip():
    for english in localization.TRANSLATIONS:
        german = localization.translate(english, "de")
        assert localization.translate(german, "en") == english
    for label in (
        "Horizon density region",
        "Horizon density contour",
        "Per-step density region",
        "Per-step density contour",
        "Covariance ellipse",
    ):
        for percent in (50, 90):
            english = f"{label} ({percent} %)"
            german = localization.translate(english, "de")
            assert german != english
            assert str(percent) in german
            assert localization.translate(german, "en") == english


def test_sidebar_and_dialog_labels_are_covered_by_the_catalog():
    from trajectory_predictor import dashboard, dialogs, settings

    labels = [
        *settings.LABELS.values(),
        *settings.METHOD_DISPLAY_OPTIONS,
        *settings.PREDICTION_REGION_STYLE_DISPLAY_OPTIONS,
        *dashboard.PARAMETER_PRIOR_TITLES.values(),
        *dashboard.PARAMETER_POSTERIOR_TITLES.values(),
        *(title for title, _ in dialogs.PRIOR_FIELD_GROUPS),
    ]
    assert all(label in localization.TRANSLATIONS for label in labels)


def test_localized_widget_labels_leave_entry_values_untouched():
    class Widget:
        def __init__(self, kind, *, master=None, **options):
            self.kind, self.master, self.options = kind, master, options
            self.children = []
            self.variables = master.variables if master else {}
            self.caption = "Bayesian CTRV — Ship Trajectory Predictor"
            self.selected = options.pop("selected", "")
            if master:
                master.children.append(self)

        def keys(self):
            return self.options.keys()

        def cget(self, key):
            return self.options.get(key, 0)

        def configure(self, **options):
            self.options.update(options)

        def winfo_class(self):
            return self.kind

        def winfo_children(self):
            return self.children

        def getvar(self, name):
            return self.variables[name]

        def setvar(self, name, value):
            self.variables[name] = value

        def get(self):
            return self.selected

        def set(self, value):
            self.selected = value

        def title(self, value=None):
            if value is not None:
                self.caption = value
            return self.caption

    root = Widget("Tk")
    root._trajectory_language = "de"
    root.variables.update(input="Settings", display="1.40°/s (derived)", seed="2026")
    label = Widget("Label", master=root, text="Reference state", width=0)
    Widget("Label", master=root, textvariable="display")
    Widget("Entry", master=root, textvariable="input")
    Widget("Entry", master=root, textvariable="seed")
    choice = Widget(
        "TCombobox",
        master=root,
        values=("Horizon density region", "Covariance ellipses"),
        selected="Horizon density region",
    )
    localization.localize_widgets(root)
    assert label.cget("text") == "Referenzzustand"
    assert root.variables == {
        "input": "Settings",
        "display": "1.40°/s (abgeleitet)",
        "seed": "2026",
    }
    assert normalize_prediction_region_style(choice.get()) == "funnel"
    localization.localize_widgets(root, "en")
    assert label.cget("text") == "Reference state"
    assert choice.get() == "Horizon density region"
    assert root.variables["display"] == "1.40°/s (derived)"


def test_translated_validation_messages_keep_field_and_numeric_meaning():
    message = "Forecast steps (0 = off) must be at least 0."
    translated = localization.translate(message, "de")
    assert translated == "Prognoseschritte (0 = aus) muss mindestens 0 betragen."
    assert localization.translate(translated, "en") == message


def test_navigator_language_switch_keeps_cache_playback_and_zoom():
    from trajectory_predictor.dashboard import (
        PosteriorDashboardTrajectory,
        create_sequential_posterior_dashboard_figure,
    )

    from bayestraj.models.bayesian_ctrv import BayesianCTRVPriors

    trajectory = PosteriorDashboardTrajectory(
        reference_x=np.arange(5.0),
        reference_y=np.arange(5.0),
        observed_x=np.arange(5.0),
        observed_y=np.arange(5.0),
    )
    figure, navigator = create_sequential_posterior_dashboard_figure(
        trajectory,
        BayesianCTRVPriors(),
        maximum_observation_count=4,
        minimum_posterior_observation_count=2,
        request_update=lambda _count: None,
    )
    try:
        cache = navigator._updates_by_count
        limits = (
            navigator.trajectory_axis.get_xlim(),
            navigator.trajectory_axis.get_ylim(),
        )
        navigator.set_language("de")
        assert navigator._updates_by_count is cache
        assert not navigator.is_playing
        assert navigator.observation_count == 0
        assert (
            navigator.trajectory_axis.get_xlim(),
            navigator.trajectory_axis.get_ylim(),
        ) == limits
        assert (
            navigator.posterior_axes[0].get_title()
            == "Anfangsprior der Geschwindigkeit"
        )
        navigator._select_parameter_group("Unsicherheiten")
        assert navigator._parameter_group == "noise"
        assert navigator.posterior_axes[0].get_ylabel() == "Dichte"
        navigator._select_parameter_group("Bewegungszustand")
        navigator.set_language("en")
        assert navigator.posterior_axes[0].get_title() == "Initial speed prior"
        assert navigator._updates_by_count is cache
    finally:
        navigator.disconnect()
        plt.close(figure)
