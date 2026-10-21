"""Interactive online forecast view for the optional local BR24 IMM tracker.

The player deliberately never passes BR24 reference labels into the tracker.  If
the flattened input contains reference columns, they are shown only as a clearly
labelled evaluation overlay.
"""

from __future__ import annotations

import runpy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from matplotlib.widgets import Button, CheckButtons, Slider
from scipy.stats import chi2

import bayestraj.observations.br24 as br24
import bayestraj.observations.paths as paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMM_PYTHON = PROJECT_ROOT / "external/IMM/br24_imm.py"
DATA_FILE = paths.data_path("processed/BR24Dataset20131107_with_reference.csv")

# Keep the presentation aligned with the rolling Bayesian CTRV comparison.
INITIAL_OBSERVATION_COUNT = 10
PREDICTION_COUNT = 2
PLAYBACK_INTERVAL_MS = 1_000
FOLLOW_SHIP_VIEW_SPAN_METERS = 300.0
CONFIDENCE_LEVELS = (0.5, 0.9)
MODE_NAMES = ("CV", "CTRV", "CA")
MODE_COLORS = ("#3C78A8", "#D27D2D", "#5A9C65")


@dataclass(frozen=True, slots=True)
class IMMPlayerConfig:
    """Fixed local settings for a BR24 IMM playback session."""

    initial_observation_count: int = INITIAL_OBSERVATION_COUNT
    prediction_count: int = PREDICTION_COUNT
    playback_interval_ms: int = PLAYBACK_INTERVAL_MS
    follow_ship_view_span_meters: float = FOLLOW_SHIP_VIEW_SPAN_METERS

    def __post_init__(self) -> None:
        for name in (
            "initial_observation_count",
            "prediction_count",
            "playback_interval_ms",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        span = float(self.follow_ship_view_span_meters)
        if not np.isfinite(span) or span <= 0.0:
            raise ValueError(
                "follow_ship_view_span_meters must be a positive finite number."
            )
        object.__setattr__(self, "follow_ship_view_span_meters", span)


@dataclass(frozen=True, slots=True)
class IMMPlayerStage:
    """One cached online IMM posterior and its newly refreshed forecast."""

    observation_count: int
    cycle_counter: int
    observation_time_seconds: float
    detection_id: int | None
    tracked_positions: np.ndarray
    forecast: Any


class IMMPlaybackStore:
    """Advance a local IMM one radar cycle at a time and cache display stages."""

    def __init__(self, config: IMMPlayerConfig) -> None:
        self.config = config
        self.cycles = br24.read_br24_tracking_input(DATA_FILE)
        self.reference = br24.read_br24_tracking_reference(DATA_FILE)
        if self.reference is not None:
            self._validate_reference_alignment()
        self.times = np.asarray(
            [cycle.time_seconds for cycle in self.cycles], dtype=float
        )
        self.maximum_observation_count = len(self.cycles) - config.prediction_count
        if not 1 <= config.initial_observation_count <= self.maximum_observation_count:
            raise ValueError(
                "initial_observation_count must leave room for the requested "
                "prediction horizon."
            )
        if not self.cycles[0].detections:
            raise ValueError(
                "The earliest BR24 radar cycle has no detection to initialize."
            )

        tracker_type = _load_optional_imm_tracker()
        initial_detection = self.cycles[0].detections[0]
        self._tracker = tracker_type.initialize(
            self.cycles[0], initial_detection_id=initial_detection.detection_id
        )
        self._processed_cycle_count = 1
        self._tracked_positions = [
            np.asarray(self._tracker.estimated_position, dtype=float)
        ]
        self._detection_ids = [initial_detection.detection_id]
        self._stages: dict[int, IMMPlayerStage] = {}

    def stage_for(self, observation_count: int) -> IMMPlayerStage:
        """Return the online forecast refreshed after exactly ``N`` observations."""
        observation_count = int(observation_count)
        if not 1 <= observation_count <= self.maximum_observation_count:
            raise ValueError(
                "observation_count must leave room for the configured forecast."
            )
        if observation_count in self._stages:
            return self._stages[observation_count]
        if observation_count < self._processed_cycle_count:
            raise RuntimeError("An earlier IMM stage was not cached.")

        while self._processed_cycle_count < observation_count:
            step = self._tracker.process_cycle(self.cycles[self._processed_cycle_count])
            self._tracked_positions.append(
                np.asarray(self._tracker.estimated_position, dtype=float)
            )
            self._detection_ids.append(step.detection_id)
            self._processed_cycle_count += 1

        future_times = self.times[
            observation_count : observation_count + self.config.prediction_count
        ]
        forecast = self._tracker.forecast(future_times)
        stage = IMMPlayerStage(
            observation_count=observation_count,
            cycle_counter=self.cycles[observation_count - 1].cycle_counter,
            observation_time_seconds=float(self._tracker.last_observation_time_seconds),
            detection_id=self._detection_ids[-1],
            tracked_positions=np.asarray(self._tracked_positions, dtype=float).copy(),
            forecast=forecast,
        )
        self._stages[observation_count] = stage
        return stage

    def _validate_reference_alignment(self) -> None:
        """Keep evaluation labels synchronized without exposing them to the IMM."""
        cycle_counters = np.asarray(
            [cycle.cycle_counter for cycle in self.cycles], dtype=int
        )
        cycle_times = np.asarray(
            [cycle.time_seconds for cycle in self.cycles], dtype=float
        )
        if not np.array_equal(
            self.reference.cycle_counters, cycle_counters
        ) or not np.array_equal(self.reference.time_seconds, cycle_times):
            raise ValueError(
                "BR24 reference labels must align exactly with radar cycles."
            )


class BR24IMMPlayer:
    """Matplotlib player styled after the trajectory predictor's route panel."""

    def __init__(self, store: IMMPlaybackStore) -> None:
        self.store = store
        self.observation_count = store.config.initial_observation_count
        self._playing = False
        self._follow_ship = True
        self._show_reference = store.reference is not None

        self.figure = plt.figure(figsize=(15.0, 8.5), facecolor="#F7F9FC")
        self.route_axis = self.figure.add_axes((0.065, 0.205, 0.62, 0.70))
        self.mode_axis = self.figure.add_axes((0.745, 0.485, 0.22, 0.42))
        self.status_axis = self.figure.add_axes((0.745, 0.205, 0.22, 0.20))
        self.status_axis.set_axis_off()

        playback_axis = self.figure.add_axes((0.04, 0.065, 0.085, 0.055))
        self.playback_button = Button(playback_axis, "Start", color="#EEF3F8")
        slider_axis = self.figure.add_axes((0.17, 0.076, 0.47, 0.033))
        self.slider = Slider(
            slider_axis,
            "N",
            store.config.initial_observation_count,
            store.maximum_observation_count,
            valinit=self.observation_count,
            valstep=1,
            color="#3C78A8",
        )
        options_axis = self.figure.add_axes((0.73, 0.035, 0.24, 0.095))
        options_axis.set_frame_on(False)
        self.options = CheckButtons(
            options_axis,
            ("Follow ship", "Show reference"),
            (self._follow_ship, self._show_reference),
            label_props={"fontsize": [10, 10]},
        )

        self._playback_timer = self.figure.canvas.new_timer(
            interval=store.config.playback_interval_ms
        )
        self._playback_timer.add_callback(self.advance_playback)
        self.playback_button.on_clicked(self._toggle_playback)
        self.slider.on_changed(self._on_slider_change)
        self.options.on_clicked(self._on_option_change)
        self.figure.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.figure.canvas.mpl_connect("close_event", self._on_close)
        self._draw()

    def show(self) -> None:
        """Show the player until its window is closed."""
        plt.show(block=True)

    def advance_playback(self) -> None:
        """Refresh the online forecast after one newly received radar cycle."""
        if self.observation_count >= self.store.maximum_observation_count:
            self._set_playing(False)
            return
        self.slider.set_val(self.observation_count + 1)

    def _toggle_playback(self, _event) -> None:
        self._set_playing(not self._playing)

    def _set_playing(self, playing: bool) -> None:
        self._playing = bool(playing)
        self.playback_button.label.set_text("Pause" if self._playing else "Start")
        if self._playing:
            self._playback_timer.start()
        else:
            self._playback_timer.stop()
        self.figure.canvas.draw_idle()

    def _on_slider_change(self, value) -> None:
        observation_count = int(round(value))
        if observation_count == self.observation_count:
            return
        self.observation_count = observation_count
        self._draw()

    def _on_option_change(self, label: str) -> None:
        if label == "Follow ship":
            self._follow_ship = not self._follow_ship
        elif label == "Show reference":
            self._show_reference = not self._show_reference
        self._draw()

    def _on_key_press(self, event) -> None:
        if event.key == "space":
            self._set_playing(not self._playing)
        elif event.key == "left":
            self.slider.set_val(max(self.slider.valmin, self.observation_count - 1))
        elif event.key == "right":
            self.slider.set_val(min(self.slider.valmax, self.observation_count + 1))

    def _on_close(self, _event) -> None:
        self._playback_timer.stop()

    def _draw(self) -> None:
        stage = self.store.stage_for(self.observation_count)
        self._draw_route(stage)
        self._draw_mode_probabilities(stage)
        self._draw_status(stage)
        self.figure.suptitle(
            "Python IMM BR24 online forecast",
            x=0.065,
            y=0.975,
            ha="left",
            fontsize=16,
            fontweight="bold",
            color="#1C2B39",
        )
        self.figure.canvas.draw_idle()

    def _draw_route(self, stage: IMMPlayerStage) -> None:
        axis = self.route_axis
        axis.clear()
        axis.set_facecolor("#FCFDFE")
        axis.grid(True, color="#DDE5EC", linewidth=0.7)
        axis.set_xlabel("East [m]")
        axis.set_ylabel("North [m]")
        axis.set_title(
            "Online IMM track and refreshed two-step forecast",
            loc="left",
            fontsize=11,
            fontweight="bold",
        )
        axis.set_aspect("equal", adjustable="box")

        if self._show_reference and self.store.reference is not None:
            axis.plot(
                self.store.reference.x_meters,
                self.store.reference.y_meters,
                color="#6E7681",
                linestyle="--",
                linewidth=1.5,
                label="Reference (evaluation only)",
                zorder=1,
            )

        tracked = stage.tracked_positions
        axis.plot(
            tracked[:, 0],
            tracked[:, 1],
            color="#2878B5",
            linewidth=2.2,
            label="IMM track",
            zorder=3,
        )
        axis.scatter(
            tracked[-1, 0],
            tracked[-1, 1],
            s=55,
            color="#1A4E75",
            edgecolor="white",
            linewidth=1.0,
            label="Current posterior",
            zorder=5,
        )

        means = np.asarray(stage.forecast.position_means, dtype=float)
        covariances = np.asarray(stage.forecast.position_covariances, dtype=float)
        forecast_route = np.vstack((tracked[-1], means))
        axis.plot(
            forecast_route[:, 0],
            forecast_route[:, 1],
            color="#D95F02",
            linestyle="--",
            marker="o",
            markersize=4,
            linewidth=2.2,
            label="IMM forecast mean",
            zorder=4,
        )
        for index, (mean, covariance) in enumerate(
            zip(means, covariances, strict=True), start=1
        ):
            self._draw_covariance_ellipses(axis, mean, covariance, index)
        self._set_route_limits(axis, tracked, means)
        axis.legend(loc="best", fontsize=8, framealpha=0.95)

    def _draw_covariance_ellipses(
        self, axis, mean, covariance, horizon_step: int
    ) -> None:
        covariance = _regularize_covariance(covariance)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        direction = eigenvectors[:, -1]
        angle = float(np.degrees(np.arctan2(direction[1], direction[0])))
        for probability, alpha in zip(CONFIDENCE_LEVELS, (0.11, 0.20), strict=True):
            radius = float(np.sqrt(chi2.ppf(probability, df=2)))
            ellipse = Ellipse(
                xy=mean,
                width=2.0 * radius * np.sqrt(eigenvalues[-1]),
                height=2.0 * radius * np.sqrt(eigenvalues[0]),
                angle=angle,
                facecolor="#F6A04D",
                edgecolor="#D95F02",
                linewidth=1.0,
                alpha=alpha,
                zorder=2,
            )
            axis.add_patch(ellipse)
        axis.annotate(
            f"+{horizon_step}",
            xy=mean,
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color="#9C4000",
            zorder=6,
        )

    def _set_route_limits(self, axis, tracked: np.ndarray, means: np.ndarray) -> None:
        center = np.asarray(tracked[-1], dtype=float)
        if self._follow_ship:
            half_span = self.store.config.follow_ship_view_span_meters / 2.0
            axis.set_xlim(center[0] - half_span, center[0] + half_span)
            axis.set_ylim(center[1] - half_span, center[1] + half_span)
            return

        visible = [tracked, means]
        if self._show_reference and self.store.reference is not None:
            visible.append(
                np.column_stack(
                    (self.store.reference.x_meters, self.store.reference.y_meters)
                )
            )
        points = np.vstack(visible)
        lower = np.min(points, axis=0)
        upper = np.max(points, axis=0)
        margin = np.maximum(10.0, 0.05 * (upper - lower))
        axis.set_xlim(lower[0] - margin[0], upper[0] + margin[0])
        axis.set_ylim(lower[1] - margin[1], upper[1] + margin[1])

    def _draw_mode_probabilities(self, stage: IMMPlayerStage) -> None:
        axis = self.mode_axis
        axis.clear()
        axis.set_facecolor("#FCFDFE")
        axis.grid(True, axis="y", color="#DDE5EC", linewidth=0.7)
        offsets = np.asarray(stage.forecast.time_seconds, dtype=float) - (
            stage.observation_time_seconds
        )
        modes = np.asarray(stage.forecast.mode_probabilities, dtype=float)
        for index, (name, color) in enumerate(
            zip(MODE_NAMES, MODE_COLORS, strict=True)
        ):
            axis.plot(
                offsets,
                modes[:, index],
                marker="o",
                color=color,
                linewidth=2.0,
                label=name,
            )
        axis.set_title(
            "Forecast mode probabilities", loc="left", fontsize=11, fontweight="bold"
        )
        axis.set_xlabel("Horizon [s]")
        axis.set_ylabel("Probability")
        axis.set_ylim(-0.02, 1.02)
        axis.legend(loc="best", fontsize=8, framealpha=0.95)

    def _draw_status(self, stage: IMMPlayerStage) -> None:
        axis = self.status_axis
        axis.clear()
        axis.set_axis_off()
        text = [
            "ONLINE UPDATE",
            f"N = {stage.observation_count} radar cycles",
            f"Cycle = {stage.cycle_counter}",
            f"Detection ID = {stage.detection_id if stage.detection_id is not None else 'none'}",
            f"Forecast = {self.store.config.prediction_count} steps",
        ]
        if self.store.reference is not None:
            target_indices = np.arange(
                stage.observation_count,
                stage.observation_count + self.store.config.prediction_count,
            )
            targets = np.column_stack(
                (
                    self.store.reference.x_meters[target_indices],
                    self.store.reference.y_meters[target_indices],
                )
            )
            errors = np.linalg.norm(
                np.asarray(stage.forecast.position_means, dtype=float) - targets,
                axis=1,
            )
            text.extend(
                (
                    f"ADE = {errors.mean():.1f} m",
                    f"FDE = {errors[-1]:.1f} m",
                    "",
                    "Reference is evaluation only.",
                )
            )
        axis.text(
            0.0,
            1.0,
            "\n".join(text),
            va="top",
            ha="left",
            fontsize=10,
            linespacing=1.55,
            color="#1C2B39",
            bbox={
                "boxstyle": "round,pad=0.55",
                "facecolor": "#EEF3F8",
                "edgecolor": "#C7D5E2",
            },
        )


def create_br24_imm_player(config: IMMPlayerConfig | None = None) -> BR24IMMPlayer:
    """Build a player whose forecast is recomputed after every radar cycle."""
    store = IMMPlaybackStore(config or IMMPlayerConfig())
    return BR24IMMPlayer(store)


def run_br24_imm_player(config: IMMPlayerConfig | None = None) -> BR24IMMPlayer:
    """Open the local IMM player and close its Matplotlib figure afterwards."""
    player = create_br24_imm_player(config)
    try:
        player.show()
    finally:
        plt.close(player.figure)
    return player


def _load_optional_imm_tracker():
    """Load the deliberately untracked local IMM implementation when available."""
    if not IMM_PYTHON.is_file():
        raise FileNotFoundError(
            "Optional IMM implementation not found: external/IMM/br24_imm.py. "
            "Restore the local external IMM files to open this player."
        )
    module = runpy.run_path(str(IMM_PYTHON))
    try:
        return module["BR24IMMTracker"]
    except KeyError as error:
        raise RuntimeError(
            "The local IMM implementation must expose BR24IMMTracker."
        ) from error


def _regularize_covariance(covariance) -> np.ndarray:
    """Return a symmetric positive-definite 2D covariance for an ellipse."""
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
        raise ValueError("IMM forecast covariance must be a finite 2x2 matrix.")
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    return (eigenvectors * np.maximum(eigenvalues, 1e-12)) @ eigenvectors.T
