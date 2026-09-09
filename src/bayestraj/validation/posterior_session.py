"""Serial background inference for an interactive posterior-analysis session.

The worker owns its loader and filter; consumers handle events on their UI thread.
Closing or replacing a session lets the current fit finish, then discards its result.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Condition, Thread

import bayestraj.observations.io as observations_io
import bayestraj.validation.bayesian_ctrv_posterior_dashboard as dashboard
from bayestraj.inference.ctrv_rbpf import SequentialCTRVFilterConfig
from bayestraj.inference.ctrv_smc import SequentialMonteCarloCTRVConfig
from bayestraj.models.bayesian_ctrv import BayesianCTRVPriors


@dataclass(frozen=True)
class AnalysisSettings:
    """One configuration snapshot, shared by data loading and inference."""

    data_file: Path
    experiment: dashboard.PosteriorDashboardConfig
    priors: BayesianCTRVPriors
    vi_config: dict
    mcmc_config: dict
    rbpf_config: SequentialCTRVFilterConfig
    smc_config: SequentialMonteCarloCTRVConfig


@dataclass(frozen=True)
class AnalysisEvent:
    """A result or progress notification belonging to one analysis generation."""

    generation: int
    kind: str
    payload: object = None


def prepare_analysis(settings: AnalysisSettings):
    """Load the selected route and construct its persistent posterior loader."""
    data = (
        observations_io.read_ship_data(
            settings.data_file, run_id=settings.experiment.run_id
        )
        .sort_values("time")
        .reset_index(drop=True)
    )
    if data.empty:
        raise ValueError(
            f"Run ID {settings.experiment.run_id} was not found in the selected CSV file."
        )
    return dashboard.create_posterior_dashboard_loader(
        data,
        experiment=settings.experiment,
        priors=settings.priors,
        vi_config=settings.vi_config,
        mcmc_config=settings.mcmc_config,
        rbpf_config=settings.rbpf_config,
        smc_config=settings.smc_config,
    )


class PosteriorAnalysisWorker:
    """Compute requested prefixes on one thread; never access GUI objects.

    Requests replace the target instead of queuing repeated fits. Results for an old
    configuration are ignored. An inference error stops that generation because a
    particle filter may already have advanced when extraction fails.
    """

    def __init__(self, prepare=prepare_analysis):
        self._prepare = prepare
        self._condition = Condition()
        self._events = Queue()
        self._generation = 0
        self._pending = None
        self._target = 0
        self._closed = False
        self._thread = Thread(target=self._run, name="posterior-inference")
        self._thread.start()

    def start(self, settings: AnalysisSettings) -> int:
        """Replace the analysis with an independent configuration snapshot."""
        snapshot = deepcopy(settings)
        with self._condition:
            if self._closed:
                raise RuntimeError("The analysis worker is closed.")
            self._generation += 1
            self._pending = (self._generation, snapshot)
            self._target = 0
            self._condition.notify()
            return self._generation

    def request(self, observation_count: int) -> None:
        """Set the latest desired stage; lowering it pauses future computation."""
        with self._condition:
            if not self._closed:
                self._target = observation_count
                self._condition.notify()

    def drain(self) -> list[AnalysisEvent]:
        """Return available notifications only for the current configuration."""
        events = []
        while True:
            try:
                event = self._events.get_nowait()
            except Empty:
                return events
            if not self._closed and event.generation == self._generation:
                events.append(event)

    def close(self) -> None:
        """Stop after the current fit without blocking the calling UI thread."""
        with self._condition:
            self._closed = True
            self._pending = None
            self._condition.notify()

    def join(self, timeout=None) -> None:
        """Wait for a fit to finish and the closed worker to exit."""
        self._thread.join(timeout)

    @property
    def is_alive(self) -> bool:
        """Return whether the worker is still running or finishing a fit."""
        return self._thread.is_alive()

    def _run(self):
        generation, loader, next_count, maximum = 0, None, 1, 0
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda loader=loader, next_count=next_count, maximum=maximum: (
                        self._closed
                        or self._pending is not None
                        or (
                            loader is not None
                            and next_count <= min(self._target, maximum)
                        )
                    )
                )
                if self._closed:
                    return
                pending = self._pending
                self._pending = None
            try:
                if pending is not None:
                    generation, settings = pending
                    loader = None
                    self._emit(generation, "loading")
                    trajectory, maximum, next_count, loader = self._prepare(settings)
                    self._emit(generation, "ready", (trajectory, maximum, next_count))
                    continue
                self._emit(generation, "computing", next_count)
                update = loader(next_count)
                if update.observation_count != next_count:
                    raise ValueError("The loader returned the wrong observation count.")
                self._emit(generation, "update", update)
                next_count += 1
            except Exception as error:
                loader = None
                self._emit(generation, "error", f"{type(error).__name__}: {error}")

    def _emit(self, generation, kind, payload=None):
        with self._condition:
            if not self._closed and generation == self._generation:
                self._events.put(AnalysisEvent(generation, kind, payload))
