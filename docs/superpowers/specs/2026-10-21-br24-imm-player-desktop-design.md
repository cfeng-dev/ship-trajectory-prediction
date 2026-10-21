# BR24 IMM Player Desktop Design

## Purpose and scope

`br24-imm-player` becomes a local desktop application for demonstrating the
online Python CV/CTRV/CA IMM on a BR24 tracking CSV.  Its layout and interaction
model deliberately resemble `trajectory-predictor`: a menu bar, a scrollable
left sidebar, and an embedded Matplotlib view.  It remains a separate tool;
neither `src/bayestraj` nor `trajectory_predictor` imports it.

The tool continues to load the local, ignored `external/IMM/br24_imm.py` at
runtime.  It never packages, copies, or changes that implementation.  Reference
columns are presentation and evaluation labels only and are never supplied to
the tracker.

## User interface

The window has the title `BR24 IMM - Ship Trajectory Player` and follows the
Predictor's visual hierarchy.

- **File** contains `Open BR24 CSV...`, `Reload`, and `Exit`.
- **View** contains `Show settings` and `Reset plot view`.
- **Settings** opens a dialog for the initial observation count, prediction
  horizon, playback interval, and follow-ship span.
- **Help** describes the online update cycle and the evaluation-only reference
  rule.

The left, scrollable sidebar exposes the selected CSV and reload action;
current derived movement state (position, speed, heading, turn rate, time);
forecast metrics (ADE, FDE, RMSE, energy score, log predictive density, 50/90
percent joint coverage); and the current radar association.  Values unavailable
without reference labels are displayed as `-`.

The embedded plot retains the route, current IMM posterior, two-step forecast,
and 50/90-percent covariance ellipses.  A right-hand column shows the
forecasted CV/CTRV/CA mode probabilities and compact forecast uncertainty.
The lower controls provide Start/Pause, a cycle-count slider, Follow ship, and
Show reference.  Advancing one slider unit processes exactly one new radar
cycle and recomputes the forecast.

## Components and data flow

The existing `IMMPlaybackStore` remains the only component that reads raw BR24
cycles, initializes the local tracker, advances it, and caches stages.  It is
extended to accept an explicitly selected CSV and to expose a stage's time and
track positions.  The UI derives speed, heading, and turn rate from successive
IMM world-frame track positions and their actual radar timestamps; no internal
IMM state is accessed.

The player package is divided by responsibility:

- `player.py`: input validation, online stage cache, display-only movement and
  evaluation calculations.
- `gui.py`: Tk application orchestration, reload lifecycle, and menu actions.
- `controls.py`: sidebar fields and file chooser.
- `plot_view.py`: embedded Matplotlib route, right-hand charts, playback, and
  view controls.
- `dialogs.py` and `view.py`: settings/help dialogs and shared Tk styling.

Selecting a file or applying settings constructs a new playback store.  The
previous cache and timer are discarded, so results from two files or settings
sets cannot mix.  Invalid CSV schemas, invalid settings, unavailable reference
metrics, and missing local IMM source are reported in a Tk error dialog.

## Evaluation definitions

For a stage with a reference, metrics use only the just-produced forecast and
the next matching reference positions.  ADE, FDE, and RMSE use Euclidean
errors.  Energy score, Gaussian log predictive density, and 50/90-percent joint
coverage use the same Gaussian IMM definitions as
`experiments/model_evaluation/imm.py`.  No metric changes the tracker state.

## Validation

No repository functional test for the unpublished local IMM is added.  The
implementation is validated with a local headless smoke run that advances one
real radar cycle, existing Predictor GUI checks where applicable, focused Ruff
and format checks, and the existing full test suite.  The external IMM source
must remain ignored by Git.
