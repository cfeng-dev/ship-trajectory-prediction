# BR24 IMM Player Desktop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the standalone BR24 IMM Matplotlib window with a local desktop
player that mirrors the Trajectory Predictor's menu, sidebar, and embedded plot
workflow.

**Architecture:** Keep `IMMPlaybackStore` as the sole owner of raw BR24 input,
local tracker lifetime, and cached online stages.  Add a Tk presentation layer
inside `tools/br24_imm_player` with independent controls, dialogs, and an
embedded Matplotlib view; it consumes immutable display values from the store
and never imports `trajectory_predictor`.

**Tech Stack:** Python 3.10+, Tkinter, Matplotlib TkAgg, NumPy, SciPy, Pandas

**Spec:** `docs/superpowers/specs/2026-10-21-br24-imm-player-desktop-design.md`

## Global Constraints

- Keep `external/IMM/` ignored, unmodified, and dynamically loaded only at
  runtime.
- Feed the tracker raw radar cycles only; reference labels are evaluation and
  display data only.
- Derive displayed speed, heading, and turn rate from IMM world-frame track
  positions and real BR24 timestamps, not private IMM state.
- Do not add an IMM functional-test module to `tests/`, per the user's explicit
  publication constraint.
- Add no dependencies and do not modify `src/bayestraj` or
  `tools/trajectory_predictor`.
- Do not run Git staging, commit, or push commands.

## Review Focus

- A valid BR24 CSV without reference columns must still play, with reference
  metrics rendered as `-`; verify by opening a temporary reference-free copy in
  the local smoke run.
- Selecting a malformed/non-BR24 CSV must preserve the current player and show
  a clear Tk error dialog; verify manually with a small invalid CSV.
- A changed prediction horizon must reset the cache and leave exactly that many
  future steps; verify through the Settings dialog and sidebar state.
- One slider increment must process exactly one new radar cycle and refresh the
  forecast; verify with the local headless player smoke run.
- A missing ignored IMM module must report the existing actionable error rather
  than silently substituting a tracker; verify by calling the loader with a
  temporary nonexistent path during smoke validation.

### Task 1: Extend the presentation-safe IMM session model

**Files:**

- Modify: `tools/br24_imm_player/player.py`

**Interfaces:**

- Consumes: `IMMPlayerConfig`, `br24.read_br24_tracking_input`, local
  `BR24IMMTracker` public properties and forecast result.
- Produces: `IMMPlaybackStore(data_file: Path, config: IMMPlayerConfig)`,
  `IMMPlayerStage`, `IMMDisplayState`, and `IMMForecastMetrics` for the Tk
  layer.

- [ ] Add `data_file: Path` to `IMMPlayerConfig`, validate all UI settings, and
  make `IMMPlaybackStore` load that selected path rather than a module global.
- [ ] Add immutable display values for cycle metadata, derived movement state,
  forecast metrics, and forecast uncertainty.  Keep raw reference values out of
  the tracker methods.
- [ ] Calculate speed, mathematical heading in degrees, and signed turn rate
  from the last available finite track differences and actual radar time
  intervals.  Show `None` before a quantity becomes identifiable.
- [ ] Calculate ADE, FDE, RMSE, energy score, Gaussian log density, and
  50/90-percent joint coverage only when synchronized reference labels exist.
- [ ] Retain a deterministic forecast cache for every processed stage: a jump to
  a later N processes only missing cycles and caches each intermediate refreshed
  forecast, so dragging the slider back to an earlier N never rewinds or mutates
  the tracker.
- [ ] Run a local headless smoke command that creates N=10 and N=11 display
  states from the real ignored IMM and confirms a two-step forecast at each.

### Task 2: Add Predictor-style Tk shell, menus, and sidebar

**Files:**

- Create: `tools/br24_imm_player/view.py`
- Create: `tools/br24_imm_player/controls.py`
- Create: `tools/br24_imm_player/dialogs.py`
- Create: `tools/br24_imm_player/gui.py`
- Modify: `tools/br24_imm_player/cli.py`

**Interfaces:**

- Consumes: `IMMPlayerConfig`, `IMMPlaybackStore`, and a new `IMMPlotView`
  created in Task 3.
- Produces: `BR24IMMPlayerApp(root: tk.Tk)` and a sidebar that can expose
  `show_display_state(state)` and the selected CSV path.

- [ ] Implement shared window colours, fonts, menu construction, and Tk window
  lifecycle in `view.py`; title the window `BR24 IMM - Ship Trajectory Player`.
- [ ] Implement a scrollable sidebar in `controls.py` with sections for Data,
  Current movement, Evaluation metrics, and BR24 radar association.  Use `-`
  for unavailable fields.
- [ ] Implement the file chooser and `Open BR24 CSV...`/`Reload` paths.  A load
  failure keeps the last successful view and reports an error using
  `messagebox.showerror`.
- [ ] Implement a modal Settings dialog in `dialogs.py` for initial observation
  count, prediction horizon, playback interval, and follow-ship span.  Parse
  all fields before applying any setting.
- [ ] Implement `BR24IMMPlayerApp` to construct a fresh store after a successful
  reload/settings apply, create the embedded view, and close its timer before
  replacement or window shutdown.
- [ ] Change the CLI to create a Tk root and launch `BR24IMMPlayerApp`; retain
  command-line help without requiring a display.
- [ ] Manually start the tool with the standard BR24 CSV, inspect the menu and
  sidebar, and verify that selecting the Settings menu opens the dialog.

### Task 3: Implement the embedded trajectory and IMM visualisation

**Files:**

- Create: `tools/br24_imm_player/plot_view.py`
- Modify: `tools/br24_imm_player/gui.py`

**Interfaces:**

- Consumes: `IMMDisplayState`, `IMMPlayerConfig`, and callbacks supplied by
  `BR24IMMPlayerApp` for stage loading and sidebar updates.
- Produces: `IMMPlotView(parent, ...)` with `set_stage`, `reset_view`,
  `set_sidebar_visible`, `destroy`, and timer-managed Start/Pause playback.

- [ ] Embed a Matplotlib `FigureCanvasTkAgg` in the right panel and render the
  IMM route, current posterior, refreshed mean forecast, and 50/90-percent
  covariance ellipses.
- [ ] Render a right-hand pair of axes for forecast CV/CTRV/CA probabilities and
  forecast uncertainty.  Do not label either panel as a Bayesian posterior.
- [ ] Add Start/Pause, integer N slider, Follow ship, Show reference, and
  Show settings controls under the route.  N+1 calls the app stage loader for
  exactly one newer radar cycle.
- [ ] Add `reset_view` and follow-ship bounds based on the configured span;
  include the optional reference trace with the exact label
  `Reference (evaluation only)`.
- [ ] Send every stage update to the sidebar so its movement, evaluation, and
  association values match the displayed forecast origin.
- [ ] Run a Tk-enabled manual playback from N=10 through N=12; verify the
  displayed N, detection ID, movement state, and forecast metrics change
  together.

### Task 4: Finish error paths, documentation, and validation

**Files:**

- Modify: `tools/br24_imm_player/player.py`
- Modify: `tools/br24_imm_player/gui.py`
- Modify: `tools/br24_imm_player/dialogs.py`
- Modify: `tools/br24_imm_player/plot_view.py`
- Modify: `docs/superpowers/specs/2026-10-21-br24-imm-player-desktop-design.md`

**Interfaces:**

- Consumes: the completed store and UI interfaces from Tasks 1-3.
- Produces: a locally runnable `br24-imm-player` that preserves all reference
  and ignored-source constraints.

- [ ] Confirm invalid setting values and CSV schema errors use an actionable
  dialog and retain the previous successful session.
- [ ] Confirm reference-free input hides reference overlays and leaves metric
  fields as `-`, while playback and movement fields still work.
- [ ] Confirm the optional IMM loader's missing-file message names
  `external/IMM/br24_imm.py` and does not use a fallback model.
- [ ] Update the design specification only if implementation reveals a material
  user-visible difference; otherwise preserve it unchanged.
- [ ] Run focused Ruff checks and formatting checks for all touched player files,
  `git diff --check`, a non-interactive local plot smoke run, and the full
  existing pytest suite with `MPLBACKEND=Agg`.
- [ ] Verify `git check-ignore -v external/IMM/br24_imm.py` still reports the
  existing ignore rule; do not stage or commit anything.
