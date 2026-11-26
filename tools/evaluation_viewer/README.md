# Evaluation Viewer

Explore saved forecast CSVs with the same desktop palette, controls and plot
toolbar as Bayesian Predictor. No model fitting is performed.

## Design and implementation plan

The approved design uses a separate CSV viewer rather than extending the
inference tools or the static comparison script. Shared CSV validation and
per-origin metric aggregation stay in `src/bayestraj`; desktop interaction stays
in `tools/evaluation_viewer`. No additional dependencies are needed.

1. Validate and align saved forecasts; cache per-origin metrics, forecast paths
   and the reference trajectory. Test physical-time alignment, coordinate
   consistency and optional Student-t scores using generated CSV fixtures.
2. Draw a reference/forecast view alongside two independently selected metrics.
   Test synchronization, model colors, time windows and single-origin inputs
   using the Agg backend.
3. Add the Bayesian Predictor styled desktop shell, model selection, language
   controls, playback and the `evaluation-viewer` entry point. Test navigation,
   timer cancellation and CLI help without a display; run comparison regressions.

## Run

```shell
uv run evaluation-viewer
uv run evaluation-viewer --input-dir results/evaluation
uv run evaluation-viewer --observations-csv data/processed/BR24_imm_tracked_trajectory.csv
```

Alternatively, run `uv run python -m evaluation_viewer` in an installed project.
The window opens without loading CSVs or creating plots. Choose a results folder
and an optional observation CSV, then select **Start analysis** in the top
**Analysis** group to read the data and build the diagrams. Browsing for a folder
or file only changes its input path. CSV reading and metric preparation run in
the background, with progress shown in the center. The button
turns pale red and becomes **Cancel analysis**, matching Bayesian Predictor.
Cancellation stops preparation at the next stage and discards the pending
result; any previously loaded evaluation remains available. An in-progress CSV
read finishes before the worker stops. A new load can be started immediately.
Model and metric selections update the plots immediately. At least one model
must remain selected. Changing a metric keeps the current forecast origin and
continues any active playback.
Toggling **Follow ship**, opening plot settings and applying display changes also
keep active playback running. Plot settings stay non-modal: the main window
remains usable and closing it also closes the settings window.
Use the time slider, arrow keys, Start/Pause or click a
metric curve to inspect a forecast origin. Metric curves use lines without point
markers and show the full time range by default. Open **View → Plot display…**
to enable **Follow metric time**, set its window width (initially 60 seconds),
or control **Show trajectory legend** and **Show metric legend** independently
under **General**. As in Bayesian Predictor, the
**Trajectory** group controls the reference trajectory and its point markers,
observations through N, the current-position marker and the follow-ship view
span. These controls affect only the display. The **Follow ship** toggle remains
in the bottom playback bar.
**Show reference forecast segment** controls the reference within the selected
forecast horizon independently of the full reference trajectory; it defaults off.
The default **One metric above trajectory** view shows a single full-width metric
diagram above the trajectory, with one metric selector. Disable
the option to restore the trajectory on the left and both metric diagrams on
the right; the second metric selection is retained. In this side-by-side layout,
the trajectory frame stays square even with rectangular zoom selections.
The stacked plots have matching widths while keeping equal spatial scaling.
With **Follow ship**, the configured span sets the initial vertical extent;
manual zoom changes its scale. The horizontal extent adapts to the available
width when resizing. Toolbar zoom and pan survive redraws. Following recenters
the chosen zoom on the current position. Disabling **Follow ship** fits the full
trajectory once; subsequent frames keep that overview or any manual zoom.
Re-enabling following focuses the current position at the configured span.
A new dataset, layout or configured span also resets the viewport.
**Apply** commits the display changes; **Cancel** leaves them unchanged.

**Observation CSV** selects the original input `time,x,y` trajectory. The command
line defaults to `data/processed/BR24_imm_tracked_trajectory.csv` when it exists.
Choose a different source with the browse button or `--observations-csv`. Its
numeric timestamps must cover the forecast origins in the same time frame;
timestamps are not normalized or interpolated. Ground-truth columns are ignored.
Use the original input for the same experiment; overlapping timestamps alone do
not verify that two files describe the same trajectory.
Select **Start analysis** after editing the source path. Clear the observation path,
or pass `--observations-csv=`, to inspect evaluation CSVs alone.

As in Bayesian Predictor, original observations through the selected origin are
drawn in dark blue and the current observation is marked in orange. Future
observations are hidden. If an origin has no exact observation, the marker is
labelled **Latest observation**, without inventing a current measurement.
Without an observation CSV, a selected IMM run's saved origin history is shown
instead, explicitly labelled as an IMM track rather than observations.

The bottom bar uses the same Matplotlib button, blue slider and checkbox style
as Bayesian Predictor. **Follow ship** and **Show settings** apply immediately.
The slider is labelled **N** and shows **current/total**, such as **15/601**.
With observations, both counts refer to observations through the selected and
last selectable forecast origins. Without an observation CSV, they refer to
positions in the common forecast sequence. Before loading data, it shows **0/0**.

The viewer compares common origins with identical complete forecast horizons
and reference positions, using the existing comparison reader. Each metric line
represents one saved model run. Point forecasts follow each CSV's declared mean
or median; older files default to medians. Student-t NLPD is offered only when
every loaded model has complete finite saved Student-t scores.

Coordinates use the saved route fields when available for every model, otherwise
the common forecast coordinates. A file with incomplete route geometry is
rejected. Reference points are deduplicated by target time without interpolation;
inconsistent positions at the same time are rejected. All model lines, including
IMM, contain only saved future points, starting at `horizon_step = 1`. No model
origin is prepended to a forecast line. Forecasts start at the first saved future
point, without a connecting line from the current-position marker. Without
observations, saved IMM origins remain visible as **IMM track at forecast origin**
markers. Bayesian origin columns can contain reference coordinates, so they are
not used as model start points. Display choices do not change saved forecasts
or metrics.
The reference remains a benchmark trajectory, without a separate origin marker.
If no selected IMM run supplies
origin coordinates, no tracked origin is invented. The follow-ship view centers
on the latest input observation, otherwise on the selected IMM origin (the mean
for multiple IMM runs), falling back to the nearest saved reference point.
Missing observations or sample trajectories are not reconstructed. The toolbar
saves the displayed figure.

`H` is the number of predicted future positions. `T`, when present in a model
label, is its configured maximum VI/MCMC observation-window size. Metrics are
indexed by forecast origin time, so a forecast issued before a bend may already
have a large error because its future segment crosses that bend.
