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
Choose a results folder, select models and one metric for each right-hand plot,
then select **Apply**. Use the time slider, arrow keys, Start/Pause or click a
metric curve to inspect a forecast origin. Follow metric time defaults to a
60-second window. Disable it for the full time range.

**Observation CSV** selects the original input `time,x,y` trajectory. The command
line defaults to `data/processed/BR24_imm_tracked_trajectory.csv` when it exists.
Choose a different source with the browse button or `--observations-csv`. Its
numeric timestamps must cover the forecast origins in the same time frame;
timestamps are not normalized or interpolated. Ground-truth columns are ignored.
Use the original input for the same experiment; overlapping timestamps alone do
not verify that two files describe the same trajectory.
Select **Load CSVs** after editing the source path. Clear the observation path,
or pass `--observations-csv=`, to inspect evaluation CSVs alone.

As in Bayesian Predictor, original observations through the selected origin are
drawn in dark blue and the current observation is marked in orange. Future
observations are hidden. If an origin has no exact observation, the marker is
labelled **Latest observation**, without inventing a current measurement.
Without an observation CSV, a selected IMM run's saved origin history is shown
instead, explicitly labelled as an IMM track rather than observations.

The bottom bar uses the same Matplotlib button, blue slider and checkbox style
as Bayesian Predictor. **Follow ship** and **Show settings** apply immediately;
the slider shows the selected forecast index and its elapsed time.

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
origin is prepended to a forecast line. With observations, dashed display links
connect the same current observation to the first future point of every model.
Without observations, the selected IMM origin supplies the common display
anchor, marked as **IMM track at forecast origin**, and the same dashed links
connect it to every model's first future point. Bayesian origin columns can
contain reference coordinates, so they are not used as model start points.
These display links do not replace inferred model origin positions or change
saved forecasts or metrics. Observation links require an exact observation at
the origin; IMM links
require one run or identical origins across selected IMM runs.
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
