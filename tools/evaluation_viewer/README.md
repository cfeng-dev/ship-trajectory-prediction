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
```

Alternatively, run `uv run python -m evaluation_viewer` in an installed project.
Choose a results folder, select models and one metric for each right-hand plot,
then select **Apply**. Use the time slider, arrow buttons, Start/Pause or click a
metric curve to inspect a forecast origin. Follow metric time defaults to a
60-second window. Disable it for the full time range.

The viewer compares common origins with identical complete forecast horizons
and reference positions, using the existing comparison reader. Each metric line
represents one saved model run. Point forecasts follow each CSV's declared mean
or median; older files default to medians. Student-t NLPD is offered only when
every loaded model has complete finite saved Student-t scores.

Coordinates use the saved route fields when available for every model, otherwise
the common forecast coordinates. A file with incomplete route geometry is
rejected. Reference points are deduplicated by target time without interpolation;
inconsistent positions at the same time are rejected. Model lines contain only
saved future points, starting at `horizon_step = 1`; saved origin coordinates
are never prepended. A separate neutral marker identifies the reference at the
forecast origin only when that exact reference timestamp is available. The
follow-ship view centers on the nearest saved reference point. Missing
observations or sample trajectories are not reconstructed. The toolbar saves
the displayed figure.

`H` is the number of predicted future positions. `T`, when present in a model
label, is its configured maximum VI/MCMC observation-window size. Metrics are
indexed by forecast origin time, so a forecast issued before a bend may already
have a large error because its future segment crosses that bend.
