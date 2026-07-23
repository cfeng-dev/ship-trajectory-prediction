"""Console reporting helpers for trajectory prediction experiments."""

MIN_SEPARATOR_WIDTH = 60


def print_prediction_setup(
    title,
    *,
    data_file,
    run_id,
    window,
    extra_rows=(),
):
    """Print shared window information and model-specific setup values."""
    rows = [
        ("Data file", data_file),
        ("Run ID", run_id),
        ("Observed positions", window.observation_count),
        ("Predicted positions", window.prediction_count),
        *extra_rows,
    ]
    separator = "=" * max(MIN_SEPARATOR_WIDTH, len(title))
    label_width = max(len(label) for label, _ in rows)

    print(separator)
    print(title)
    print(separator)
    for label, value in rows:
        print(f"{label:<{label_width}}: {value}")
