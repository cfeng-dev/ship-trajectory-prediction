"""GUI-independent formatting for simulated elapsed time."""


def format_elapsed_time(seconds):
    """Format short simulated durations precisely and longer ones readably."""
    if seconds < 60:
        return f"{seconds:.1f} s"
    whole_seconds = round(seconds)
    minutes, remaining_seconds = divmod(whole_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} h {minutes:02d} min {remaining_seconds:02d} s"
    return f"{minutes} min {remaining_seconds:02d} s"
