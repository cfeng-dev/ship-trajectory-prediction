"""Background CSV loading and playback with cancellable Tk timers."""

from queue import Empty, Queue
from threading import Event, Thread


class _LoadCancelled(Exception):
    pass


class EvaluationLoader:
    """Read saved results off the Tk thread; cancellation discards the result."""

    def __init__(self, directory, *, observations_csv=None):
        self._messages = Queue()
        self._cancelled = Event()
        self.thread = Thread(
            target=self._run,
            args=(directory, observations_csv),
            name="evaluation-csv-loader",
            daemon=True,
        )
        self.thread.start()

    def _progress(self, message):
        if self._cancelled.is_set():
            raise _LoadCancelled
        self._messages.put(("progress", message))

    def _run(self, directory, observations_csv):
        try:
            self._progress("Preparing analysis …")
            from bayestraj.validation.evaluation_viewer import load_evaluation_directory

            data = load_evaluation_directory(
                directory, observations_csv=observations_csv, on_progress=self._progress
            )
            self._progress("Preparing diagrams …")
            self._messages.put(("ready", data))
        except _LoadCancelled:
            pass
        except Exception as error:
            if not self._cancelled.is_set():
                self._messages.put(("error", error))

    def take_messages(self):
        messages = []
        while True:
            try:
                messages.append(self._messages.get_nowait())
            except Empty:
                return messages

    def cancel(self):
        self._cancelled.set()


class Playback:
    def __init__(self, scheduler, count, on_frame, interval_ms=200):
        if count < 1:
            raise ValueError("Playback needs at least one origin.")
        self.scheduler = scheduler
        self.count = count
        self.on_frame = on_frame
        self.interval_ms = interval_ms
        self.index = 0
        self.playing = False
        self._timer = None

    def pause(self):
        self.playing = False
        if self._timer is not None:
            self.scheduler.after_cancel(self._timer)
            self._timer = None

    def seek(self, index):
        self.pause()
        self.index = max(0, min(int(index), self.count - 1))
        self.on_frame(self.index)

    def start(self):
        if self.playing:
            return
        if self.index == self.count - 1:
            self.index = 0
        self.playing = self.count > 1
        self.on_frame(self.index)
        if self.playing:
            self._timer = self.scheduler.after(self.interval_ms, self._tick)

    def _tick(self):
        self._timer = None
        if not self.playing:
            return
        self.index += 1
        self.playing = self.index < self.count - 1
        self.on_frame(self.index)
        if self.playing:
            self._timer = self.scheduler.after(self.interval_ms, self._tick)

    def close(self):
        self.pause()
