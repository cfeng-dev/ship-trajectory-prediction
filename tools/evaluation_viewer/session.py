"""Playback of cached origins with one cancellable Tk-compatible timer."""


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
