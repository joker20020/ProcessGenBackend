from PySide6.QtCore import QObject, QRunnable, Signal


class _Signals(QObject):
    finished = Signal(object)
    failed = Signal(str)


class ApiWorker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = _Signals()

    @property
    def finished(self):
        return self.signals.finished

    @property
    def failed(self):
        return self.signals.failed

    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
            try:
                self.signals.finished.emit(result)
            except RuntimeError:
                pass
        except Exception as e:
            try:
                self.signals.failed.emit(str(e))
            except RuntimeError:
                pass
