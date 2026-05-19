import os
import sys


class _Tee:
    """Writes stdout to both the console and a log file simultaneously."""

    def __init__(self, log_path):
        self._stdout = sys.stdout
        self._file = open(log_path, 'w')

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        sys.stdout = self._stdout
        self._file.close()


class Logger:
    """Optional experiment logger that tees stdout to a log file.

    Args:
        active: Whether logging is enabled. When False all methods are no-ops.
        logs_dir: Root directory under which experiment subdirectories are created.
    """

    def __init__(self, active: bool = True, logs_dir: str = "logs"):
        self.active = active
        self.logs_dir = logs_dir
        self.log_dir = None
        self.fig_dir = None
        self._tee = None

    def init(self, experiment_name: str):
        """Creates the experiment directory and opens log.txt for writing.

        Args:
            experiment_name: Name of the experiment; used as a subdirectory name.
        """
        if not self.active:
            return

        os.makedirs(self.logs_dir, exist_ok=True)

        self.log_dir = os.path.join(self.logs_dir, experiment_name)
        self.fig_dir = os.path.join(self.log_dir, "figures")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.fig_dir, exist_ok=True)

        log_path = os.path.join(self.log_dir, "log.txt")
        self._tee = _Tee(log_path)
        sys.stdout = self._tee

        print(f"Logger initialized. Writing to: {log_path}")

    def close(self):
        """Restores stdout and closes the log file."""
        if self._tee is not None:
            self._tee.close()
            self._tee = None
