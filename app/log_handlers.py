"""Logging handlers for the local API log (uvicorn_log.json). Standard library only."""

import os
import time
from logging.handlers import RotatingFileHandler


class SharedRotatingFileHandler(RotatingFileHandler):
    """A RotatingFileHandler that survives a second process writing the same file.

    uvicorn configures logging in the parent AND again in every --reload / --workers
    child, and two servers started from the repo root (the demo API on :8000 and a test
    API on :8010) share logs/api.log. On Windows a file another process holds open
    cannot be renamed, and the stock rollover then failed on EVERY record once the file
    reached maxBytes: a '--- Logging error ---' traceback each, the record lost from the
    file, and the backups shifted again on each try until they were gone (measured Oct
    15: 169 of 200 records lost, api.log.1 missing).

    Here the live file is renamed FIRST, so a refusal touches no backup; the process
    keeps appending past maxBytes and tries again only after `retry_s` seconds. Once
    the other writer exits, the next try rotates as usual. With one process (or on
    POSIX, where an open file can be renamed) it behaves as RotatingFileHandler."""

    def __init__(self, filename, mode="a", maxBytes=0, backupCount=0, encoding=None,
                 delay=False, errors=None, retry_s: float = 60.0):
        super().__init__(filename, mode=mode, maxBytes=maxBytes, backupCount=backupCount,
                         encoding=encoding, delay=delay, errors=errors)
        self.retry_s = float(retry_s)
        self._retry_at = 0.0

    def shouldRollover(self, record):
        if time.monotonic() < self._retry_at:
            return False
        return super().shouldRollover(record)

    def doRollover(self):
        if self.backupCount <= 0:
            super().doRollover()
            return
        if self.stream:
            self.stream.close()
            self.stream = None
        staged = self.baseFilename + ".rotating"
        try:
            os.replace(self.baseFilename, staged)
        except FileNotFoundError:
            pass                                    # nothing to rotate yet
        except OSError:
            # Another process has the file open (Windows): keep appending, try later
            self._retry_at = time.monotonic() + self.retry_s
        else:
            for i in range(self.backupCount - 1, 0, -1):
                source = self.rotation_filename(f"{self.baseFilename}.{i}")
                if os.path.exists(source):
                    os.replace(source, self.rotation_filename(f"{self.baseFilename}.{i + 1}"))
            os.replace(staged, self.rotation_filename(f"{self.baseFilename}.1"))
        if not self.delay:
            self.stream = self._open()
