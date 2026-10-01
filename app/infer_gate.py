"""Serialises model calls and records enough to tell a stalled model from a busy one."""

import threading
import time
from contextlib import contextmanager
from typing import Optional


class InferGate:
    """One lock for every model call, batch and streaming alike.

    The underlying engine is a single synchronous object; calling it from
    several threads at once wedged the vLLM deployment for good. The timings
    kept here are what /healthz reports, so a wedge shows up as a growing
    ``busy_seconds`` instead of a cheerful "healthy".
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._meta = threading.Lock()
        self._waiting = 0
        self._busy_since: Optional[float] = None
        self._last_ok: Optional[float] = None
        self._last_wait = 0.0
        self._max_wait = 0.0
        self._calls = 0
        self._errors = 0

    @contextmanager
    def run(self):
        t_wait = time.monotonic()
        with self._meta:
            self._waiting += 1
        self._lock.acquire()
        waited = time.monotonic() - t_wait
        with self._meta:
            self._waiting -= 1
            self._busy_since = time.monotonic()
            self._last_wait = waited
            self._max_wait = max(self._max_wait, waited)
        ok = False
        try:
            yield
            ok = True
        finally:
            with self._meta:
                self._busy_since = None
                self._calls += 1
                if ok:
                    self._last_ok = time.monotonic()
                else:
                    self._errors += 1
            self._lock.release()

    def snapshot(self) -> dict:
        now = time.monotonic()
        with self._meta:
            return {
                "calls": self._calls,
                "errors": self._errors,
                "waiting": self._waiting,
                "busy_seconds": (
                    round(now - self._busy_since, 2)
                    if self._busy_since is not None
                    else 0.0
                ),
                "last_success_seconds_ago": (
                    round(now - self._last_ok, 2)
                    if self._last_ok is not None
                    else None
                ),
                "last_wait_seconds": round(self._last_wait, 3),
                "max_wait_seconds": round(self._max_wait, 3),
            }
