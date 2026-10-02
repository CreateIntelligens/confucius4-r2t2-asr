"""Serialises model calls and records enough to tell a stalled model from a busy one."""

import threading
import time
from contextlib import contextmanager
from typing import Optional

LIVE = 0  # a step of a real-time stream: someone is waiting for the caption
BATCH = 1  # a segment of an uploaded file


class InferGate:
    """One queue for every model call, batch and streaming alike.

    The underlying engine is a single synchronous object; calling it from
    several threads at once wedged the vLLM deployment for good. The timings
    kept here are what /healthz reports, so a wedge shows up as a growing
    ``busy_seconds`` instead of a cheerful "healthy".

    即時串流排在上傳音檔前面：上傳的每一段要算一兩秒，這段時間所有正在講話的人字幕
    都會頓住。只看「現在有沒有串流在排隊」不夠 —— 串流每 160ms 才來一步，兩步之間
    佇列是空的，上傳會一段接一段插進來。所以改看時間：最近有串流在跑（有人在講話）
    時，上傳跑完一段要先讓出它所花時間的 BATCH_COOLDOWN_RATIO 倍（上限 batch_interval
    秒）才能再跑下一段，也就是上傳最多佔兩成的推論時間。依花費計算而不是固定間隔，
    幾秒的短音檔（互動式的批次辨識）只等零點幾秒，30 秒的長片段才會等到數秒。
    沒人講話時上傳不受限制。按次數輪流（每 N 步讓一段）在這裡行不通，三路同時講話時
    字幕仍會晚好幾秒。
    """

    # 多久沒有串流步驟就視為沒人在講話
    LIVE_IDLE_SECONDS = 1.0
    BATCH_COOLDOWN_RATIO = 4.0

    def __init__(self, batch_interval: float = 5.0) -> None:
        self._cond = threading.Condition()
        self._busy = False
        self._waiting = [0, 0]
        self._batch_interval = batch_interval
        self._last_live_end = float("-inf")
        self._last_batch_end = float("-inf")
        self._batch_cooldown = 0.0
        self._busy_since: Optional[float] = None
        self._last_ok: Optional[float] = None
        self._last_wait = 0.0
        self._max_wait = 0.0
        self._calls = 0
        self._errors = 0

    def _batch_may_run(self, now: float) -> bool:
        nobody_speaking = (
            self._waiting[LIVE] == 0 and now - self._last_live_end > self.LIVE_IDLE_SECONDS
        )
        return nobody_speaking or now - self._last_batch_end >= self._batch_cooldown

    def _may_run(self, priority: int) -> bool:
        if self._busy:
            return False
        now = time.monotonic()
        if priority == BATCH:
            return self._batch_may_run(now)
        # a stream step waits only for an upload whose turn has come
        overdue = now - self._last_batch_end >= self._batch_cooldown
        return not (self._waiting[BATCH] > 0 and overdue)

    @contextmanager
    def run(self, priority: int = LIVE):
        t_wait = time.monotonic()
        with self._cond:
            self._waiting[priority] += 1
            while not self._may_run(priority):
                # 上傳能不能跑也取決於時間流逝，不能只等別人通知
                self._cond.wait(0.05 if priority == BATCH else None)
            self._waiting[priority] -= 1
            self._busy = True
            started = time.monotonic()
            waited = started - t_wait
            self._busy_since = started
            self._last_wait = waited
            self._max_wait = max(self._max_wait, waited)
        ok = False
        try:
            yield
            ok = True
        finally:
            with self._cond:
                self._busy = False
                self._busy_since = None
                now = time.monotonic()
                if priority == LIVE:
                    self._last_live_end = now
                else:
                    self._last_batch_end = now
                    took = now - started
                    self._batch_cooldown = min(self._batch_interval, self.BATCH_COOLDOWN_RATIO * took)
                self._calls += 1
                if ok:
                    self._last_ok = time.monotonic()
                else:
                    self._errors += 1
                self._cond.notify_all()

    def snapshot(self) -> dict:
        now = time.monotonic()
        with self._cond:
            return {
                "calls": self._calls,
                "errors": self._errors,
                "waiting": sum(self._waiting),
                "waiting_live": self._waiting[LIVE],
                "waiting_batch": self._waiting[BATCH],
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
