import threading
import time

import pytest

from infer_gate import InferGate


def test_calls_never_overlap():
    gate = InferGate()
    inside = 0
    overlaps = []

    def work():
        nonlocal inside
        with gate.run():
            inside += 1
            overlaps.append(inside)
            time.sleep(0.01)
            inside -= 1

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert max(overlaps) == 1
    snap = gate.snapshot()
    assert snap["calls"] == 8
    assert snap["waiting"] == 0
    assert snap["busy_seconds"] == 0.0


def test_snapshot_shows_a_call_in_progress():
    gate = InferGate()
    entered, release = threading.Event(), threading.Event()

    def stuck():
        with gate.run():
            entered.set()
            release.wait(5)

    t = threading.Thread(target=stuck)
    t.start()
    entered.wait(5)
    time.sleep(0.05)
    assert gate.snapshot()["busy_seconds"] > 0
    release.set()
    t.join()
    assert gate.snapshot()["busy_seconds"] == 0.0


def test_failed_call_releases_the_lock():
    gate = InferGate()
    with pytest.raises(RuntimeError):
        with gate.run():
            raise RuntimeError("boom")
    with gate.run():
        pass
    snap = gate.snapshot()
    assert snap["errors"] == 1
    assert snap["calls"] == 2
