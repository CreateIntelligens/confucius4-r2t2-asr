import threading
import time

import pytest

from infer_gate import BATCH, LIVE, InferGate


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


def run_order(gate, arrivals):
    """Hold the gate, queue ``arrivals`` behind it, release, and return the order they ran in."""
    order, threads = [], []
    release = threading.Event()

    def holder():
        with gate.run():
            release.wait(5)

    def worker(name, priority):
        with gate.run(priority):
            order.append(name)

    first = threading.Thread(target=holder)
    first.start()
    while gate.snapshot()["busy_seconds"] == 0:
        time.sleep(0.005)
    for name, priority in arrivals:
        t = threading.Thread(target=worker, args=(name, priority))
        t.start()
        threads.append(t)
        while gate.snapshot()["waiting"] < len(threads):
            time.sleep(0.005)
    release.set()
    for t in [first] + threads:
        t.join(5)
    return order


def test_live_steps_run_before_waiting_uploads():
    gate = InferGate(batch_interval=60)
    with gate.run(BATCH):  # an upload just ran, so the next one has to wait its turn
        time.sleep(0.5)
    order = run_order(gate, [("upload", BATCH), ("live1", LIVE), ("live2", LIVE)])
    assert order == ["live1", "live2", "upload"] or order == ["live2", "live1", "upload"]


def test_upload_cannot_slip_in_between_stream_steps():
    """Between two steps of a stream nothing is queued; an upload must still wait."""
    gate = InferGate(batch_interval=60)
    with gate.run(BATCH):
        time.sleep(0.5)
    ran = threading.Event()

    def upload():
        with gate.run(BATCH):
            ran.set()

    with gate.run(LIVE):
        pass
    t = threading.Thread(target=upload)
    t.start()
    for _ in range(5):  # someone keeps speaking: a step every 50 ms
        time.sleep(0.05)
        assert not ran.is_set()
        with gate.run(LIVE):
            pass
    # speech stops; after LIVE_IDLE_SECONDS the upload goes ahead
    assert ran.wait(gate.LIVE_IDLE_SECONDS + 1)
    t.join(5)


def test_uploads_are_not_starved_by_continuous_speech():
    gate = InferGate(batch_interval=0.3)
    with gate.run(BATCH):
        time.sleep(0.5)
    ran = threading.Event()

    def upload():
        with gate.run(BATCH):
            ran.set()

    t = threading.Thread(target=upload)
    t.start()
    deadline = time.monotonic() + 3
    while not ran.is_set() and time.monotonic() < deadline:
        with gate.run(LIVE):  # speech never pauses
            time.sleep(0.01)
    t.join(5)
    assert ran.is_set()


def test_uploads_run_back_to_back_when_nobody_is_speaking():
    gate = InferGate(batch_interval=60)
    t0 = time.monotonic()
    for _ in range(3):
        with gate.run(BATCH):
            pass
    assert time.monotonic() - t0 < 1


def test_snapshot_reports_who_is_waiting():
    gate = InferGate()
    snap = gate.snapshot()
    assert snap["waiting"] == snap["waiting_live"] == snap["waiting_batch"] == 0


def test_a_cheap_upload_only_costs_a_short_wait():
    """Cooldown follows what the upload cost: a short clip must not wait as long as a 30 s segment."""
    gate = InferGate(batch_interval=60)
    with gate.run(BATCH):
        time.sleep(0.05)
    ran = threading.Event()

    def upload():
        with gate.run(BATCH):
            ran.set()

    t = threading.Thread(target=upload)
    t.start()
    deadline = time.monotonic() + 2
    while not ran.is_set() and time.monotonic() < deadline:
        with gate.run(LIVE):  # speech never pauses
            time.sleep(0.01)
    t.join(5)
    assert ran.is_set()
