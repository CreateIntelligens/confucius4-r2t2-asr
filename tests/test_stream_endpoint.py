import json
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import main


class BrokenModel:
    """Loads fine, fails on the first decode step."""

    def get_supported_languages(self):
        return ["Chinese", "English"]

    def init_streaming_state(self, **kwargs):
        return SimpleNamespace(buffer=np.zeros((0,), dtype=np.float32), chunk_text=[], text="")

    def streaming_transcribe_no_reset(self, *args, **kwargs):
        raise ValueError("decoder exploded")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setitem(main.STATE, "model", BrokenModel())
    monkeypatch.setitem(main.STATE, "vad", None)
    # no `with`: the lifespan would start loading the real model
    return TestClient(main.app)


def header(**extra):
    return json.dumps({"requestId": "r1", "secret_key": "test0102", **extra})


def test_inference_failure_is_reported_before_closing(client):
    with client.websocket_connect("/asr_stream_api_v1") as ws:
        ws.send_text(header(use_vad=False))
        assert json.loads(ws.receive_text())["status"] == "connected"
        ws.send_bytes(np.zeros(5120, dtype=np.int16).tobytes())
        error = json.loads(ws.receive_text())
        assert error["status"] == "error"
        assert error["requestId"] == "r1"
        assert "decoder exploded" in error["msg"]
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_text()
        # 1000 would look like a normal end of stream to the client
        assert closed.value.code == 1011
    assert main.STATE["active_streams"] == 0


def test_wrong_secret_key_is_rejected(client):
    with client.websocket_connect("/asr_stream_api_v1") as ws:
        ws.send_text(json.dumps({"requestId": "r1", "secret_key": "nope"}))
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_text()
        assert closed.value.code == 4401


def test_unsupported_language_gets_an_error_message(client):
    with client.websocket_connect("/asr_stream_api_v1") as ws:
        ws.send_text(header(language="klingon"))
        reply = json.loads(ws.receive_text())
        assert reply["status"] == "error"
        assert "klingon" in reply["msg"]


class WorkingModel(BrokenModel):
    def transcribe(self, audio, context="", language=None, return_time_stamps=False):
        return [SimpleNamespace(text="请问软件在哪里|")]


@pytest.fixture
def upload_client(monkeypatch, tmp_path):
    import soundfile as sf

    monkeypatch.setitem(main.STATE, "model", WorkingModel())
    monkeypatch.setitem(main.STATE, "backend", "fake")
    path = tmp_path / "a.wav"
    sf.write(path, np.zeros(16000 * 65, dtype=np.float32), 16000)  # 65 s -> 3 segments
    return TestClient(main.app), path


def sse_events(response):
    return [json.loads(line[len("data: "):]) for line in response.text.splitlines() if line.startswith("data: ")]


def test_transcribe_output_script(upload_client):
    client, path = upload_client
    with open(path, "rb") as f:
        body = client.post("/transcribe", files={"file": f}, data={"output_script": "traditional"}).json()
    assert body["text"] == "\n".join(["請問軟體在哪裡"] * 3)
    with open(path, "rb") as f:
        assert client.post("/transcribe", files={"file": f}).json()["text"].startswith("请问软件")
    with open(path, "rb") as f:
        assert client.post("/transcribe", files={"file": f}, data={"output_script": "pinyin"}).status_code == 400


def test_upload_stream_reports_a_job_and_cleans_it_up(upload_client):
    client, path = upload_client
    with open(path, "rb") as f:
        events = sse_events(client.post("/transcribe/stream", files={"file": f}, data={"output_script": "traditional"}))
    assert [e["type"] for e in events] == ["start", "segment", "segment", "segment", "done"]
    assert events[0]["job_id"] and events[0]["total_segments"] == 3
    assert events[1]["text"] == "請問軟體在哪裡"
    assert main.UPLOAD_JOBS == {}


def test_cancel_stops_before_the_next_segment(upload_client, monkeypatch):
    client, path = upload_client
    real = main._transcribe_segment

    def cancel_after_first(model, segment, context, lang, priority):
        text = real(model, segment, context, lang, priority)
        (job_id,) = main.UPLOAD_JOBS
        assert client.post("/transcribe/cancel", json={"job_id": job_id}).json()["status"] == "cancelling"
        return text

    monkeypatch.setattr(main, "_transcribe_segment", cancel_after_first)
    with open(path, "rb") as f:
        events = sse_events(client.post("/transcribe/stream", files={"file": f}))
    assert [e["type"] for e in events] == ["start", "segment", "cancelled"]
    assert events[-1]["completed_segments"] == 1
    assert main.UPLOAD_JOBS == {}


def test_cancel_unknown_job_is_404(upload_client):
    client, _ = upload_client
    assert client.post("/transcribe/cancel", json={"job_id": "nope"}).status_code == 404


def test_stream_header_output_script_is_echoed_and_validated(client):
    with client.websocket_connect("/asr_stream_api_v1") as ws:
        ws.send_text(header(output_script="traditional", use_vad=False))
        assert json.loads(ws.receive_text())["output_script"] == "traditional"
    with client.websocket_connect("/asr_stream_api_v1") as ws:
        ws.send_text(header(output_script="pinyin"))
        assert json.loads(ws.receive_text())["status"] == "error"


def test_streams_beyond_the_limit_are_turned_away(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_CONCURRENT_STREAMS", 1)
    with client.websocket_connect("/asr_stream_api_v1") as first:
        first.send_text(header(use_vad=False))
        assert json.loads(first.receive_text())["status"] == "connected"
        with client.websocket_connect("/asr_stream_api_v1") as second:
            second.send_text(header(use_vad=False))
            reply = json.loads(second.receive_text())
            assert reply["status"] == "error" and "max concurrent streams" in reply["msg"]
            with pytest.raises(WebSocketDisconnect) as closed:
                second.receive_text()
            assert closed.value.code == 1013
    assert main.STATE["active_streams"] == 0


def gate_priorities(monkeypatch):
    seen = []
    real_run = main.GATE.run

    def spy(priority=0):
        seen.append(priority)
        return real_run(priority)

    monkeypatch.setattr(main.GATE, "run", spy)
    return seen


def test_long_uploads_queue_behind_live_work(upload_client, monkeypatch):
    client, path = upload_client  # 65 s -> three segments
    seen = gate_priorities(monkeypatch)
    with open(path, "rb") as f:
        client.post("/transcribe", files={"file": f})
    assert seen and set(seen) == {main.BATCH}


def test_short_clips_are_treated_as_interactive(upload_client, monkeypatch, tmp_path):
    import soundfile as sf

    client, _ = upload_client
    clip = tmp_path / "clip.wav"
    sf.write(clip, np.zeros(16000 * 4, dtype=np.float32), 16000)
    seen = gate_priorities(monkeypatch)
    with open(clip, "rb") as f:
        client.post("/transcribe", files={"file": f})
    assert seen == [main.LIVE]


@pytest.mark.parametrize("language", [None, "zhen", "auto"])
def test_unset_zhen_and_auto_detect_per_sentence(client, language):
    extra = {"language": language} if language else {}
    with client.websocket_connect("/asr_stream_api_v1") as ws:
        ws.send_text(header(**extra))
        connected = json.loads(ws.receive_text())
    assert connected["language"] == "auto"
    # the default list is trimmed to what the model supports
    assert connected["languages"] == ["Chinese", "English"]


def test_header_can_narrow_the_candidates(client):
    with client.websocket_connect("/asr_stream_api_v1") as ws:
        ws.send_text(header(language="auto", languages=["en", "zh"]))
        connected = json.loads(ws.receive_text())
    assert connected["languages"] == ["English", "Chinese"]


def test_a_specific_language_turns_detection_off(client):
    with client.websocket_connect("/asr_stream_api_v1") as ws:
        ws.send_text(header(language="en", languages=["zh"]))
        connected = json.loads(ws.receive_text())
    assert connected["language"] == "English"
    assert connected["languages"] == []


@pytest.mark.parametrize("languages", [["klingon"], ["auto"], "zh,,en", [1]])
def test_bad_candidates_get_an_error_message(client, languages):
    with client.websocket_connect("/asr_stream_api_v1") as ws:
        ws.send_text(header(language="auto", languages=languages))
        msg = json.loads(ws.receive_text())
    assert msg["status"] == "error"


def test_process_vram_counts_only_this_process(monkeypatch):
    import os
    import subprocess

    out = f"{os.getpid()}, 4411\n99999, 30594\n{os.getpid()}, 100\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=out))
    assert main._process_vram_gb() == round(4511 / 1024, 2)


def test_process_vram_is_unknown_without_nvidia_smi(monkeypatch):
    import subprocess

    def missing(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", missing)
    assert main._process_vram_gb() is None
