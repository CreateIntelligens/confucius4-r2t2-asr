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
