from types import SimpleNamespace

import numpy as np

from infer_gate import InferGate
from stream_session import (
    CHUNK_SAMPLES,
    FIRST_CHUNK_SAMPLES,
    MAX_CATCHUP_CHUNKS,
    PREROLL_SAMPLES,
    StreamSession,
)


class FakeModel:
    """Emits one character of ``script`` per decode step, like an incremental decoder."""

    def __init__(self, script="今天下午開會", final="今天下午開會。"):
        self.script = script
        self.final = final
        self.steps = []
        self.transcribed = []

    def init_streaming_state(self, **kwargs):
        return SimpleNamespace(
            kwargs=kwargs,
            buffer=np.zeros((0,), dtype=np.float32),
            chunk_size_samples=CHUNK_SAMPLES,
            chunk_text=[],
            text="",
            last_fixed_text="",
            n=0,
        )

    def _advance(self, state):
        state.n = min(len(self.script), state.n + 1)
        fixed = self.script[: max(0, state.n - 1)]
        state.chunk_text.append(fixed[len(state.last_fixed_text):])
        state.last_fixed_text = fixed
        state.text = self.script[: state.n]

    def streaming_transcribe_no_reset(self, pcm, state, max_new_tokens, rollback=False):
        assert len(pcm) == state.chunk_size_samples
        self.steps.append(len(pcm))
        self._advance(state)
        return state.text, state.last_fixed_text

    def finish_streaming_transcribe_no_reset(self, state, max_new_tokens):
        self.steps.append(len(state.buffer))
        state.buffer = np.zeros((0,), dtype=np.float32)
        self._advance(state)
        return state.last_fixed_text

    def transcribe(self, audio, context="", language=None, return_time_stamps=False):
        self.transcribed.append((len(audio[0][0]), context, language))
        return [SimpleNamespace(text=self.final)]


class FakeVad:
    """Speech starts/ends on the chunk indexes given."""

    def __init__(self, start_at, end_at):
        self.start_at, self.end_at = start_at, end_at
        self.i = -1

    def detect(self, chunk):
        self.i += 1
        return self.i == self.start_at, self.i == self.end_at


def chunks(n):
    return np.zeros(n * CHUNK_SAMPLES, dtype=np.float32)


def make(model, vad=None, **kwargs):
    return StreamSession(model, InferGate(), request_id="r1", language="Chinese", vad=vad, **kwargs)


def feed_one_by_one(session, n):
    msgs = []
    for _ in range(n):
        msgs += session.feed(chunks(1))
    return [m["msg"] for m in msgs]


def test_silence_is_not_decoded():
    model = FakeModel()
    session = make(model, FakeVad(start_at=99, end_at=99))
    msgs = feed_one_by_one(session, 10)
    assert model.steps == []
    assert all(m["text"] == "" and m["reset"] is False for m in msgs)


def test_sentence_gets_preroll_deltas_and_final_text():
    model = FakeModel()
    session = make(model, FakeVad(start_at=4, end_at=9), context="熱詞")
    msgs = feed_one_by_one(session, 12)

    # audio from just before speech start is decoded along with the first chunk
    assert model.steps[0] == PREROLL_SAMPLES + CHUNK_SAMPLES
    resets = [m for m in msgs if m["reset"]]
    assert len(resets) == 1
    end = resets[0]
    assert end["final_text"] == "今天下午開會。"
    assert end["final_pass"] is True
    # deltas are append-only and, with the uncommitted tail, add up to what was streamed
    streamed = "".join(m["text"] for m in msgs[: msgs.index(end) + 1])
    assert streamed == model.script[: len(model.steps)]
    # the one-pass decode covers exactly the audio decoded for this sentence
    assert model.transcribed == [(sum(model.steps), "熱詞", ["Chinese"])]
    # after the sentence the session waits for speech again
    assert msgs[-1]["text"] == "" and msgs[-1]["reset"] is False


def test_final_pass_can_be_turned_off():
    model = FakeModel()
    session = make(model, FakeVad(start_at=0, end_at=3), final_pass=False)
    end = [m for m in feed_one_by_one(session, 5) if m["reset"]][0]
    assert model.transcribed == []
    assert end["final_pass"] is False
    assert end["final_text"] == model.script[:4]


def test_backlog_is_merged_into_bigger_steps():
    model = FakeModel()
    session = make(model, FakeVad(start_at=0, end_at=99))
    session.feed(chunks(1))
    session.feed(chunks(MAX_CATCHUP_CHUNKS + 3))
    assert model.steps == [CHUNK_SAMPLES, MAX_CATCHUP_CHUNKS * CHUNK_SAMPLES, 3 * CHUNK_SAMPLES]


class ScriptedVad:
    """Start/end events at given chunk indexes; several sentences allowed."""

    def __init__(self, starts, ends):
        self.starts, self.ends = set(starts), set(ends)
        self.i = -1

    def detect(self, chunk):
        assert len(chunk) == CHUNK_SAMPLES, "VAD must see every 160 ms chunk on its own"
        self.i += 1
        return self.i in self.starts, self.i in self.ends


def test_backlog_never_swallows_a_sentence_boundary():
    """Two sentences arriving in one backlog must still come out as two."""
    model = FakeModel()
    session = make(model, ScriptedVad(starts=[1, 6], ends=[3, 9]))
    msgs = [m["msg"] for m in session.feed(chunks(12))]
    resets = [m for m in msgs if m["reset"]]
    assert len(resets) == 2
    # one decode per sentence, each covering its own audio (preroll included) and nothing else
    assert model.steps == [
        1 * CHUNK_SAMPLES + 3 * CHUNK_SAMPLES,
        2 * CHUNK_SAMPLES + 4 * CHUNK_SAMPLES,
    ]
    assert [n for n, _, _ in model.transcribed] == model.steps


def test_partial_chunk_waits_for_more_audio():
    model = FakeModel()
    session = make(model, FakeVad(start_at=0, end_at=99))
    assert session.feed(np.zeros(CHUNK_SAMPLES - 1, dtype=np.float32)) == []
    assert len(session.feed(np.zeros(1, dtype=np.float32))) == 1


def test_finish_flushes_tail_and_closes_sentence():
    model = FakeModel()
    session = make(model, FakeVad(start_at=0, end_at=99))
    session.feed(chunks(2))
    session.feed(np.zeros(100, dtype=np.float32))
    end = session.finish()["msg"]
    assert model.steps[-1] == 100
    assert end["reset"] is True
    assert end["final_text"] == "今天下午開會。"


def test_finish_without_speech_returns_empty_reset():
    model = FakeModel()
    session = make(model, FakeVad(start_at=99, end_at=99))
    session.feed(chunks(3))
    end = session.finish()["msg"]
    assert end == {"text": "", "reset": True, "final_text": ""}
    assert model.steps == []


def test_without_vad_decoding_starts_after_first_window():
    model = FakeModel()
    session = make(model, vad=None)
    assert session.feed(chunks(1)) == []
    session.feed(chunks(1))
    assert model.steps == [FIRST_CHUNK_SAMPLES]


def test_repetition_forces_a_reset_without_final_pass():
    model = FakeModel(script="好的" * 20)
    session = make(model, FakeVad(start_at=0, end_at=99))
    msgs = feed_one_by_one(session, 14)
    resets = [m for m in msgs if m["reset"]]
    assert resets and resets[0]["final_pass"] is False
    assert model.transcribed == []
    assert len(resets[0]["final_text"]) < 8


def test_traditional_output_converts_deltas_and_final_text():
    model = FakeModel(script="请问软件在哪里", final="请问软件在哪里。")
    session = make(model, FakeVad(start_at=0, end_at=6), output_script="traditional")
    msgs = feed_one_by_one(session, 8)
    end = [m for m in msgs if m["reset"]][0]
    streamed = "".join(m["text"] for m in msgs[: msgs.index(end) + 1])
    # the phrase 软件 arrives one character per step and still becomes 軟體
    assert streamed == "請問軟體在哪裡"
    assert end["final_text"] == "請問軟體在哪裡。"


def test_finish_keeps_the_text_decoded_from_the_tail():
    model = FakeModel(script="今天下午開會")
    session = make(model, FakeVad(start_at=0, end_at=99))
    streamed = "".join(m["msg"]["text"] for m in feed_one_by_one_raw(session, 3))
    session.feed(np.zeros(100, dtype=np.float32))
    end = session.finish()["msg"]
    # three full chunks plus the tail are four decode steps; nothing they produced may be lost
    assert len(model.steps) == 4
    assert streamed + end["text"] == model.script[:4]


def feed_one_by_one_raw(session, n):
    msgs = []
    for _ in range(n):
        msgs += session.feed(chunks(1))
    return msgs
