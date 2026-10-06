from types import SimpleNamespace

import llama_backend
from llama_backend import R2T2LlamaModel, find_gguf


def touch(path):
    path.write_bytes(b"")
    return str(path)


def test_find_gguf_pairs_decoder_with_projector(tmp_path):
    decoder = touch(tmp_path / "Confucius4-R2T2.Q8_0.gguf")
    projector = touch(tmp_path / "Confucius4-R2T2.mmproj-f16.gguf")
    assert find_gguf(str(tmp_path)) == (decoder, projector)


def test_find_gguf_refuses_to_guess(tmp_path):
    assert find_gguf(str(tmp_path)) is None
    touch(tmp_path / "a.Q8_0.gguf")
    assert find_gguf(str(tmp_path)) is None  # no projector
    touch(tmp_path / "a.mmproj-f16.gguf")
    touch(tmp_path / "a.Q4_K_M.gguf")
    assert find_gguf(str(tmp_path)) is None  # two decoders: which one?


class FakeModel:
    """Records which parts of the thinker still exist when it moves to the GPU."""

    def __init__(self):
        self.thinker = SimpleNamespace(model="decoder", lm_head="head")
        self.moved_with = None

    def to(self, device):
        self.moved_with = (device, sorted(vars(self.thinker)))
        return self


def test_decoder_weights_never_reach_the_gpu(monkeypatch):
    """Loading the full model on the GPU and deleting the decoder afterwards
    left 2.6 GB reserved that empty_cache() could not return (GB10, 2026-10-06)."""
    calls = []

    def from_pretrained(model_dir, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(model=FakeModel(), processor=None)

    monkeypatch.setattr(llama_backend.R2T2ASRModel, "from_pretrained", from_pretrained)
    model = R2T2LlamaModel.load("/model", "d.gguf", "p.gguf", lambda *args: "native")

    assert calls[0]["device_map"] == "cpu"
    # only the audio encoder side is left when the weights go to the GPU
    assert model.model.moved_with == ("cuda", [])
