from llama_backend import find_gguf


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
