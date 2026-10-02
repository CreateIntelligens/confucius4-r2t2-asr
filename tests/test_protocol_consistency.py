"""The stream protocol is implemented in two servers and described in two more
places; these tests fail when one of them drifts from the others."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def eos_of(source, name):
    return re.search(rf'^{name}\s*=\s*"([^"]+)"', source, flags=re.M).group(1)


def test_eos_string_is_the_same_everywhere():
    main_eos = eos_of(read("app/main.py"), "STREAM_EOS")
    server_eos = eos_of(read("server.py"), "YOUDAO_ONETIME_ASR_EOS_STRING")
    assert main_eos == server_eos
    assert main_eos in read("README.md")
    assert f'ws.send("{main_eos}")' in read("web/assets/app.js")


def test_both_servers_expose_the_stream_endpoint():
    assert '@app.websocket("/asr_stream_api_v1")' in read("app/main.py")
    assert '@app.websocket("/asr_stream_api_v1")' in read("server.py")
    assert "/asr_stream_api_v1" in read("web/assets/app.js")


def test_server_py_never_calls_the_model_on_the_event_loop():
    """Inside the websocket handler every model call must go through run_infer."""
    source = read("server.py")
    handler = source[source.index("async def asr_stream_api_v1"):source.index("def args_parser")]
    direct = re.findall(r"(?<!run_infer\(\n)(?<![\w.])asr_model\.\w+\(", handler)
    assert direct == []
    assert "stream_vad.reset()" not in handler
    assert "stream_vad.detect_chunk" not in handler


def test_web_language_options_are_understood_by_the_server():
    from textproc import normalize_language

    options = re.findall(r'<option value="([^"]+)"', read("web/index.html").split('id="lang-select"')[1].split("</select>")[0])
    assert "zhen" in options
    supported = {"Chinese", "English", "Cantonese", "Japanese", "Korean", "French", "German", "Spanish"}
    for code in options:
        assert normalize_language(code) in supported | {None}


def test_nothing_documents_a_wrong_eos_string():
    main_eos = eos_of(read("app/main.py"), "STREAM_EOS")
    for rel in ("README.md", "examples/client_stream_demo.py", "web/assets/app.js"):
        text = read(rel)
        for found in re.findall(r"youdao_onetime_asr\w*", text, flags=re.I):
            if found.upper().endswith("EOS"):
                assert found == main_eos, f"{rel}: {found}"
        assert "asr_eos_string\"" not in text.lower().replace("youdao_onetime_asr_eos_string = ", "")


def test_server_py_transcribe_goes_through_the_gate():
    source = read("server.py")
    handlers = source[source.index("async def transcribe_audio_segment"):source.index("# Streaming WebSocket v1")]
    # decoding audio bytes may use a plain thread; model calls may not
    assert re.findall(r"(?<![\w.])asr_model\.\w+\(", handlers) == []
    assert "to_thread(_transcribe" not in handlers and "to_thread(run_infer" not in handlers


def test_upload_stream_events_match_between_servers_and_ui():
    """app.js drives the upload progress from these SSE event types and fields."""
    ui = read("web/assets/app.js")
    for server in ("server.py", "app/main.py"):
        source = read(server)
        assert '"/transcribe/stream"' in source
        assert '"/transcribe/cancel"' in source
        for event_type in ("start", "segment", "done", "error", "cancelled"):
            assert f'"type": "{event_type}"' in source, (server, event_type)
            assert f"event.type === '{event_type}'" in ui
        for field in ("job_id", "completed_segments", "output_script", "total_segments", "duration_sec", "start_sec", "end_sec", "index", "total", "text"):
            assert f'"{field}"' in source, (server, field)


def test_both_servers_serve_the_ui_assets():
    assert 'app.static("/assets"' in read("server.py")
    assert 'app.mount("/assets"' in read("app/main.py")
    assert 'src="/assets/app.js"' in read("web/index.html")


def test_both_servers_accept_the_same_output_scripts():
    from script_convert import OUTPUT_SCRIPTS

    server = read("server.py")
    for script in OUTPUT_SCRIPTS:
        assert f'"{script}"' in server
        assert f'value="{script}"' in read("web/index.html")
    assert 'OpenCC("s2twp")' in server and 'OpenCC("s2twp")' in read("app/script_convert.py")


def test_both_servers_map_language_codes_the_same_way():
    """server.py and app/main.py each carry a language table; they must agree,
    including the deliberate choice that zhen means Chinese rather than auto."""
    import ast

    from textproc import normalize_language

    source = read("server.py")
    table = ast.literal_eval(re.search(r"^_LANG_MAP = (\{.*?^\})", source, flags=re.M | re.S).group(1))
    assert table["zhen"] == "Chinese"
    for code, expected in table.items():
        assert normalize_language(code) == expected, code
    # a stream header without a language is treated as zhen by both
    assert 'normalize_asr_language(header.get("language") or "zhen")' in source
    assert normalize_language(None, "zhen") == "Chinese"


def test_both_servers_serve_llms_txt():
    assert '@app.route("/llms.txt")' in read("server.py")
    assert '@app.get("/llms.txt")' in read("app/main.py")
    assert 'href="/llms.txt"' in read("web/index.html")
    assert read("llms.txt") == read("web/llms.txt")
