from app.commands.run import split_trace_by_file


def test_split_trace_by_file_ignores_prefixed_lines():
    trace = "\n".join(
        [
            "[Local::validate_brackets_plain ]-> [validate_brackets_plain] enter _Z17validate_bracketsPKc__br 200000",
            "[validate_brackets.cpp] enter validate_brackets 1",
        ]
    )

    result = split_trace_by_file(trace)
    assert list(result.keys()) == ["validate_brackets.cpp"]
    assert result["validate_brackets.cpp"] == (
        "[validate_brackets.cpp] enter validate_brackets 1"
    )


def test_split_trace_by_file_ignores_non_exact_lines():
    trace = "\n".join(
        [
            "[validate_brackets.cpp] enter validate_brackets 1 trailing_text",
            "[validate_brackets.cpp] exit validate_brackets 1",
        ]
    )

    result = split_trace_by_file(trace)
    assert result["validate_brackets.cpp"] == (
        "[validate_brackets.cpp] exit validate_brackets 1"
    )


def test_adapt_frida_trace_if_needed(monkeypatch, tmp_path):
    import json

    from app.commands.run import _adapt_frida_trace_if_needed

    trace_map = {
        "symbol_mappings": {
            "_Z17validate_bracketsPKc": {
                "ace_mapping_available": True,
                "ace_file": "validate_brackets.cpp",
                "ace_function": "validate_brackets",
                "ace_block_ids": [1, 2],
            }
        },
        "frida_function_id_to_ace_block_id": {},
    }
    map_path = tmp_path / "frida_trace_map.json"
    map_path.write_text(json.dumps(trace_map), encoding="utf-8")
    monkeypatch.setenv("FRIDA_TRACE_MAP_PATH", str(map_path))

    raw = "[validate_brackets_plain] enter _Z17validate_bracketsPKc__br 200000"
    adapted = _adapt_frida_trace_if_needed(raw)
    assert adapted == "[validate_brackets.cpp] enter validate_brackets 1"


def test_adapt_frida_trace_if_needed_for_function_channel(monkeypatch, tmp_path):
    import json

    from app.commands.run import _adapt_frida_trace_if_needed

    trace_map = {
        "symbol_mappings": {
            "_Z17validate_bracketsPKc": {
                "ace_mapping_available": True,
                "ace_file": "validate_brackets.cpp",
                "ace_function": "validate_brackets",
                "ace_block_ids": [1, 2],
            }
        },
        "frida_function_id_to_ace_block_id": {"1": 2},
    }
    map_path = tmp_path / "frida_trace_map.json"
    map_path.write_text(json.dumps(trace_map), encoding="utf-8")
    monkeypatch.setenv("FRIDA_TRACE_MAP_PATH", str(map_path))

    raw = "[validate_brackets_plain] enter _Z17validate_bracketsPKc 1"
    adapted = _adapt_frida_trace_if_needed(raw)
    assert adapted == "[validate_brackets.cpp] enter validate_brackets 2"
