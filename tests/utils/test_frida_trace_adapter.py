from app.utils.frida_trace_adapter import adapt_frida_trace_to_ace


def test_adapt_frida_trace_to_ace_branch_mapping():
    trace_map = {
        "symbol_mappings": {
            "_Z17validate_bracketsPKc": {
                "ace_mapping_available": True,
                "ace_file": "validate_brackets.cpp",
                "ace_function": "validate_brackets",
                "ace_block_ids": [5, 6, 7],
            }
        },
        "frida_function_id_to_ace_block_id": {"1": 5},
    }

    frida_trace = "\n".join(
        [
            "[validate_brackets_plain] enter _Z17validate_bracketsPKc__br 200000",
            "[validate_brackets_plain] enter _Z17validate_bracketsPKc__br 200001",
            "[validate_brackets_plain] enter _Z17validate_bracketsPKc__br 200000",
        ]
    )

    adapted, stats = adapt_frida_trace_to_ace(frida_trace, trace_map)
    lines = [line for line in adapted.splitlines() if line.strip()]

    assert lines == [
        "[validate_brackets.cpp] enter validate_brackets 5",
        "[validate_brackets.cpp] enter validate_brackets 6",
        "[validate_brackets.cpp] enter validate_brackets 5",
    ]
    assert stats["mapped_lines"] == 3
    assert stats["skipped_unknown_symbol"] == 0


def test_adapt_frida_trace_to_ace_skips_unmapped_symbols():
    trace_map = {"symbol_mappings": {}, "frida_function_id_to_ace_block_id": {}}
    frida_trace = "[validate_brackets_plain] enter _Z17validate_bracketsPKc__br 200000"

    adapted, stats = adapt_frida_trace_to_ace(frida_trace, trace_map)
    assert adapted == ""
    assert stats["mapped_lines"] == 0
    assert stats["skipped_unknown_symbol"] == 1

