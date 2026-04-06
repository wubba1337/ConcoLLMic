from app.commands.run import (
    BinaryTraceCoverage,
    _build_binary_only_target,
    _build_binary_event_space_summary,
    _build_round_input_summary,
    _collect_binary_trace_and_check_coverage,
    _extract_testcase_input_summary,
    _extract_enter_trace_events,
    _load_binary_expected_event_space,
)


class _DummyTestCase:
    def __init__(self, execution_trace: str):
        self.execution_trace = execution_trace


def test_extract_enter_trace_events_filters_and_normalizes():
    trace = "\n".join(
        [
            "[./bin/../bin/validate] enter foo 1",
            "[./bin/../bin/validate] exit foo 1",
            "[Local::validate ]-> [validate] enter foo 9",
            "",
        ]
    )

    events = _extract_enter_trace_events(trace)

    assert events == [("bin/validate", "foo", 1)]


def test_collect_binary_trace_and_check_coverage_tracks_new_events():
    coverage = BinaryTraceCoverage()
    trace = "\n".join(
        [
            "[validate_brackets_plain] enter validate_brackets 1",
            "[validate_brackets_plain] enter validate_brackets 1",
            "[validate_brackets_plain] enter validate_brackets 2",
        ]
    )

    new_events, reached_objective, summary = _collect_binary_trace_and_check_coverage(
        trace,
        coverage,
    )
    assert new_events == 2
    assert reached_objective is True
    assert "Total discovered events: 2" in summary

    new_events_2, reached_objective_2, _ = _collect_binary_trace_and_check_coverage(
        trace,
        coverage,
    )
    assert new_events_2 == 0
    assert reached_objective_2 is False

    # Read-only summary refresh should always report objective as reachable.
    new_events_3, reached_objective_3, _ = _collect_binary_trace_and_check_coverage(
        trace,
        coverage,
        add_coverage=False,
    )
    assert new_events_3 == 0
    assert reached_objective_3 is True


def test_build_binary_only_target_produces_source_free_target():
    coverage = BinaryTraceCoverage()
    coverage.collect_trace("[validate_brackets_plain] enter validate_brackets 9")
    src_testcase = _DummyTestCase(
        "\n".join(
            [
                "[validate_brackets_plain] enter validate_brackets 9",
                "[validate_brackets_plain] enter validate_brackets 10",
            ]
        )
    )

    (
        target_branch,
        justification,
        target_file_lines,
        target_path_constraint,
    ) = _build_binary_only_target(src_testcase, coverage)

    assert "binary-only" in target_branch.lower()
    assert "Discovered so far: 1" in justification
    assert target_file_lines == (None, (None, None))
    assert "Historically discovered events" in target_path_constraint


def test_extract_testcase_input_summary_temp_file_and_argv():
    exec_code = """
def execute_program(timeout: int) -> tuple[str, int]:
    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w") as temp_file:
        temp_file.write(")(")
        temp_file.flush()
        temp_file_path = temp_file.name
        subprocess.run(["frida", "-f", "bin", "--", temp_file_path], capture_output=True)
    return "", 0
"""
    summary = _extract_testcase_input_summary(exec_code)
    assert "file_input=" in summary
    assert "')('" in summary
    assert "argv_input=" in summary
    assert "temp_file_path" in summary


def test_extract_testcase_input_summary_resolves_constant_args():
    exec_code = """
def execute_program(timeout: int) -> tuple[str, int]:
    import subprocess
    arg1 = "1.0"
    arg2 = "2.0"
    subprocess.run(["frida", "-f", "count_plain", "--", arg1, arg2], capture_output=True)
    return "", 0
"""
    summary = _extract_testcase_input_summary(exec_code)
    assert "argv_input=" in summary
    assert "arg1='1.0'" in summary
    assert "arg2='2.0'" in summary


def test_load_binary_expected_event_space_from_map_and_script(monkeypatch, tmp_path):
    import json

    map_path = tmp_path / "frida_trace_map.json"
    map_path.write_text(
        json.dumps(
            {
                "frida_mode": "static_offset",
                "symbols": ["main", "_Z17validate_bracketsPKc"],
                "expected_event_space": {"function": 2, "bb": 3, "branch": 2},
                "symbol_mappings": {
                    "_Z17validate_bracketsPKc": {"ace_block_ids": [5, 6, 6]},
                    "main": {"ace_block_ids": []},
                },
            }
        ),
        encoding="utf-8",
    )
    script_path = tmp_path / "frida_hooks.js"
    script_path.write_text(
        'const HOOK_SPECS = [{"id": 1, "name": "main"}, {"id": 2, "name": "foo"}];',
        encoding="utf-8",
    )
    monkeypatch.setenv("FRIDA_TRACE_MAP_PATH", str(map_path))
    monkeypatch.setenv("FRIDA_SCRIPT_PATH", str(script_path))

    expected = _load_binary_expected_event_space()
    assert expected["function"] == 2
    assert expected["bb"] == 3
    assert expected["branch"] == 2
    assert expected["frida_mode"] == "static_offset"


def test_binary_event_space_summary_and_round_input_summary():
    coverage = BinaryTraceCoverage()
    coverage.collect_trace(
        "\n".join(
            [
                "[validate_brackets_plain] enter _Z17validate_bracketsPKc 1",
                "[validate_brackets_plain] enter _Z17validate_bracketsPKc__bb 100000",
                "[validate_brackets_plain] enter _Z17validate_bracketsPKc__br 200000",
            ]
        )
    )
    summary = _build_binary_event_space_summary(
        coverage,
        {
            "function": 2,
            "bb": None,
            "branch": 4,
            "frida_mode": "static_offset",
            "notes": ["test-note"],
        },
    )
    assert "function events (discovered/expected): 1/2 (50.00%)" in summary
    assert "branch events (discovered/expected): 1/4 (25.00%)" in summary

    tc1 = _DummyTestCase("[x] enter foo 1")
    tc1.id = 1
    tc1.src_id = 0
    tc1.is_target_covered = True
    tc1.is_crash = False
    tc1.is_hang = False
    tc1.exec_code = """
def execute_program(timeout: int) -> tuple[str, int]:
    import subprocess
    arg1 = "abc"
    subprocess.run(["frida", "-f", "bin", "--", arg1], capture_output=True)
    return "", 0
"""
    round_summary = _build_round_input_summary([tc1])
    assert "Generated testcases input summary:" in round_summary
    assert "TestCase #1" in round_summary
    assert "arg1='abc'" in round_summary


def test_load_binary_expected_event_space_from_legacy_hook_specs(monkeypatch, tmp_path):
    script_path = tmp_path / "frida_hooks.js"
    script_path.write_text(
        'const HOOK_SPECS = [{"id": 1, "name": "main"}, {"id": 2, "name": "foo"}];',
        encoding="utf-8",
    )
    monkeypatch.delenv("FRIDA_TRACE_MAP_PATH", raising=False)
    monkeypatch.setenv("FRIDA_SCRIPT_PATH", str(script_path))

    expected = _load_binary_expected_event_space()
    assert expected["function"] == 2
    assert expected["bb"] is None
    assert expected["branch"] is None
