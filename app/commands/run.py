"""
Concolic execution command for ACE.
"""

import atexit
import ast
import concurrent.futures
import json
import os
import random
import re
import sys
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from enum import Enum

from loguru import logger

from app.agents.agent_scheduling import TestCaseScheduler
from app.agents.agent_solver import review_solve, solve
from app.agents.agent_summarizer import review_summary, summarize
from app.agents.common import (
    FILE_TRACE_PATTERN,
    TRACE_PATTERN,
    filter_instr_print,
    set_concolic_execution_state,
)
from app.agents.coverage import Coverage
from app.agents.states import ConcolicExecutionState, TestcaseState
from app.agents.testcase import TestCase, TestCaseManager
from app.agents.trace import trace_compress
from app.data_structures import MessageThread
from app.log import print_phase, print_step
from app.model.common import Usage
from app.utils.utils import (
    compress_repeating_sequences,
    format_code,
    run_target,
    update_ce_start_time,
    update_project_dir,
)


def _format_trace_event(event: tuple[str, str, int]) -> str:
    file_path, func_name, block_id = event
    return f"[{file_path}] enter {func_name} {block_id}"


def _extract_enter_trace_events(trace_str: str) -> list[tuple[str, str, int]]:
    """Extract exact enter-events from trace lines."""
    if not trace_str:
        return []

    events: list[tuple[str, str, int]] = []
    for line in trace_str.strip().split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        match = re.fullmatch(FILE_TRACE_PATTERN, stripped)
        if match is None:
            continue

        file_path, action, func_name, block_id = match.groups()
        if action != "enter":
            continue
        events.append((os.path.normpath(file_path), func_name, int(block_id)))
    return events


def _dedup_preserve_order(
    events: list[tuple[str, str, int]],
) -> list[tuple[str, str, int]]:
    seen = set()
    result = []
    for event in events:
        if event in seen:
            continue
        seen.add(event)
        result.append(event)
    return result


def _classify_event_channel(func_name: str, block_id: int) -> str:
    """Classify event channel from function suffix / id range."""
    if func_name.endswith("__br") or block_id >= 200000:
        return "branch"
    if func_name.endswith("__bb") or block_id >= 100000:
        return "bb"
    return "function"


class BinaryTraceCoverage:
    """Coverage tracker for source-independent (binary-only) trace events."""

    def __init__(self):
        self.event2cov: dict[tuple[str, str, int], int] = {}

    def collect_trace(
        self,
        trace_str: str,
        add_coverage: bool = True,
    ) -> tuple[int, str]:
        events = _dedup_preserve_order(_extract_enter_trace_events(trace_str))
        new_events: list[tuple[str, str, int]] = []

        if add_coverage:
            for event in events:
                prev_cov = self.event2cov.get(event, 0)
                self.event2cov[event] = prev_cov + 1
                if prev_cov == 0:
                    new_events.append(event)

        lines = [
            "Binary-only trace summary",
            f"- Enter events in this execution: {len(events)}",
            f"- Newly discovered events: {len(new_events)}",
            f"- Total discovered events: {len(self.event2cov)}",
        ]

        if events:
            lines.append("- Observed events (first 120):")
            for event in events[:120]:
                lines.append(f"  {_format_trace_event(event)}")
        else:
            lines.append("- Observed events: (none)")

        return len(new_events), "\n".join(lines)

    def get_seen_events(self) -> list[tuple[str, str, int]]:
        return sorted(self.event2cov.keys())

    def get_discovered_counts_by_channel(self) -> dict[str, int]:
        counts = {"function": 0, "bb": 0, "branch": 0, "total": 0}
        for _, func_name, block_id in self.event2cov.keys():
            channel = _classify_event_channel(func_name, block_id)
            counts[channel] += 1
        counts["total"] = counts["function"] + counts["bb"] + counts["branch"]
        return counts


def _load_binary_expected_event_space() -> dict[str, int | str | list[str] | None]:
    """Load expected event-space metadata for binary-only runs when available."""
    expected: dict[str, int | str | list[str] | None] = {
        "function": None,
        "bb": None,
        "branch": None,
        "frida_mode": None,
        "notes": [],
    }

    map_path = os.environ.get("FRIDA_TRACE_MAP_PATH")
    if map_path and os.path.exists(map_path):
        try:
            with open(map_path, encoding="utf-8") as f:
                trace_map = json.load(f)

            frida_mode = trace_map.get("frida_mode")
            if isinstance(frida_mode, str):
                expected["frida_mode"] = frida_mode

            expected_event_space = trace_map.get("expected_event_space")
            if isinstance(expected_event_space, dict):
                for channel in ("function", "bb", "branch"):
                    value = expected_event_space.get(channel)
                    if isinstance(value, int):
                        expected[channel] = value
                if any(isinstance(expected_event_space.get(k), int) for k in ("function", "bb", "branch")):
                    expected["notes"].append(
                        f"expected event space from trace map ({map_path})"
                    )

            symbols = trace_map.get("symbols")
            if expected["function"] is None and isinstance(symbols, list):
                expected["function"] = len({str(s) for s in symbols if str(s)})
                expected["notes"].append(f"function expected from trace map symbols ({map_path})")
            elif expected["function"] is None:
                symbol_mappings = trace_map.get("symbol_mappings")
                if isinstance(symbol_mappings, dict):
                    expected["function"] = len(symbol_mappings)
                    expected["notes"].append(
                        f"function expected from symbol_mappings ({map_path})"
                    )

            # Fallback: derive branch expectation from mapped ACE block IDs.
            if expected["branch"] is None:
                symbol_mappings = trace_map.get("symbol_mappings", {})
                if isinstance(symbol_mappings, dict):
                    ace_branch_ids: set[int] = set()
                    for info in symbol_mappings.values():
                        if not isinstance(info, dict):
                            continue
                        block_ids = info.get("ace_block_ids")
                        if not isinstance(block_ids, list):
                            continue
                        for block_id in block_ids:
                            if isinstance(block_id, int):
                                ace_branch_ids.add(block_id)
                    if ace_branch_ids:
                        expected["branch"] = len(ace_branch_ids)
                        expected["notes"].append(
                            f"branch expected from mapped ace_block_ids ({map_path})"
                        )
        except Exception as e:
            logger.warning("Failed to load FRIDA_TRACE_MAP_PATH {}: {}", map_path, e)

    # Fallback: parse HOOK_SPECS from generated script for channel-wise expected counts.
    script_path = os.environ.get("FRIDA_SCRIPT_PATH")
    if (
        (expected["function"] is None or expected["bb"] is None or expected["branch"] is None)
        and script_path
        and os.path.exists(script_path)
    ):
        try:
            script_text = open(script_path, encoding="utf-8").read()
            hook_specs_match = re.search(
                r"const\s+HOOK_SPECS\s*=\s*(\[[\s\S]*?\]);",
                script_text,
            )
            if hook_specs_match:
                hook_specs = json.loads(hook_specs_match.group(1))
                if isinstance(hook_specs, list):
                    channel_counts = {"function": 0, "bb": 0, "branch": 0}
                    has_channel_field = False
                    for spec in hook_specs:
                        if not isinstance(spec, dict):
                            continue
                        channel = spec.get("channel")
                        if channel in channel_counts:
                            has_channel_field = True
                            channel_counts[channel] += 1

                    # Backward compatibility: older scripts had only function hooks
                    # and no explicit "channel" field.
                    if not has_channel_field:
                        channel_counts["function"] = len(hook_specs)

                    if expected["function"] is None:
                        expected["function"] = channel_counts["function"] or None
                    if expected["bb"] is None:
                        expected["bb"] = channel_counts["bb"] or None
                    if expected["branch"] is None:
                        expected["branch"] = channel_counts["branch"] or None
                    expected["notes"].append(
                        f"channel expected counts from HOOK_SPECS ({script_path})"
                    )
        except Exception as e:
            logger.warning("Failed to parse FRIDA_SCRIPT_PATH {}: {}", script_path, e)

    return expected


def _format_discovered_vs_expected(discovered: int, expected: int | None) -> str:
    if expected is None:
        return f"{discovered}/unknown"
    if expected == 0:
        return f"{discovered}/0"
    return f"{discovered}/{expected} ({discovered / expected:.2%})"


def _build_binary_event_space_summary(
    binary_coverage: BinaryTraceCoverage,
    expected_space: dict[str, int | str | list[str] | None] | None,
) -> str:
    discovered = binary_coverage.get_discovered_counts_by_channel()

    expected_function = None
    expected_bb = None
    expected_branch = None
    notes: list[str] = []
    frida_mode = None
    if expected_space:
        expected_function = expected_space.get("function")
        expected_bb = expected_space.get("bb")
        expected_branch = expected_space.get("branch")
        frida_mode = expected_space.get("frida_mode")
        notes = [
            str(note) for note in (expected_space.get("notes") or []) if str(note).strip()
        ]

    lines = [
        "Binary-only event space coverage:",
        f"- total discovered events: {discovered['total']}",
        f"- function events (discovered/expected): {_format_discovered_vs_expected(discovered['function'], expected_function)}",
        f"- basic-block events (discovered/expected): {_format_discovered_vs_expected(discovered['bb'], expected_bb)}",
        f"- branch events (discovered/expected): {_format_discovered_vs_expected(discovered['branch'], expected_branch)}",
    ]
    if frida_mode:
        lines.append(f"- frida mode: {frida_mode}")
    if notes:
        lines.append(f"- expected-space source: {'; '.join(notes)}")
    else:
        lines.append("- expected-space source: unavailable (dynamic/runtime-only mapping)")
    return "\n".join(lines)


def _shorten_for_display(value: str, max_len: int = 120) -> str:
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def _expr_to_summary(node: ast.AST, constant_bindings: dict[str, object]) -> str:
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.Name):
        if node.id in constant_bindings:
            return f"{node.id}={repr(constant_bindings[node.id])}"
        return node.id
    try:
        return ast.unparse(node)
    except Exception:
        return node.__class__.__name__


def _extract_testcase_input_summary(exec_code: str | None) -> str:
    """Best-effort extraction of concrete program inputs from execute_program code."""
    if not exec_code:
        return "(no execute_program code)"

    try:
        tree = ast.parse(exec_code)
    except SyntaxError:
        return "(unable to parse execute_program code)"

    execute_fn: ast.AST | None = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "execute_program":
            execute_fn = node
            break
    if execute_fn is None:
        execute_fn = tree

    constant_bindings: dict[str, object] = {}
    for node in ast.walk(execute_fn):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        if not isinstance(node.value, ast.Constant):
            continue
        value = node.value.value
        if isinstance(value, (str, int, float, bool)):
            constant_bindings[node.targets[0].id] = value

    file_input_exprs: list[str] = []
    argv_exprs: list[str] = []

    for node in ast.walk(execute_fn):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == "write" and node.args:
            file_input_exprs.append(_expr_to_summary(node.args[0], constant_bindings))
            continue

        is_subprocess_run = (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "run"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "subprocess"
        )
        if not is_subprocess_run:
            continue
        if not node.args:
            continue
        if not isinstance(node.args[0], (ast.List, ast.Tuple)):
            continue

        elements = list(node.args[0].elts)
        literal_values = [
            elt.value if isinstance(elt, ast.Constant) else None for elt in elements
        ]

        start_idx = None
        for idx, literal in enumerate(literal_values):
            if literal == "--":
                start_idx = idx + 1
                break

        # In non-Frida harnesses there is no "--"; fall back to skipping argv[0].
        if start_idx is None:
            start_idx = 1 if len(elements) > 1 else len(elements)

        for elt in elements[start_idx:]:
            argv_exprs.append(_expr_to_summary(elt, constant_bindings))

    parts: list[str] = []
    if file_input_exprs:
        shown = ", ".join(_shorten_for_display(v, 80) for v in file_input_exprs[:3])
        parts.append(f"file_input={shown}")
    if argv_exprs:
        shown = ", ".join(_shorten_for_display(v, 80) for v in argv_exprs[:5])
        parts.append(f"argv_input={shown}")

    if not parts:
        return "(input extraction unavailable)"
    return " | ".join(parts)


def _build_round_input_summary(testcases: list[TestCase]) -> str:
    if not testcases:
        return "Generated testcases input summary: (none)"

    lines = ["Generated testcases input summary:"]
    for tc in sorted(testcases, key=lambda t: t.id):
        status = "target✓" if tc.is_target_covered else "target✗"
        if tc.is_crash:
            status += ", crash"
        if tc.is_hang:
            status += ", hang"
        lines.append(
            f"- TestCase #{tc.id} (src={tc.src_id}, {status}): {_extract_testcase_input_summary(tc.exec_code)}"
        )
    return "\n".join(lines)


def _adapt_frida_trace_if_needed(trace_str: str) -> str:
    """Best-effort adaptation from raw Frida trace lines to ACE trace format.

    This handles cases where solver-generated execution code returns raw Frida
    output (module/symbol trace lines) instead of already-adapted ACE lines.
    """
    if not trace_str:
        return trace_str

    # No trace-like lines to adapt.
    if re.search(FILE_TRACE_PATTERN, trace_str) is None:
        return trace_str

    map_path = os.environ.get("FRIDA_TRACE_MAP_PATH")
    if not map_path or not os.path.exists(map_path):
        return trace_str

    try:
        from app.utils.frida_trace_adapter import (
            adapt_frida_trace_to_ace,
            load_frida_trace_map,
        )

        trace_map = load_frida_trace_map(map_path)
        adapted_trace, stats = adapt_frida_trace_to_ace(trace_str, trace_map)
        if adapted_trace.strip() and stats.get("mapped_lines", 0) > 0:
            logger.debug(
                "Adapted raw Frida trace using {} (mapped_lines={}, skipped_non_trace_lines={}, skipped_unknown_symbol={})",
                map_path,
                stats.get("mapped_lines", 0),
                stats.get("skipped_non_trace_lines", 0),
                stats.get("skipped_unknown_symbol", 0),
            )
            return adapted_trace
    except Exception as e:
        logger.warning("Failed to adapt raw Frida trace using {}: {}", map_path, e)

    return trace_str


def split_trace_by_file(trace_str: str) -> dict[str, str]:
    """Split execution trace by file name.

    Parses the execution trace string, filters out instrumentation output, and groups them by source file.

    Args:
        trace_str (str): Execution trace string

    Returns:
        dict: Dictionary with file names as keys and corresponding trace information as values
    """
    # Regular expression to search "[filename] enter/exit function_name basic_block_id"
    # Note: OTHER content (the content that is not instrumentation output) would be detected

    # Create a default dictionary to store trace information grouped by file name
    file_traces = defaultdict(list)

    # Split trace information by line
    for line in trace_str.strip().split("\n"):
        if not line.strip():
            continue

        # Require a strict single trace line to avoid parsing tool/runtime logs
        # that may contain a trace-like substring (e.g., Frida REPL prefixes).
        match = re.fullmatch(FILE_TRACE_PATTERN, line.strip())
        if match is None:
            continue

        file_path, action, func_name, block_id = match.groups()
        assert (
            re.search(TRACE_PATTERN, f"{action} {func_name} {block_id}") is not None
        )
        file_traces[file_path].append(f"[{file_path}] {action} {func_name} {block_id}")

    # Convert lists to strings
    return {
        os.path.normpath(file_path): "\n".join(traces)
        for file_path, traces in file_traces.items()
    }


def _format_exec_code(exec_code: str) -> str:
    """Format execution code.

    Args:
        exec_code (str): Execution code
    """
    exec_code_dict = {
        line_no: code for line_no, code in enumerate(exec_code.split("\n"), start=1)
    }
    return format_code(exec_code_dict, "python", numbered=False, qouted=True)


def _get_file_lines(
    file_path: str | None, line_range: tuple[int, int] | tuple[None, None]
) -> str:
    """Get the content of the file lines.

    Args:
        file_path (str): File path
        line_range (tuple): Line range
    """
    if file_path is None or line_range == (None, None):
        return None

    coverage = Coverage.get_instance()
    _contents = []
    for real_line in range(line_range[0], line_range[1] + 1):
        _contents.append(
            coverage.get_file_coverage(file_path).get_real_line_content(real_line)
        )
    return "\n".join(_contents)


def _collect_trace_and_check_coverage(
    execution_trace: str | None,
    target_file_lines: tuple[str, tuple[int, int]] | tuple[None, tuple[None, None]],
    add_coverage: bool = True,
) -> tuple[int, bool, str]:
    """Collect trace and check coverage.

    Args:
        execution_trace (str): Execution trace
        target_file_lines (tuple): Target file lines
        add_coverage (bool): Whether to add coverage information
    Returns:
        tuple: Tuple of (new_covered_lines, is_target_lines_covered, execution_summary)
    """
    if execution_trace is None or execution_trace == "":
        return 0, False, ""

    execution_trace = _adapt_frida_trace_if_needed(execution_trace)

    coverage = Coverage.get_instance()
    file_traces = split_trace_by_file(execution_trace)
    is_target_lines_covered = False if target_file_lines[0] is not None else True
    new_covered_lines = 0
    execution_summary = ""
    for relative_file_path, trace in file_traces.items():
        relative_file_path = os.path.normpath(relative_file_path)

        # Collect trace and check target coverage
        file_new_covered_lines, file_target_lines_covered, file_execution_summary = (
            coverage.collect_trace(
                relative_file_path,
                trace,
                target_lines=(
                    target_file_lines[1]
                    if target_file_lines[0] is not None
                    and relative_file_path == os.path.normpath(target_file_lines[0])
                    else (None, None)
                ),
                add_coverage=add_coverage,
            )
        )

        if target_file_lines[0] is not None and relative_file_path == os.path.normpath(
            target_file_lines[0]
        ):
            is_target_lines_covered = file_target_lines_covered

        new_covered_lines += file_new_covered_lines
        execution_summary += file_execution_summary + "\n"

    return new_covered_lines, is_target_lines_covered, execution_summary


def _collect_binary_trace_and_check_coverage(
    execution_trace: str | None,
    binary_coverage: BinaryTraceCoverage,
    add_coverage: bool = True,
) -> tuple[int, bool, str]:
    """Collect trace for binary-only mode (no source file dependency)."""
    if execution_trace is None or execution_trace == "":
        return 0, False, ""

    new_covered_events, execution_summary = binary_coverage.collect_trace(
        execution_trace,
        add_coverage=add_coverage,
    )
    # In binary-only mode, the round objective is to discover at least one NEW event.
    is_target_covered = new_covered_events > 0 if add_coverage else True
    return new_covered_events, is_target_covered, execution_summary


def _build_binary_only_target(
    src_testcase: TestCase,
    binary_coverage: BinaryTraceCoverage,
) -> tuple[str, str, tuple[None, tuple[None, None]], str]:
    """Build a synthetic target/constraint for source-independent exploration."""
    seen_events = binary_coverage.get_seen_events()
    src_events = _dedup_preserve_order(_extract_enter_trace_events(src_testcase.execution_trace))

    target_branch = "Discover at least one NEW runtime trace event (binary-only)"
    justification = (
        f"Binary-only mode objective: maximize unique runtime trace events. "
        f"Discovered so far: {len(seen_events)}."
    )

    observed_lines = [_format_trace_event(event) for event in src_events[:80]]
    observed_block = "\n".join(f"- {line}" for line in observed_lines) or "- (none)"
    historical_lines = [_format_trace_event(event) for event in seen_events[:120]]
    historical_block = "\n".join(f"- {line}" for line in historical_lines) or "- (none)"

    target_path_constraint = (
        "Goal: Generate NEW concrete inputs for the existing harness so execution reveals at least one NEW "
        "runtime trace event not observed before.\n"
        "Requirements:\n"
        "1. Keep the harness structure and target binary invocation unchanged.\n"
        "2. Only modify external inputs (arguments, temp file content, environment variables) used by the harness.\n"
        "3. Aim to hit code paths that produce new `[file] enter function block_id` events.\n\n"
        "Observed events in source execution (sample):\n"
        f"{observed_block}\n\n"
        "Historically discovered events (sample):\n"
        f"{historical_block}\n"
    )

    return target_branch, justification, (None, (None, None)), target_path_constraint


class TestCaseSelection(Enum):
    LLM = "llm"
    RANDOM = "random"
    DFS = "dfs"


def solve_and_execute(
    tc: TestCase,
    summarize_msg_thread: MessageThread,
    exec_timeout: int,
    disable_review: bool = True,
    binary_only: bool = False,
    binary_trace_coverage: BinaryTraceCoverage | None = None,
):
    # process single test case, including solve and execute
    try:
        tc.add_state(TestcaseState.SOLVE)
        tc.save_to_disk()
        solver_msg_thread = None

        while tc.current_state != TestcaseState.FINISHED:

            usage_details = {"TOTAL": Usage()}

            if disable_review:
                assert not str(tc.current_state).startswith("REVIEW")

            if (
                tc.current_state == TestcaseState.SOLVE
                or tc.current_state == TestcaseState.REVIEW_SUMMARY_SOLVE
            ):
                # Solve constraints to generate new inputs
                if tc.current_state == TestcaseState.SOLVE:
                    logger.info(
                        "TestCase #{}: Solving path constraint...",
                        tc.id,
                    )
                    print_step(f"TestCase #{tc.id}: Solving path constraint...")
                else:
                    logger.info(
                        "TestCase #{}: Solving constraints after reviewing summarizer...",
                        tc.id,
                    )
                    print_step(f"TestCase #{tc.id}: Re-solving after review...")

                (
                    tc.is_satisfiable,
                    tc.exec_code,
                    usage_details,
                    msg_thread,
                ) = solve(
                    _format_exec_code(tc.src_exec_code),
                    tc.target_path_constraint,
                )

                solver_msg_thread = msg_thread.copy()

                if tc.is_satisfiable:
                    logger.info("Constraints were satisfiable.")
                    # Constraints were satisfiable, use the new execution
                    if tc.current_state == TestcaseState.SOLVE:
                        next_state = TestcaseState.EXECUTE
                    elif tc.current_state == TestcaseState.REVIEW_SUMMARY_SOLVE:
                        next_state = TestcaseState.REVIEW_SUMMARY_EXECUTE
                else:
                    # Constraints were unsatisfiable, finish the test case
                    next_state = TestcaseState.FINISHED

            elif tc.current_state == TestcaseState.REVIEW_SOLVER:
                # Review solver's solution
                logger.info("TestCase #{}: Reviewing solver's solution...", tc.id)

                need_adjust, corrected_execution, usage_details, msg_thread = (
                    review_solve(solver_msg_thread)
                )

                if need_adjust:
                    assert corrected_execution is not None
                    # Solver solution was incorrect and has been corrected
                    logger.info(
                        "TestCase #{}: Solver solution was incorrect. Will try corrected solution.",
                        tc.id,
                    )
                    tc.exec_code = corrected_execution
                    next_state = TestcaseState.REVIEW_SOLVER_EXECUTE
                else:
                    # Solver was correct, problem must be in summarizer
                    logger.info(
                        "TestCase #{}: Solver solution was correct. Problem is in summarizer's constraints.",
                        tc.id,
                    )
                    next_state = TestcaseState.REVIEW_SUMMARY

            elif tc.current_state == TestcaseState.REVIEW_SUMMARY:
                logger.info(
                    "TestCase #{}: Reviewing previous path constraint...", tc.id
                )

                (
                    is_summarizer_correct,
                    corrected_path_constraint,
                    usage_details,
                    msg_thread,
                ) = review_summary(
                    summarize_msg_thread,
                    _format_exec_code(tc.exec_code),
                    tc.execution_summary,
                )

                if not is_summarizer_correct:
                    # Summarizer was incorrect, apply corrections
                    logger.info(
                        "TestCase #{}: Previous path constraint was incorrect. Will try with corrected information.",
                        tc.id,
                    )
                    tc.target_path_constraint = corrected_path_constraint

                    next_state = TestcaseState.REVIEW_SUMMARY_SOLVE
                else:
                    # Summarizer was correct, but something else is wrong
                    # This is a fallback case - ideally we should never get here
                    logger.warning(
                        "TestCase #{}: Both summarizer and solver were correct, but target is still not covered.",
                        tc.id,
                    )
                    next_state = TestcaseState.FINISHED

            elif (
                tc.current_state == TestcaseState.EXECUTE
                or tc.current_state == TestcaseState.REVIEW_SOLVER_EXECUTE
                or tc.current_state == TestcaseState.REVIEW_SUMMARY_EXECUTE
            ):
                # Execute the test case and collect trace
                print_step(f"TestCase #{tc.id}: Executing program...")
                result = run_target(tc.exec_code, timeout=exec_timeout)

                if not result["exec_success"]:
                    raise Exception(
                        "TestCase #{}: The python execution should be successful, as guaranteed by previous state. However, the execution failed: {}",
                        tc.id,
                        (result["exec_error"]),
                    )

                tc.execution_trace = result["target_stderr"]
                tc.returncode = result["target_return_code"]

                if result["target_timeout"]:
                    tc.is_hang = True
                    logger.warning(
                        "Test case #{} timed out, exec code:\n{}",
                        tc.id,
                        tc.exec_code,
                    )

                if result["target_crashed"]:
                    tc.is_crash = True
                    tc.crash_info = f"Crash reason: {result['target_crash_reason']}\nStderr:\n{filter_instr_print(result['target_stderr'])}"

                    logger.warning(
                        "Test case #{} crashed, crash info:\n{}",
                        tc.id,
                        tc.crash_info,
                    )
                    logger.debug(
                        "Code of crashed test case #{}:\n{}",
                        tc.id,
                        tc.exec_code,
                    )

                # Process execution trace and check coverage
                if binary_only:
                    assert binary_trace_coverage is not None
                    (
                        tc.newly_covered_lines,
                        tc.is_target_covered,
                        tc.execution_summary,
                    ) = _collect_binary_trace_and_check_coverage(
                        tc.execution_trace,
                        binary_trace_coverage,
                    )
                    tc.new_coverage = True if tc.newly_covered_lines > 0 else False
                    logger.info(
                        "TestCase #{} (binary-only): New runtime events: {}. Objective reached: {}",
                        tc.id,
                        tc.newly_covered_lines,
                        tc.is_target_covered,
                    )
                else:
                    assert tc.target_file_lines[0] is not None
                    (
                        tc.newly_covered_lines,
                        tc.is_target_covered,
                        tc.execution_summary,
                    ) = _collect_trace_and_check_coverage(
                        tc.execution_trace,
                        tc.target_file_lines,
                    )
                    tc.new_coverage = True if tc.newly_covered_lines > 0 else False

                    logger.info(
                        "TestCase #{}: Target lines ({}) covered: {}. Newly covered code lines: {}",
                        tc.id,
                        tc.target_file_lines,
                        tc.is_target_covered,
                        tc.newly_covered_lines,
                    )
                covered_icon = "[green]✓[/]" if tc.is_target_covered else "[red]✗[/]"
                new_cov_label = "events" if binary_only else "lines"
                print_step(
                    f"TestCase #{tc.id}: Target covered {covered_icon}  |  New {new_cov_label}: {tc.newly_covered_lines}"
                )

                # Determine next state
                if tc.is_valuable() or tc.is_crash_or_hang():
                    next_state = TestcaseState.FINISHED
                else:
                    logger.info(
                        "TestCase #{}: Target lines are not covered and code coverage is also not improved.",
                        tc.id,
                    )
                    if tc.current_state == TestcaseState.EXECUTE:
                        if disable_review:
                            next_state = TestcaseState.FINISHED
                        else:
                            logger.info(
                                "TestCase #{}: We need to first review the solver's solution.",
                                tc.id,
                            )
                            next_state = TestcaseState.REVIEW_SOLVER
                    elif tc.current_state == TestcaseState.REVIEW_SOLVER_EXECUTE:
                        logger.info(
                            "TestCase #{}: We need to then review the summarizer's constraints.",
                            tc.id,
                        )
                        next_state = TestcaseState.REVIEW_SUMMARY
                    elif tc.current_state == TestcaseState.REVIEW_SUMMARY_EXECUTE:
                        logger.info(
                            "TestCase #{}: All reviews are done, but still not covered.",
                            tc.id,
                        )
                        next_state = TestcaseState.FINISHED

            tc.add_usage(usage_details)
            tc.add_state(next_state)
            # Explicitly save the test case
            tc.save_to_disk()
    except Exception as e:
        logger.error(f"TestCase #{tc.id}: Exception during solve_and_execute: {e}")
        # Ensure the exception is propagated up
        raise


class TaskExecutor:
    """Class to manage concurrent task execution using a thread pool."""

    def __init__(self, max_workers=10):
        """Initialize the task executor with a thread pool.

        Args:
            max_workers (int): Maximum number of concurrent workers in the thread pool
        """
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.submitted_futures = (
            {}
        )  # Map future to testcase id for better status reporting
        self.futures_lock = threading.Lock()

        # Register cleanup method to be called at program exit
        atexit.register(self.cleanup)

    def cleanup(self):
        """Clean up thread pool resources"""
        logger.debug("Shutting down thread pool executor...")
        self.executor.shutdown(wait=True)
        logger.debug("Thread pool executor has been shut down.")

    def submit_task(self, func: Callable, tc: TestCase, kwargs: dict):
        """Submit a test case to the thread pool for processing

        Args:
            tc (TestCase): The test case to process
            msg_thread (MessageThread): Message thread for the test case
            timeout (int): Execution timeout in seconds

        Returns:
            Future: The future representing the submitted task
        """
        logger.info(f"Submitting TestCase #{tc.id} to thread pool")

        future = self.executor.submit(func, tc, **kwargs)

        with self.futures_lock:
            self.submitted_futures[future] = tc.id

        logger.debug(f"TestCase #{tc.id} successfully submitted")
        return future

    def wait_for_all_tasks(self):
        """Wait for all submitted tasks to complete"""
        with self.futures_lock:
            if not self.submitted_futures:
                logger.info("No tasks to wait for")
                return

            # Create a copy of the mapping to work with
            futures_map = self.submitted_futures.copy()
            self.submitted_futures.clear()

        total_tasks = len(futures_map)
        completed_tasks = 0

        logger.info(
            f"Waiting for {total_tasks} test cases to complete: {list(futures_map.values())}"
        )

        # Process futures as they complete
        for future in concurrent.futures.as_completed(futures_map.keys()):
            tc_id = futures_map[future]
            completed_tasks += 1

            try:
                # Get the result, which will raise any exception that occurred
                future.result()
                logger.info(
                    f"TestCase #{tc_id} completed successfully ({completed_tasks}/{total_tasks})"
                )
            except Exception as e:
                logger.error(f"Exception in TestCase #{tc_id}: {e}")
                # Re-raise the exception
                raise


def run_concolic_execution(
    project_dir,
    out_dir,
    timeout,  # seconds
    test_selection: TestCaseSelection,
    initial_execution_file=None,
    rounds=None,
    resume_in=None,
    plateau_slot=None,
    parallel_num=5,
    binary_only=False,
):
    """Run concolic execution phase using a state machine approach.

    Testcase states:
    - SUMMARIZE: Analyze execution paths and generate constraints
    - SOLVE: Solve constraints to generate new inputs
    - EXECUTE: Execute the NEW test case, collect traces, and check if target lines are covered
    - REVIEW_SOLVER: Review solver solution when target lines aren't covered
    - REVIEW_SOLVER_EXECUTE: Execute corrected solver solution
    - REVIEW_SUMMARY: Problem in summarizer, try a different branch
    - REVIEW_SUMMARY_SOLVE: Solve the new constraints after reviewing summarizer
    - REVIEW_SUMMARY_EXECUTE: Execute after reviewing summarizer and solve the new constraints
    - FINISHED: Finished
    """

    if not initial_execution_file and not resume_in:
        logger.error("Either initial_execution_file or resume_in must be provided")
        sys.exit(1)

    if initial_execution_file and resume_in:
        logger.error(
            "Only one of initial_execution_file or resume_in should be provided"
        )
        sys.exit(1)

    logger.info(
        "Starting concolic execution phase, using testcase selection strategy: {}, parallel_num: {}, plateau_slot: {}, rounds: {}...",
        str(test_selection),
        parallel_num,
        f"{plateau_slot} minutes" if plateau_slot else "None",
        f"{rounds}" if rounds else "None",
    )
    print_step(
        f"Starting concolic execution: {rounds or '∞'} rounds, parallel={parallel_num}, strategy={test_selection.value}"
    )
    update_project_dir(project_dir)

    if binary_only and test_selection == TestCaseSelection.LLM:
        logger.warning(
            "Binary-only mode does not use LLM test-case scheduling. Falling back to DFS selection."
        )
        test_selection = TestCaseSelection.DFS

    if binary_only and parallel_num != 1:
        logger.warning(
            "Binary-only mode currently runs with shared in-memory event coverage. Forcing parallel_num=1."
        )
        parallel_num = 1

    # Initialize TestCaseManager
    crash_cnt = 0
    hang_cnt = 0

    testcase_manager: TestCaseManager = TestCaseManager(out_dir)
    binary_trace_coverage = BinaryTraceCoverage() if binary_only else None
    binary_expected_event_space = (
        _load_binary_expected_event_space() if binary_only else None
    )

    # Initialize task executor for concurrent processing
    task_executor = TaskExecutor(max_workers=parallel_num)

    time_bias = 0

    if initial_execution_file:
        # Initialize execution variables
        print_step("Running initial test case...")
        exec_code = open(initial_execution_file).read()
        result = run_target(exec_code, timeout=timeout)

        if not result["exec_success"]:
            logger.error("Initial test case execution failed: {}", result["exec_error"])
            sys.exit(1)

        if result["target_timeout"]:
            logger.error(
                "Initial test case execution timed out. Please increase the timeout limit (current timeout: {} seconds).".format(
                    timeout
                )
            )
            sys.exit(1)

        # Collect initial execution trace
        execution_trace = result["target_stderr"]

        if binary_only:
            assert binary_trace_coverage is not None
            new_covered_lines, is_target_lines_covered, execution_summary = (
                _collect_binary_trace_and_check_coverage(
                    execution_trace,
                    binary_trace_coverage,
                )
            )
        else:
            new_covered_lines, is_target_lines_covered, execution_summary = (
                _collect_trace_and_check_coverage(execution_trace, (None, (None, None)))
            )

        if new_covered_lines == 0:
            logger.error(
                "Initial test case execution did not cover any new events/lines, please check"
            )
            sys.exit(1)

        assert is_target_lines_covered
        if binary_only:
            print_step(f"Initial test case: {new_covered_lines} runtime events covered")
        else:
            print_step(f"Initial test case: {new_covered_lines} lines covered")
        # Create initial test case

        testcase_manager.add_initial_testcase(
            exec_code, execution_trace, execution_summary, new_covered_lines
        )
    else:

        testcase_manager.load_testcases(resume_in)
        testcase_manager.save_all_testcases()  # "crashes_or_hangs" directory will also be synchronized

        # First, ensure the testcase is runable, using the first (initial) testcase as an example
        result = run_target(testcase_manager.get_testcase(0).exec_code, timeout=timeout)

        if not result["exec_success"]:
            logger.error("Initial test case execution failed: {}", result["exec_error"])
            sys.exit(1)

        if result["target_timeout"]:
            logger.error(
                "Initial test case execution timed out. Please increase the timeout limit (current timeout: {} seconds).".format(
                    timeout
                )
            )
            sys.exit(1)

        if result["target_stderr"] != testcase_manager.get_testcase(0).execution_trace:
            logger.error(
                "The execution trace of the initial testcase has changed. There may be some issues with the execution environment. New execution trace:\n{}",
                result["target_stderr"],
            )
            sys.exit(1)

        # Then, using the execution trace of each testcase to replay the "coverage"
        for id, testcase in testcase_manager.test_cases.items():
            if testcase is None:
                logger.warning(
                    f"Test case #{id} not found, skipping coverage replay..."
                )
            elif testcase.execution_trace is not None:
                if binary_only:
                    assert binary_trace_coverage is not None
                    _collect_binary_trace_and_check_coverage(
                        testcase.execution_trace,
                        binary_trace_coverage,
                    )
                else:
                    _collect_trace_and_check_coverage(
                        testcase.execution_trace, (None, (None, None))
                    )

        time_bias = testcase_manager.get_max_time_taken()
        crash_cnt, hang_cnt = testcase_manager.get_crash_and_hang_count()

        logger.info(
            "Have loaded {} test cases, with time bias {} seconds, original crashes: {}, hangs: {}.",
            testcase_manager.next_testcase_id,
            time_bias,
            crash_cnt,
            hang_cnt,
        )

    update_ce_start_time(time.time() - time_bias)
    last_new_coverage_time = time.time() - time_bias

    under_gen_tc_msg_thread: dict[int, MessageThread] = (
        {}
    )  # if one testcase is finished, it should be removed

    under_gen_tcs: list[TestCase] = []

    round_cnt = 1
    cur_ce_state = ConcolicExecutionState.SELECT

    if test_selection == TestCaseSelection.DFS:
        priority_queue = deque()
        priority_queue.append(0)
        last_added_max_id = 0
    elif test_selection == TestCaseSelection.LLM:
        scheduler = TestCaseScheduler()

    while round_cnt <= (rounds or float("inf")):
        logger.info(
            "====== Iteration {}: Current state: {} ======",
            round_cnt,
            str(cur_ce_state),
        )
        set_concolic_execution_state(cur_ce_state)
        match cur_ce_state:
            case ConcolicExecutionState.SELECT:
                print_phase("SELECT", round_num=round_cnt)
                selection_usage_details = {"TOTAL": Usage()}
                # select a test case
                if test_selection == TestCaseSelection.LLM:
                    provided_tc_info: dict[int, str] = (
                        testcase_manager.get_all_scheduling_information()
                    )
                    logger.info(
                        f"Selecting test case using LLM STRATEGY from list: {list(provided_tc_info.keys())}"
                    )
                    if len(list(provided_tc_info.keys())) == 0:
                        raise Exception("No test cases are available for scheduling.")
                    elif len(list(provided_tc_info.keys())) == 1:
                        selected_testcase_id = list(provided_tc_info.keys())[0]
                        if testcase_manager.next_testcase_id != 1:
                            raise Exception(
                                "Only one test case is selected to be scheduled."
                            )
                    else:
                        selected_testcase_id, selection_usage_details, msg_thread = (
                            scheduler.schedule(provided_tc_info)
                        )

                elif test_selection == TestCaseSelection.RANDOM:
                    valuable_tc_ids = [
                        tc_id
                        for tc_id, tc in testcase_manager.test_cases.items()
                        if tc is not None and tc.is_valuable()
                    ]
                    assert len(valuable_tc_ids) > 0
                    selected_testcase_id = random.choice(valuable_tc_ids)
                    logger.info(
                        f"Selecting test case using RANDOM STRATEGY from list: {valuable_tc_ids}"
                    )

                elif test_selection == TestCaseSelection.DFS:
                    for tc_id in range(
                        last_added_max_id + 1, testcase_manager.next_testcase_id
                    ):
                        tc = testcase_manager.get_testcase(tc_id)
                        if tc is not None and tc.is_valuable():
                            priority_queue.append(tc_id)
                    last_added_max_id = testcase_manager.next_testcase_id - 1

                    logger.info(
                        f"Selecting test case using DFS from list: {priority_queue if len(priority_queue) > 0 else 'empty (will use the first testcase)'}"
                    )
                    if len(priority_queue) > 0:
                        selected_testcase_id = priority_queue.pop()
                    else:
                        # start again?
                        selected_testcase_id = 0

                src_testcase = testcase_manager.get_testcase(selected_testcase_id)

                logger.info(
                    "Using test case #{} as the base test case.", src_testcase.id
                )
                print_step(f"Using test case #{src_testcase.id} as the base test case")

                # Refresh execution summary but DO NOT update the original execution summary of the testcase (used as backup)
                if binary_only:
                    assert binary_trace_coverage is not None
                    _, _, latest_src_exec_summary = _collect_binary_trace_and_check_coverage(
                        src_testcase.execution_trace,
                        binary_trace_coverage,
                        add_coverage=False,
                    )
                else:
                    _, _, latest_src_exec_summary = _collect_trace_and_check_coverage(
                        src_testcase.execution_trace,
                        (None, (None, None)),
                        add_coverage=False,
                    )

                # state transition
                cur_ce_state = ConcolicExecutionState.SUMMARIZE
            case ConcolicExecutionState.SUMMARIZE:
                # summarize the test case
                print_phase("SUMMARIZE")
                branches_yielded_cnt = 0

                if binary_only:
                    assert binary_trace_coverage is not None
                    (
                        target_branch,
                        justification,
                        target_file_lines,
                        target_path_constraint,
                    ) = _build_binary_only_target(src_testcase, binary_trace_coverage)

                    logger.info(
                        "Processing binary-only objective for testcase #{}: {}",
                        src_testcase.id,
                        target_branch,
                    )

                    new_testcase: TestCase = testcase_manager.create_new_testcase(
                        src_testcase.id,
                        latest_src_exec_summary,
                        target_branch,
                        justification,
                        target_file_lines,
                        None,
                        target_path_constraint,
                    )

                    # assign selection usage to the first generated testcase in this round
                    new_testcase.add_usage(
                        usage=selection_usage_details,
                        state=TestcaseState.SELECT,
                    )
                    branches_yielded_cnt = 1
                    new_testcase.save_to_disk()

                    under_gen_tc_msg_thread[new_testcase.id] = MessageThread()
                    under_gen_tcs.append(new_testcase)

                    task_executor.submit_task(
                        solve_and_execute,
                        under_gen_tcs[-1],
                        {
                            "summarize_msg_thread": under_gen_tc_msg_thread[
                                under_gen_tcs[-1].id
                            ],
                            "exec_timeout": timeout,
                            "disable_review": True,
                            "binary_only": True,
                            "binary_trace_coverage": binary_trace_coverage,
                        },
                    )
                    logger.info(
                        "Submitted binary-only testcase #{} for solving and executing",
                        new_testcase.id,
                    )
                    print_step(
                        f"Submitted test case #{new_testcase.id} → solve & execute"
                    )

                    cur_ce_state = ConcolicExecutionState.SOLVE_AND_EXECUTE
                    continue

                func_call_chain_list = [
                    (file, func)
                    for file, func, _ in trace_compress(src_testcase.execution_trace)
                ]

                # compress the repeating sequences
                compressed_func_call_chain_list = compress_repeating_sequences(
                    func_call_chain_list
                )

                formatted_func_call_chain_list = []
                for item_count_pair in compressed_func_call_chain_list:
                    item, count = item_count_pair
                    if isinstance(item, tuple):
                        file_name, func_name = item
                        formatted_func_call_chain_list.append(
                            f"[{file_name}]({func_name}){f'*{count}' if count > 1 else ''}"
                        )
                    elif isinstance(item, list):
                        assert len(item) >= 2
                        assert count > 1
                        combined_file_func_list = []
                        for file_name, func_name in item:
                            combined_file_func_list.append(
                                f"[{file_name}]({func_name})"
                            )
                        formatted_func_call_chain_list.append(
                            f"[{'=>'.join(combined_file_func_list)}]*{count}"
                        )
                    else:
                        raise Exception(
                            f"Unexpected item format in compressed_func_call_chain_list: {item}"
                        )

                if len(formatted_func_call_chain_list) > 500:
                    formatted_func_call_chain_list = formatted_func_call_chain_list[
                        :500
                    ] + ["...(truncated)"]

                function_call_chain = " => ".join(formatted_func_call_chain_list)

                # Get generator and process generated branches
                branches_generator = summarize(
                    _format_exec_code(src_testcase.exec_code),
                    latest_src_exec_summary,
                    function_call_chain,
                    testcase_manager.get_already_selected_branch_but_not_reached(
                        src_testcase.id
                    ),
                    parallel_num,
                )

                # Process each generated branch with its usage and message thread
                for branch_data in branches_generator:
                    # Unpack the yielded data
                    (
                        target_branch,
                        justification,
                        target_file_lines,
                        target_path_constraint,
                        usage_details,
                        current_msg_thread,
                    ) = branch_data

                    logger.info(f"Processing new branch: {target_branch}...")

                    target_lines_content = _get_file_lines(
                        target_file_lines[0], target_file_lines[1]
                    )

                    if target_lines_content:
                        assert (
                            len(target_lines_content.split("\n"))
                            == target_file_lines[1][1] - target_file_lines[1][0] + 1
                        )

                    # Create a new test case for each branch
                    new_testcase: TestCase = testcase_manager.create_new_testcase(
                        src_testcase.id,
                        latest_src_exec_summary,
                        target_branch,
                        justification,
                        target_file_lines,
                        target_lines_content,
                        target_path_constraint,
                    )

                    if branches_yielded_cnt == 0:
                        new_testcase.add_usage(
                            usage=selection_usage_details,
                            state=TestcaseState.SELECT,
                        )  # assign selection usage to the FIRST testcase
                    branches_yielded_cnt += 1
                    if branches_yielded_cnt > parallel_num:
                        logger.error(
                            f"Reached the maximum number ({parallel_num}) of branches to explore, but there are still branches to be processed...",
                        )
                    # Add current usage information
                    new_testcase.add_usage(
                        usage_details
                    )  # newly-generated testcase is in SUMMARIZE state
                    # Explicitly save the test case
                    new_testcase.save_to_disk()

                    # Store the message thread for this test case
                    under_gen_tc_msg_thread[new_testcase.id] = current_msg_thread.copy()
                    under_gen_tcs.append(new_testcase)

                    # use concurrent processing, submit tasks to the thread pool
                    task_executor.submit_task(
                        solve_and_execute,
                        under_gen_tcs[-1],
                        {
                            "summarize_msg_thread": under_gen_tc_msg_thread[
                                under_gen_tcs[-1].id
                            ],
                            "exec_timeout": timeout,
                            "disable_review": True,
                            "binary_only": False,
                            "binary_trace_coverage": None,
                        },
                    )

                    logger.info(
                        f"Submitted new test case #{new_testcase.id} for solving and executing"
                    )
                    print_step(
                        f"Submitted test case #{new_testcase.id} → solve & execute"
                    )

                # If no branches were generated, the should be prompt exceeded the token limit
                if branches_yielded_cnt == 0:
                    logger.error(
                        "No branches were selected and summarized during summarize process, maybe the prompt is too long"
                    )
                    cur_ce_state = ConcolicExecutionState.ITERATION_FINISHED
                else:
                    # state transition
                    cur_ce_state = ConcolicExecutionState.SOLVE_AND_EXECUTE
            case ConcolicExecutionState.SOLVE_AND_EXECUTE:
                # wait for all submitted test cases to finish
                print_phase("SOLVE & EXECUTE (waiting for all tasks to complete)")
                logger.info(
                    f"Waiting for all submitted test cases to finish... (Round #{round_cnt})"
                )
                task_executor.wait_for_all_tasks()
                logger.info(
                    f"All test cases in round #{round_cnt} completed successfully"
                )

                # state transition
                cur_ce_state = ConcolicExecutionState.ITERATION_FINISHED
            case ConcolicExecutionState.ITERATION_FINISHED:
                # this iteration is finished, do some clean up and go to next iteration

                # update the successful generation count of the src testcase
                print_phase("RESULTS")
                for tc in under_gen_tcs:
                    if tc.is_valuable():
                        src_testcase.successful_generation_cnt += 1
                src_testcase.save_to_disk()

                # update the plateau time
                last_new_coverage_time = (
                    time.time()
                    if any(tc.new_coverage for tc in under_gen_tcs)
                    else last_new_coverage_time
                )

                crash_cnt += sum(1 for tc in under_gen_tcs if tc.is_crash)
                hang_cnt += sum(1 for tc in under_gen_tcs if tc.is_hang)

                stats_msg = testcase_manager.get_statistics()[1]
                logger.info(
                    f"{stats_msg}\nCrash count: {crash_cnt}, hang count: {hang_cnt}"
                )
                print_step(stats_msg)

                if binary_only and binary_trace_coverage is not None:
                    event_space_msg = _build_binary_event_space_summary(
                        binary_trace_coverage,
                        binary_expected_event_space,
                    )
                    logger.info(event_space_msg)
                    print_step(event_space_msg)

                    input_summary_msg = _build_round_input_summary(under_gen_tcs)
                    logger.info(input_summary_msg)
                    print_step(input_summary_msg)

                if crash_cnt > 0 or hang_cnt > 0:
                    print_step(f"Crashes: {crash_cnt}, Hangs: {hang_cnt}")

                under_gen_tc_msg_thread.clear()
                under_gen_tcs.clear()

                # state transition
                cur_ce_state = ConcolicExecutionState.SELECT
                round_cnt += 1

                if rounds is None and plateau_slot is not None:
                    if time.time() - last_new_coverage_time > plateau_slot * 60:
                        logger.info(
                            "The code coverage has not improved for {} minutes, stopping the concolic execution...",
                            plateau_slot,
                        )
                        break


def setup_run_parser(parser):
    """Setup the run command parser."""
    run_parser = parser.add_parser(
        "run", help="run the concolic execution on an existing instrumented project"
    )
    run_parser.add_argument(
        "--project_dir",
        type=str,
        help="(instrumented) project directory",
        required=True,
    )
    run_parser.add_argument(
        "--execution",
        type=str,
        help="initial execution file path. Note: In this file, you can use (1) file path relative to the project directory, or (2) absolute file path.",
        required=False,
        default=None,
    )
    run_parser.add_argument(
        "--out",
        type=str,
        help="output directory",
        required=True,
    )
    run_parser.add_argument(
        "--resume_in",
        type=str,
        help="the original output directory to resume from",
        required=False,
        default=None,
    )
    run_parser.add_argument(
        "--rounds",
        type=int,
        help="number of rounds",
        required=False,
        default=None,
    )
    run_parser.add_argument(
        "--selection",
        type=str,
        help="test case selection method",
        required=False,
        default=TestCaseSelection.DFS,
        choices=[e.value for e in TestCaseSelection],
    )
    run_parser.add_argument(
        "--timeout",
        type=int,
        help="program execution timeout in seconds",
        required=False,
        default=3,
    )
    run_parser.add_argument(
        "--plateau_slot",
        type=int,
        help="plateau slot in minutes. The concolic execution will stop if the code coverage does not improve for this period of time (only used when --rounds is not specified, otherwise, the concolic execution will stop after running --rounds test cases).",
        required=False,
        default=None,
    )
    run_parser.add_argument(
        "--parallel_num",
        type=int,
        help="number of parallel testcase generation",
        required=False,
        default=5,
    )
    run_parser.add_argument(
        "--binary_only",
        action="store_true",
        help="enable source-independent binary-only mode (uses runtime trace-event coverage as objective)",
    )
    return run_parser
