def execute_program(timeout: int) -> tuple[str, int]:
    import os
    import re
    import signal
    import subprocess
    import sys

    repo_root = os.environ.get(
        "CONCOLLMIC_REPO_ROOT",
        os.path.abspath(os.path.join(os.getcwd(), "..", "..")),
    )
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from app.utils.frida_trace_adapter import adapt_frida_trace_to_ace, load_frida_trace_map

    target_binary = os.environ.get("FRIDA_TARGET_BINARY", "../bin/count_plain")
    frida_script = os.environ.get("FRIDA_SCRIPT_PATH", "./frida_hooks.js")
    frida_trace_map = os.environ.get("FRIDA_TRACE_MAP_PATH", "./frida_trace_map.json")
    # ace: force map-based adaptation, raw: emit raw frida trace lines, auto: ace then raw fallback
    frida_trace_mode = os.environ.get("FRIDA_TRACE_MODE", "auto").strip().lower()
    frida_cmd = os.environ.get("FRIDA_BIN", "frida")
    frida_capture_timeout = int(
        os.environ.get("FRIDA_CAPTURE_TIMEOUT", str(max(1, min(timeout, 5))))
    )
    frida_max_retries = int(os.environ.get("FRIDA_MAX_RETRIES", "3"))

    # Initial seed input for count.c
    arg1 = "1.0"
    arg2 = "1.00001"

    def _to_text(value) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    def _extract_raw_trace_lines(output: str) -> str:
        # Keep only exact ACE-style trace lines: [token] enter/exit symbol id
        # so REPL/runtime noise does not pollute coverage parsing.
        pattern = re.compile(r"^\[[^\]\n]+\]\s+(enter|exit)\s+\S+\s+\d+\s*$")
        lines = [line.strip() for line in output.splitlines() if pattern.fullmatch(line.strip())]
        return "\n".join(lines)

    def _render_trace(frida_output: str) -> str:
        if frida_trace_mode == "raw":
            return _extract_raw_trace_lines(frida_output)

        if frida_trace_mode not in ("auto", "ace"):
            raise RuntimeError(
                f"Unsupported FRIDA_TRACE_MODE='{frida_trace_mode}'. Use one of: auto, ace, raw."
            )

        try:
            trace_map = load_frida_trace_map(frida_trace_map)
            adapted_trace, stats = adapt_frida_trace_to_ace(frida_output, trace_map)
            if adapted_trace.strip():
                return adapted_trace
            if frida_trace_mode == "ace":
                raise RuntimeError(
                    "Frida trace adaptation produced no ACE trace lines. "
                    f"Stats: {stats}. "
                    "Check FRIDA_TRACE_MAP_PATH and frida target symbol mappings."
                )
        except Exception:
            if frida_trace_mode == "ace":
                raise

        # auto mode fallback
        return _extract_raw_trace_lines(frida_output)

    last_output = ""
    for _attempt in range(max(1, frida_max_retries)):
        try:
            result = subprocess.run(
                [
                    frida_cmd,
                    "-q",
                    "-t",
                    str(frida_capture_timeout),
                    "-f",
                    target_binary,
                    "-l",
                    frida_script,
                    "--",
                    arg1,
                    arg2,
                ],
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=max(timeout, frida_capture_timeout + 2),
            )

            frida_output = _to_text(result.stdout) + "\n" + _to_text(result.stderr)
            last_output = frida_output
            rendered_trace = _render_trace(frida_output)
            if rendered_trace.strip():
                return rendered_trace, 0
        except subprocess.TimeoutExpired as e:
            partial = _to_text(e.stdout) + "\n" + _to_text(e.stderr)
            last_output = partial
            try:
                rendered_trace = _render_trace(partial)
                if rendered_trace.strip():
                    return rendered_trace, 0
            except Exception:
                pass
            continue

    raise RuntimeError(
        "Frida trace processing produced no trace lines after "
        f"{max(1, frida_max_retries)} attempt(s). "
        "Check FRIDA_SCRIPT_PATH, FRIDA_TARGET_BINARY, and FRIDA_TRACE_MODE.\n"
        f"Last Frida output:\n{last_output}"
    )
