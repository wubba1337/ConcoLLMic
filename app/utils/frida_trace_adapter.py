"""
Helpers for adapting Frida trace lines to ACE-compatible trace format.
"""

import json
import re
from collections import defaultdict
from typing import Any

from app.agents.common import FILE_TRACE_PATTERN


def load_frida_trace_map(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _split_frida_function_token(func_token: str) -> tuple[str, str]:
    """
    Returns (base_symbol, channel):
      channel in {"function", "bb", "br"}
    """
    if func_token.endswith("__bb"):
        return func_token[: -len("__bb")], "bb"
    if func_token.endswith("__br"):
        return func_token[: -len("__br")], "br"
    return func_token, "function"


def adapt_frida_trace_to_ace(
    frida_trace: str, trace_map: dict[str, Any]
) -> tuple[str, dict[str, int]]:
    """
    Convert Frida trace lines to ACE file-trace lines:
      [<ace_file>] enter|exit <ace_function> <ace_block_id>
    """
    symbol_mappings: dict[str, dict[str, Any]] = trace_map.get("symbol_mappings", {})
    fn_id_mapping: dict[str, int] = trace_map.get(
        "frida_function_id_to_ace_block_id", {}
    )

    # Runtime mapping for branch/bb IDs (which are discovered dynamically).
    runtime_id_to_ace: dict[str, dict[str, int]] = defaultdict(dict)
    runtime_next_idx: dict[str, int] = defaultdict(int)

    output_lines: list[str] = []
    stats = {
        "mapped_lines": 0,
        "skipped_non_trace_lines": 0,
        "skipped_unknown_symbol": 0,
        "skipped_unmapped_symbol": 0,
        "skipped_missing_blocks": 0,
    }

    for raw_line in frida_trace.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        matches = re.findall(FILE_TRACE_PATTERN, line)
        if not matches:
            stats["skipped_non_trace_lines"] += 1
            continue

        for _file_tag, action, frida_func_token, frida_id_str in matches:
            base_symbol, channel = _split_frida_function_token(frida_func_token)
            symbol_meta = symbol_mappings.get(base_symbol)
            if symbol_meta is None:
                stats["skipped_unknown_symbol"] += 1
                continue
            if not symbol_meta.get("ace_mapping_available", False):
                stats["skipped_unmapped_symbol"] += 1
                continue

            ace_file = symbol_meta.get("ace_file")
            ace_function = symbol_meta.get("ace_function")
            ace_block_ids = symbol_meta.get("ace_block_ids", [])
            if not ace_file or not ace_function or not ace_block_ids:
                stats["skipped_missing_blocks"] += 1
                continue

            frida_id = int(frida_id_str)
            if channel == "function":
                ace_block_id = fn_id_mapping.get(str(frida_id), ace_block_ids[0])
            else:
                symbol_runtime = runtime_id_to_ace[base_symbol]
                ace_block_id = symbol_runtime.get(str(frida_id))
                if ace_block_id is None:
                    idx = runtime_next_idx[base_symbol] % len(ace_block_ids)
                    ace_block_id = ace_block_ids[idx]
                    symbol_runtime[str(frida_id)] = ace_block_id
                    runtime_next_idx[base_symbol] += 1

            output_lines.append(f"[{ace_file}] {action} {ace_function} {ace_block_id}")
            stats["mapped_lines"] += 1

    return "\n".join(output_lines), stats

