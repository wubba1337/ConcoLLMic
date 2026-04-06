"""
Code instrumentation command for ACE.
"""

import atexit
import concurrent.futures
import datetime
import json
import os
import re
import subprocess
import threading
from collections import deque
from collections.abc import Callable
from typing import Any

import yaml
from loguru import logger

from app.agents.common import FILE_TRACE_PATTERN
from app.agents.agent_instrumentation import InstrumentationAgent
from app.utils.utils import (
    compress_paths,
    detect_language,
    get_comment_token,
    list_all_files,
)

INSTRUMENTATION_INFO_FILE = ".instrument_info"
FRIDA_TRACE_MAP_FILE = "frida_trace_map.json"


def _run_capture(cmd: list[str]) -> str:
    result = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout


def _normalize_symbol_name(name: str) -> str:
    normalized = name.strip()
    if not normalized:
        return ""
    if "@" in normalized:
        normalized = normalized.split("@")[0]
    return normalized


def _extract_function_symbols(binary_path: str) -> list[str]:
    """
    Extract function symbol names from an ELF binary.
    Prefer readelf output; fall back to nm if needed.
    """
    symbols: list[str] = []
    seen: set[str] = set()

    parse_errors = []

    # 1) readelf -Ws
    try:
        output = _run_capture(["readelf", "-Ws", binary_path])
        for line in output.splitlines():
            parts = line.split()
            if len(parts) < 8:
                continue
            # Num: Value Size Type Bind Vis Ndx Name
            sym_type = parts[3]
            ndx = parts[6]
            if sym_type != "FUNC" or ndx == "UND":
                continue
            name = _normalize_symbol_name(parts[7])
            if not name or name in seen:
                continue
            seen.add(name)
            symbols.append(name)
        if symbols:
            return symbols
    except Exception as e:
        parse_errors.append(f"readelf failed: {e}")

    # 2) nm -D --defined-only
    try:
        output = _run_capture(["nm", "-D", "--defined-only", binary_path])
        for line in output.splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            name = _normalize_symbol_name(parts[-1])
            if not name or name in seen:
                continue
            seen.add(name)
            symbols.append(name)
        if symbols:
            return symbols
    except Exception as e:
        parse_errors.append(f"nm -D failed: {e}")

    # 3) nm --defined-only
    try:
        output = _run_capture(["nm", "--defined-only", binary_path])
        for line in output.splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            name = _normalize_symbol_name(parts[-1])
            if not name or name in seen:
                continue
            seen.add(name)
            symbols.append(name)
        if symbols:
            return symbols
    except Exception as e:
        parse_errors.append(f"nm failed: {e}")

    raise RuntimeError(
        f"Failed to extract symbols from {binary_path}. Details: {parse_errors}"
    )


def _demangle_symbol_name(symbol: str) -> str:
    symbol = _normalize_symbol_name(symbol)
    if not symbol:
        return symbol
    try:
        demangled = _run_capture(["c++filt", symbol]).strip()
        return demangled or symbol
    except Exception:
        return symbol


def _extract_function_basename(function_name: str) -> str:
    """
    Normalize a symbol / demangled signature to a short function basename.
    Examples:
      "_Z17validate_bracketsPKc" -> "_Z17validate_bracketsPKc" (fallback path)
      "validate_brackets(char const*)" -> "validate_brackets"
      "ns::Foo::bar(int)" -> "bar"
    """
    if not function_name:
        return ""
    short = function_name.strip()
    if "(" in short:
        short = short.split("(", 1)[0].strip()
    if "::" in short:
        short = short.rsplit("::", 1)[-1]
    return short


def _collect_instrumented_function_blocks(project_dir: str) -> dict[str, dict[str, Any]]:
    """
    Scan instrumented source files and extract ACE function/block metadata.

    Returns:
      {
        "<file_tag>::<func_name>": {
          "file": "<file_tag>",
          "function": "<func_name>",
          "block_ids": [1,2,...],
          "block_count": N
        },
        ...
      }
    """
    result: dict[str, dict[str, Any]] = {}

    for file_path in list_all_files(project_dir, recursive=True):
        rel_path = os.path.relpath(file_path, project_dir)
        try:
            with open(file_path, encoding="utf-8") as f:
                lines = f.readlines()
        except Exception:
            # Ignore binary/unreadable files when building trace map.
            continue

        for raw_line in lines:
            matches = re.findall(FILE_TRACE_PATTERN, raw_line.strip())
            if not matches:
                continue
            for file_tag, action, func_name, block_id in matches:
                if action != "enter":
                    continue
                key = f"{file_tag}::{func_name}"
                if key not in result:
                    result[key] = {
                        "file": file_tag,
                        "function": func_name,
                        "block_ids": set(),
                        "source_file_relpath": rel_path,
                    }
                result[key]["block_ids"].add(int(block_id))

    # Normalize sets to sorted lists.
    normalized: dict[str, dict[str, Any]] = {}
    for key, metadata in result.items():
        block_ids = sorted(metadata["block_ids"])
        normalized[key] = {
            "file": metadata["file"],
            "function": metadata["function"],
            "block_ids": block_ids,
            "block_count": len(block_ids),
            "source_file_relpath": metadata["source_file_relpath"],
        }
    return normalized


def _build_frida_trace_map(
    *,
    binary_path: str,
    module_name: str,
    frida_mode: str,
    symbols: list[str],
    project_dir: str | None,
    expected_event_space: dict[str, int] | None = None,
    hook_specs: list[dict[str, Any]] | None = None,
    image_base: int | None = None,
) -> dict[str, Any]:
    """
    Build a Frida->ACE trace mapping artifact.
    This artifact is consumed by the Frida adapter harness in Step 2.
    """
    function_blocks = (
        _collect_instrumented_function_blocks(project_dir) if project_dir else {}
    )

    # Index instrumented functions by basename for symbol matching.
    basename_to_candidates: dict[str, list[dict[str, Any]]] = {}
    for metadata in function_blocks.values():
        basename = _extract_function_basename(metadata["function"])
        basename_to_candidates.setdefault(basename, []).append(metadata)

    symbol_mappings: dict[str, dict[str, Any]] = {}
    frida_function_id_to_ace_block: dict[str, int] = {}

    for idx, sym in enumerate(symbols, start=1):
        demangled = _demangle_symbol_name(sym)
        candidate_names = [
            _extract_function_basename(demangled),
            _extract_function_basename(sym),
            _normalize_symbol_name(sym),
        ]

        chosen: dict[str, Any] | None = None
        for cname in candidate_names:
            if not cname:
                continue
            candidates = basename_to_candidates.get(cname, [])
            if not candidates:
                continue
            # Deterministic tie-break: prefer more ACE blocks, then lexicographically.
            chosen = sorted(
                candidates,
                key=lambda c: (-c["block_count"], c["file"], c["function"]),
            )[0]
            break

        mapping_entry: dict[str, Any] = {
            "frida_symbol": sym,
            "demangled_symbol": demangled,
            "frida_function_id": idx,
            "ace_file": None,
            "ace_function": None,
            "ace_block_ids": [],
            "ace_mapping_available": False,
        }

        if chosen:
            mapping_entry.update(
                {
                    "ace_file": chosen["file"],
                    "ace_function": chosen["function"],
                    "ace_block_ids": chosen["block_ids"],
                    "ace_mapping_available": len(chosen["block_ids"]) > 0,
                }
            )
            if chosen["block_ids"]:
                # Function mode uses deterministic frida function IDs (1..N).
                # Map them directly to the first ACE block ID.
                frida_function_id_to_ace_block[str(idx)] = chosen["block_ids"][0]

        symbol_mappings[sym] = mapping_entry

    return {
        "version": 1,
        "generated_at": datetime.datetime.now().isoformat(),
        "binary_path": os.path.normpath(binary_path),
        "module_name": module_name,
        "frida_mode": frida_mode,
        "symbols": symbols,
        "symbol_mappings": symbol_mappings,
        "frida_function_id_to_ace_block_id": frida_function_id_to_ace_block,
        "runtime_branch_mapping_policy": (
            None
            if frida_mode == "static_offset"
            else "first_seen_round_robin_within_symbol"
        ),
        "static_id_mapping_policy": (
            "fixed_id_per_static_offset" if frida_mode == "static_offset" else None
        ),
        "expected_event_space": expected_event_space or {},
        "hook_spec_count": len(hook_specs or []),
        "image_base": f"0x{image_base:x}" if image_base is not None else None,
        "project_dir": os.path.normpath(project_dir) if project_dir else None,
        "ace_function_count": len(function_blocks),
    }


def _save_frida_trace_map(trace_map: dict[str, Any], out_path: str) -> None:
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(trace_map, f, indent=2, ensure_ascii=False)

def _get_elf_image_base(binary_path: str) -> int:
    """
    Return ELF image base (minimum LOAD virtual address), used to convert
    static instruction addresses to module-relative offsets.
    """
    try:
        output = _run_capture(["readelf", "-W", "-l", binary_path])
    except Exception as e:
        logger.warning("Failed to read ELF program headers for {}: {}", binary_path, e)
        return 0

    load_vaddrs: list[int] = []
    for line in output.splitlines():
        parts = line.split()
        if len(parts) < 3 or parts[0] != "LOAD":
            continue
        try:
            load_vaddrs.append(int(parts[2], 16))
        except Exception:
            continue
    return min(load_vaddrs) if load_vaddrs else 0


def _is_branch_mnemonic(mnemonic: str) -> bool:
    m = mnemonic.lower()
    return (
        m.startswith("j")  # x86 jcc / jmp
        or m == "b"
        or m.startswith("b.")
        or m.startswith("cb")
        or m.startswith("tb")
        or m in {"br", "blr", "bx"}
    )


def _is_conditional_branch_mnemonic(mnemonic: str) -> bool:
    m = mnemonic.lower()
    if m.startswith("j"):
        return m not in {"jmp", "jmpq"}
    if m == "b":
        return False
    if m.startswith("b."):
        return True
    if m.startswith("cb") or m.startswith("tb"):
        return True
    if m in {"br", "blr", "bx"}:
        return False
    return False


def _is_epilogue_mnemonic(mnemonic: str) -> bool:
    """
    Return True for instruction mnemonics that are typically function epilogues
    or terminal returns and are poor candidates for inline interception.
    """
    m = mnemonic.lower().strip()
    return m in {
        "ret",
        "retq",
        "iret",
        "iretq",
        "leave",
        "leaveq",
        "sysret",
        "sysexit",
    }


def _extract_direct_target_address(operand: str) -> int | None:
    """
    Best-effort parser for direct branch target addresses from objdump operands.
    """
    operand = operand.strip()
    if not operand:
        return None

    # Common pattern: "11d4 <label>" or "0x11d4 <label>"
    match = re.search(r"(0x[0-9a-fA-F]+|[0-9a-fA-F]+)\s*(?:<[^>]*>)?$", operand)
    if not match:
        return None
    raw = match.group(1)
    try:
        return int(raw, 16)
    except Exception:
        return None


def _disassemble_symbol_instructions(binary_path: str, symbol: str) -> list[dict[str, Any]]:
    """
    Disassemble a symbol and return normalized instruction records:
      [{"address": int, "mnemonic": str, "operand": str}, ...]
    """
    try:
        output = _run_capture(
            [
                "objdump",
                "-d",
                "--no-show-raw-insn",
                f"--disassemble={symbol}",
                binary_path,
            ]
        )
    except Exception as e:
        logger.warning("objdump failed for symbol {}: {}", symbol, e)
        return []

    instructions: list[dict[str, Any]] = []
    insn_pattern = re.compile(
        r"^\s*([0-9a-fA-F]+):\s+([a-zA-Z][a-zA-Z0-9._]*)\s*(.*)$"
    )
    for line in output.splitlines():
        match = insn_pattern.match(line)
        if not match:
            continue
        try:
            address = int(match.group(1), 16)
        except Exception:
            continue
        mnemonic = match.group(2).strip()
        operand = match.group(3).strip()
        instructions.append(
            {
                "address": address,
                "mnemonic": mnemonic,
                "operand": operand,
            }
        )
    return instructions


def _compute_static_offset_hook_specs(
    binary_path: str,
    symbols: list[str],
) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    """
    Build static-offset hook specs with deterministic IDs for:
      - function entry
      - basic-block leaders
      - branch instructions
    """
    image_base = _get_elf_image_base(binary_path)
    hook_specs: list[dict[str, Any]] = []

    next_function_id = 1
    next_bb_id = 100000
    next_branch_id = 200000

    min_hook_instruction_bytes = 2

    for symbol in symbols:
        insns = _disassemble_symbol_instructions(binary_path, symbol)
        if not insns:
            logger.warning(
                "Skipping symbol {} for static-offset hooks: no disassembly found",
                symbol,
            )
            continue

        insns = sorted(insns, key=lambda i: i["address"])
        addr_set = {insn["address"] for insn in insns}
        insn_by_addr = {insn["address"]: insn for insn in insns}
        insn_size_by_addr: dict[int, int] = {}
        for idx in range(len(insns) - 1):
            cur_addr = insns[idx]["address"]
            next_addr = insns[idx + 1]["address"]
            delta = next_addr - cur_addr
            # Keep only sane, positive deltas as best-effort instruction lengths.
            if 0 < delta <= 16:
                insn_size_by_addr[cur_addr] = delta

        def _is_hookable_addr(addr: int) -> bool:
            insn = insn_by_addr.get(addr)
            if insn is None:
                return False
            if _is_epilogue_mnemonic(insn["mnemonic"]):
                return False
            insn_size = insn_size_by_addr.get(addr)
            # Skip tiny 1-byte instruction slots (e.g., `ret`, some epilogues)
            # when size is known. This avoids fragile hook sites.
            if (
                insn_size is not None
                and insn_size < min_hook_instruction_bytes
            ):
                return False
            return True

        first_addr = insns[0]["address"]

        if first_addr < image_base:
            logger.warning(
                "Skipping symbol {} entry hook: address 0x{:x} below image base 0x{:x}",
                symbol,
                first_addr,
                image_base,
            )
        elif not _is_hookable_addr(first_addr):
            logger.warning(
                "Skipping symbol {} entry hook: unhookable entry instruction {} at 0x{:x}",
                symbol,
                insn_by_addr[first_addr]["mnemonic"],
                first_addr,
            )
        else:
            hook_specs.append(
                {
                    "id": next_function_id,
                    "channel": "function",
                    "symbol": symbol,
                    "func_token": symbol,
                    "source_address": f"0x{first_addr:x}",
                    "offset": f"0x{first_addr - image_base:x}",
                }
            )
            next_function_id += 1

        branch_addrs: set[int] = set()
        bb_leaders: set[int] = {first_addr}

        for idx, insn in enumerate(insns):
            mnemonic = insn["mnemonic"]
            if not _is_branch_mnemonic(mnemonic):
                continue

            cur_addr = insn["address"]
            branch_addrs.add(cur_addr)

            target_addr = _extract_direct_target_address(insn["operand"])
            if target_addr is not None and target_addr in addr_set:
                bb_leaders.add(target_addr)

            if _is_conditional_branch_mnemonic(mnemonic) and idx + 1 < len(insns):
                bb_leaders.add(insns[idx + 1]["address"])

        for addr in sorted(bb_leaders):
            if addr < image_base:
                continue
            if not _is_hookable_addr(addr):
                continue
            hook_specs.append(
                {
                    "id": next_bb_id,
                    "channel": "bb",
                    "symbol": symbol,
                    "func_token": f"{symbol}__bb",
                    "source_address": f"0x{addr:x}",
                    "offset": f"0x{addr - image_base:x}",
                }
            )
            next_bb_id += 1

        for addr in sorted(branch_addrs):
            if addr < image_base:
                continue
            if not _is_hookable_addr(addr):
                continue
            hook_specs.append(
                {
                    "id": next_branch_id,
                    "channel": "branch",
                    "symbol": symbol,
                    "func_token": f"{symbol}__br",
                    "source_address": f"0x{addr:x}",
                    "offset": f"0x{addr - image_base:x}",
                }
            )
            next_branch_id += 1

    expected_event_space = {
        "function": sum(1 for s in hook_specs if s["channel"] == "function"),
        "bb": sum(1 for s in hook_specs if s["channel"] == "bb"),
        "branch": sum(1 for s in hook_specs if s["channel"] == "branch"),
    }
    expected_event_space["total"] = (
        expected_event_space["function"]
        + expected_event_space["bb"]
        + expected_event_space["branch"]
    )

    return hook_specs, expected_event_space, image_base


def _generate_frida_static_offset_script_content(
    module_name: str,
    hook_specs: list[dict[str, Any]],
) -> str:
    hooks_json = json.dumps(hook_specs, indent=2)
    target_module_json = json.dumps(module_name)
    trace_tag_json = json.dumps(module_name)

    return f"""// Auto-generated by ConcoLLMic `instrument` (Frida static-offset mode)
'use strict';

const TARGET_MODULE = {target_module_json};
const TRACE_TAG = {trace_tag_json};
const HOOK_SPECS = {hooks_json};

const targetModule = Process.findModuleByName(TARGET_MODULE);
if (!targetModule) {{
  throw new Error(`[frida-static] target module not found: ${{TARGET_MODULE}}`);
}}

let writePtr = null;
if (typeof Module.findGlobalExportByName === 'function') {{
  writePtr = Module.findGlobalExportByName('write');
}}
if (!writePtr && typeof Module.findExportByName === 'function') {{
  writePtr = Module.findExportByName(null, 'write');
}}
if (!writePtr && typeof DebugSymbol === 'function' && typeof DebugSymbol.fromName === 'function') {{
  const sym = DebugSymbol.fromName('write');
  if (sym && sym.address) {{
    writePtr = sym.address;
  }}
}}
const writeSync =
  writePtr && typeof writePtr.isNull === 'function' && !writePtr.isNull()
    ? new NativeFunction(writePtr, 'int', ['int', 'pointer', 'ulong'])
    : null;

function resolveAddress(spec) {{
  const offset = parseInt(spec.offset, 16);
  if (!Number.isFinite(offset) || offset < 0) {{
    return null;
  }}
  if (offset >= targetModule.size) {{
    return null;
  }}
  return targetModule.base.add(offset);
}}

function emitTrace(action, spec) {{
  // Format aligned with ConcoLLMic FILE_TRACE_PATTERN:
  // [file] enter|exit function_name block_id
  const line = `[${{TRACE_TAG}}] ${{action}} ${{spec.func_token}} ${{spec.id}}\\n`;
  if (writeSync) {{
    try {{
      const buf = Memory.allocUtf8String(line);
      writeSync(2, buf, line.length);
      return;
    }} catch (_e) {{
      // fall through to async logging below
    }}
  }}
  console.log(line.trim());
}}

let attached = 0;
for (const spec of HOOK_SPECS) {{
  try {{
    const addr = resolveAddress(spec);
    if (!addr) {{
      continue;
    }}
    Interceptor.attach(addr, {{
      onEnter(_args) {{
        emitTrace('enter', spec);
      }},
      onLeave(_retval) {{
        if (spec.channel === 'function') {{
          emitTrace('exit', spec);
        }}
      }},
    }});
    attached += 1;
  }} catch (e) {{
    console.error(`[frida-static] failed: ${{spec.func_token}} @ ${{spec.offset}} (${{e}})`);
  }}
}}

console.error(
  `[frida-static] attached ${{attached}}/${{HOOK_SPECS.length}} static hooks on module ${{TARGET_MODULE}}`
);
"""


def generate_frida_hook_script(
    binary_path: str,
    script_out: str,
    module_name: str | None = None,
    max_hooks: int = 500,
    include_regex: str | None = None,
    exclude_regex: str | None = None,
    target_functions: list[str] | None = None,
    mode: str = "static_offset",
) -> tuple[str, int, list[str], str, dict[str, Any], list[dict[str, Any]], int]:
    if not os.path.exists(binary_path):
        raise FileNotFoundError(f"Binary not found: {binary_path}")
    if max_hooks <= 0:
        raise ValueError("max_hooks must be positive")
    if mode != "static_offset":
        raise ValueError(
            f"Unsupported frida mode: {mode}. Only 'static_offset' is supported."
        )

    symbols = _extract_function_symbols(binary_path)

    include_re = re.compile(include_regex) if include_regex else None
    exclude_re = re.compile(exclude_regex) if exclude_regex else None

    filtered: list[str] = []
    for sym in symbols:
        if include_re and not include_re.search(sym):
            continue
        if exclude_re and exclude_re.search(sym):
            continue
        filtered.append(sym)

    if target_functions:
        target_set = {_normalize_symbol_name(f) for f in target_functions if f.strip()}
        filtered = [sym for sym in filtered if sym in target_set]

    if len(filtered) > max_hooks:
        filtered = filtered[:max_hooks]

    if not filtered:
        raise RuntimeError(
            "No function symbols available after filtering. "
            "Try removing include/exclude regex or increasing symbol visibility."
        )

    hook_specs, expected_event_space, image_base = _compute_static_offset_hook_specs(
        binary_path,
        filtered,
    )
    if not hook_specs:
        raise RuntimeError(
            "Failed to produce static hook specs from target symbols. "
            "Ensure objdump/readelf are available and symbols are disassemblable."
        )

    target_module_name = module_name or os.path.basename(binary_path)
    content = _generate_frida_static_offset_script_content(
        target_module_name,
        hook_specs,
    )

    script_dir = os.path.dirname(script_out)
    if script_dir:
        os.makedirs(script_dir, exist_ok=True)
    with open(script_out, "w", encoding="utf-8") as f:
        f.write(content)

    symbols_out = script_out + ".symbols.txt"
    with open(symbols_out, "w", encoding="utf-8") as f:
        f.write("# id\tchannel\tfunc_token\tsymbol\toffset\tsource_address\n")
        for spec in hook_specs:
            f.write(
                f"{spec['id']}\t{spec['channel']}\t{spec['func_token']}\t{spec['symbol']}\t{spec['offset']}\t{spec['source_address']}\n"
            )

    logger.info(
        "Frida static-offset script generated at {} with {} hooks over {} symbols (binary: {})",
        script_out,
        len(hook_specs),
        len(filtered),
        binary_path,
    )
    logger.info(
        "Expected static event space: function={}, bb={}, branch={}, total={}",
        expected_event_space["function"],
        expected_event_space["bb"],
        expected_event_space["branch"],
        expected_event_space["total"],
    )
    logger.info("Frida symbol map saved at {}", symbols_out)
    return (
        script_out,
        len(hook_specs),
        filtered,
        target_module_name,
        expected_event_space,
        hook_specs,
        image_base,
    )


class TaskExecutor:
    """Class to manage concurrent file instrumentation tasks using a thread pool."""

    def __init__(self, max_workers):
        """Initialize the task executor with a thread pool.

        Args:
            max_workers (int): Maximum number of concurrent workers in the thread pool
        """
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.submitted_futures = (
            {}
        )  # Map future to file path for better status reporting
        self.futures_lock = threading.Lock()
        self.completed_results = []  # Store completed results (future, success)
        self.results_lock = threading.Lock()

        # Register cleanup method to be called at program exit
        atexit.register(self.cleanup)

    def cleanup(self):
        """Clean up thread pool resources"""
        logger.debug("Shutting down thread pool executor...")
        self.executor.shutdown(wait=True)
        logger.debug("Thread pool executor has been shut down.")

    def submit_task(
        self, func: Callable, file_path: str, rel_path: str, chunk_size: int
    ):
        """Submit a file instrumentation task to the thread pool

        Args:
            func (Callable): The function to execute
            file_path (str): The file path to process
            rel_path (str): The relative path of the file

        Returns:
            Future: The future representing the submitted task
        """
        logger.info(f'Submitting file "{file_path}" to thread pool for instrumentation')

        future = self.executor.submit(
            func, file_path=file_path, rel_path=rel_path, chunk_size=chunk_size
        )

        with self.futures_lock:
            self.submitted_futures[future] = file_path

        logger.debug(f'File "{file_path}" successfully submitted')
        return future

    def wait_for_available_worker(self):
        """Wait until there's at least one available worker in the pool.
        This method blocks until at least one worker becomes available.
        Also processes and yields completed futures as they finish.

        Returns:
            list: List of (future, success) tuples for each completed task
        """
        results = []

        def _process_completed_futures(done_futures):
            """Process completed futures and add them to the results list"""
            nonlocal results
            for future in done_futures:
                if future in self.submitted_futures:  # Make sure it's still there
                    file_path = self.submitted_futures.pop(future, "Unknown file")
                    try:
                        # Get the result, which will raise any exception that occurred
                        result = future.result()
                        assert isinstance(result, bool)
                        logger.info(
                            f'File "{file_path}" instrumentation completed, success: {result}'
                        )
                        results.append((future, result))
                    except Exception as e:
                        logger.error(f'Exception in file "{file_path}": {e}')
                        results.append((future, False))

        with self.futures_lock:
            # Process any completed futures even if the pool is not full
            current_futures = list(self.submitted_futures.keys())

            if current_futures:
                # Release the lock temporarily to check for completed futures
                self.futures_lock.release()

                # Check if any futures are done without blocking
                done, _ = concurrent.futures.wait(
                    current_futures,
                    timeout=0,  # Non-blocking check
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )

                # Reacquire the lock
                self.futures_lock.acquire()

                # Process completed futures
                _process_completed_futures(done)

            # If the pool is still full, wait for at least one task to complete
            while len(self.submitted_futures) >= self.max_workers:
                # Get current futures to check
                current_futures = list(self.submitted_futures.keys())

                # Release the lock temporarily to allow other threads to update futures
                self.futures_lock.release()

                # Wait for any future to complete
                done, _ = concurrent.futures.wait(
                    current_futures,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )

                # Reacquire the lock
                self.futures_lock.acquire()

                # Process completed futures
                _process_completed_futures(done)

        # Return all completed results
        return results

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
            f"Finally, waiting for {total_tasks} files to complete instrumentation"
        )

        # Process futures as they complete
        for future in concurrent.futures.as_completed(futures_map.keys()):
            file_path = futures_map[future]
            completed_tasks += 1

            try:
                # Get the result, which will raise any exception that occurred
                result = future.result()
                assert isinstance(result, bool)
                logger.info(
                    f'File "{file_path}" instrumentation completed, success: {result} ({completed_tasks}/{total_tasks})'
                )
                yield future, result
            except Exception as e:
                logger.error(f'Exception in file "{file_path}": {e}')
                yield future, False


def _is_in_excluded_dirs(file_path, exclude_dirs_list):
    """Check if file is in an excluded directory"""
    if not exclude_dirs_list:
        return False

    return any(
        os.path.normpath(file_path).startswith(os.path.normpath(dir))
        or os.path.normpath(dir) in os.path.normpath(file_path).split(os.sep)
        for dir in exclude_dirs_list
    )


def _is_already_instrumented(file_path):
    """
    Check if file is already instrumented by looking for cost summary comments at the end

    Returns False if file doesn't exist or can't be read
    """
    if not os.path.exists(file_path):
        return False

    try:
        language = detect_language(file_path)
        if not language:
            return False

        comment_token = get_comment_token(language)

        with open(file_path, encoding="utf-8", errors="ignore") as f:
            last_lines = deque(f, 10)

            for line in last_lines:
                line = line.strip()
                if (
                    line.lstrip().startswith(comment_token)
                    and "instrumented cost" in line
                ):
                    return True
            return False
    except Exception as e:
        logger.warning(f"Error checking if {file_path} is already instrumented: {e}")
        return False


def _copy_file(src, dst):
    """Copy a file"""
    os.system(f"cp {src} {dst}")


def _prepare_instrumentation_summary(
    instrumented_files,
    instr_failed_files,
    excluded_files,
    already_instrumented_files,
    unsupported_files,
    special_files,
    binary_files,
    read_error_files,
):
    """
    Prepare common data for instrumentation summary and detailed information

    Args:
        instrumented_files: List of successfully instrumented files
        instr_failed_files: List of files that failed during instrumentation
        excluded_files: List of excluded files
        already_instrumented_files: List of already instrumented files
        unsupported_files: List of unsupported language files
        special_files: List of special files
        binary_files: List of binary files
        read_error_files: List of files with errors

    Returns:
        Tuple containing:
        - Dictionary of compressed file paths
        - Dictionary of calculated totals
    """
    # Compress all file lists
    compressed_paths = {
        "instr": compress_paths(instrumented_files),
        "instr_failed": compress_paths(instr_failed_files),
        "excluded": compress_paths(excluded_files),
        "already_instr": compress_paths(already_instrumented_files),
        "unsupported": compress_paths(unsupported_files),
        "special": compress_paths(special_files),
        "binary": compress_paths(binary_files),
        "read_error": compress_paths(read_error_files),
    }

    # Calculate totals
    total_instrumented = len(instrumented_files) + len(already_instrumented_files)

    total_other_skipped = (
        len(unsupported_files)
        + len(special_files)
        + len(binary_files)
        + len(read_error_files)
    )

    total_non_instrumented = len(excluded_files) + total_other_skipped

    total_processed = (
        total_instrumented + len(instr_failed_files) + total_non_instrumented
    )

    totals = {
        "total_processed": total_processed,
        "total_instrumented": total_instrumented,
        "total_non_instrumented": total_non_instrumented,
        "total_other_skipped": total_other_skipped,
    }

    return compressed_paths, totals


def format_instrumentation_paths(
    compressed_paths, log_output=True, max_files_to_show=5, indent_level=1
):
    """
    Format path information for instrumentation output

    Args:
        compressed_paths: Dictionary with compressed directory paths as keys and lists of files as values
        log_output: If True, format for terminal/log output with limited details. If False, more verbose output.
        max_files_to_show: Maximum number of files to display in log output mode (ignored in detailed mode)
        indent_level: Level of indentation for log output (number of "  " prefixes)

    Returns:
        Dictionary with formatted directory information or formatted log lines depending on log_output
    """
    indent = "  " * indent_level

    if log_output:
        # Format for log output - returns list of strings
        log_lines = []
        for dirname, files in sorted(compressed_paths.items()):
            if dirname.endswith("/*"):
                # Compressed directory
                log_lines.append(f"{indent}{dirname}: {len(files)} files")
            else:
                # Regular directory with files
                display_name = "./" if dirname == "" else f"{dirname}/"
                log_lines.append(f"{indent}{display_name}: {len(files)} files")
                _files_to_show = min(len(files), max_files_to_show)
                if _files_to_show > 0:  # Only show individual files if there are few
                    for file in sorted(files)[:_files_to_show]:
                        log_lines.append(f"{indent}  - {file}")
                    if len(files) > max_files_to_show:
                        log_lines.append(
                            f"{indent}  - ... and {len(files) - max_files_to_show} more files"
                        )
        return log_lines
    else:
        # Format for detailed info - returns structured data
        structured_data = {}
        for dirname, files in sorted(compressed_paths.items()):
            is_compressed = dirname.endswith("/*")
            # use "./" as root directory
            key_name = "./" if dirname == "" else dirname
            structured_data[key_name] = {
                "is_compressed": is_compressed,
                # Include all files in detailed info even for compressed directories
                f"{len(files)}_files": sorted(files),
            }
        return structured_data


def format_file_category(
    file_list, compressed_dirs, title, indent_level=1, max_dirs_to_show=5, out_dir=None
):
    """
    Format a file category with consistent styling for the log output

    Args:
        file_list: List of files in this category
        compressed_dirs: Dictionary of compressed directories for these files
        title: Category title to display
        indent_level: Indentation level for this category
        max_dirs_to_show: Maximum number of directories to show before truncating
        out_dir: Output directory (for info_file reference in truncation message)

    Returns:
        List of formatted log lines for this category
    """
    lines = []

    # Add category header with file count
    lines.append(f"{' ' * (indent_level * 2)}{title}: {len(file_list)} files")

    # Only proceed if there are files in this category
    if file_list:
        # Limit by directory count
        if max_dirs_to_show is None or len(compressed_dirs) <= max_dirs_to_show:
            for line in format_instrumentation_paths(
                compressed_dirs, indent_level=indent_level + 1
            ):
                lines.append(line)
        elif out_dir:  # Only show truncation message if out_dir is provided
            lines.append(
                f"{' ' * ((indent_level + 1) * 2)}(Too many directories to display, see detailed info in {out_dir}/{INSTRUMENTATION_INFO_FILE})"
            )

    return lines


def format_other_skipped_files(
    unsupported_files,
    compressed_unsupported,
    special_files,
    compressed_special,
    binary_files,
    compressed_binary,
    read_error_files,
    compressed_error,
    out_dir,
):
    """
    Format the "Other skipped files" section with all its subcategories

    Args:
        unsupported_files: List of unsupported language files
        compressed_unsupported: Compressed directories for unsupported files
        special_files: List of special files
        compressed_special: Compressed directories for special files
        binary_files: List of binary files
        compressed_binary: Compressed directories for binary files
        read_error_files: List of files with read errors
        compressed_error: Compressed directories for read error files
        out_dir: Output directory for info file reference

    Returns:
        List of formatted log lines for the entire "Other skipped files" section
    """
    total_skipped = (
        len(unsupported_files)
        + len(special_files)
        + len(binary_files)
        + len(read_error_files)
    )

    skipped_summary = []
    # Main header for other skipped files
    skipped_summary.append(f"  b) Other skipped files ({total_skipped} files):")

    # i. Unsupported language files
    unsupported_lines = format_file_category(
        unsupported_files,
        compressed_unsupported,
        "i. Unsupported language",
        indent_level=3,
        max_dirs_to_show=5,
        out_dir=out_dir,
    )
    skipped_summary.extend(unsupported_lines)

    # ii. Special files
    special_lines = format_file_category(
        special_files,
        compressed_special,
        "ii. Special files",
        indent_level=3,
        max_dirs_to_show=5,
        out_dir=out_dir,
    )
    skipped_summary.extend(special_lines)

    # iii. Binary/non-UTF-8 files
    binary_lines = format_file_category(
        binary_files,
        compressed_binary,
        "iii. Binary/non-UTF-8 files",
        indent_level=3,
        max_dirs_to_show=5,
        out_dir=out_dir,
    )
    skipped_summary.extend(binary_lines)

    # iv. Error files
    error_lines = format_file_category(
        read_error_files,
        compressed_error,
        "iv. Error files",
        indent_level=3,
        max_dirs_to_show=5,
        out_dir=out_dir,
    )
    skipped_summary.extend(error_lines)

    return skipped_summary


def generate_detailed_instrumentation_info(
    instrumented_files,
    instr_failed_files,
    excluded_files,
    already_instrumented_files,
    unsupported_files,
    special_files,
    binary_files,
    read_error_files,
):
    """
    Generate detailed instrumentation information and save to file

    Args:
        instrumented_files: List of successfully instrumented files
        instr_failed_files: List of files that failed during instrumentation
        excluded_files: List of excluded files
        already_instrumented_files: List of already instrumented files
        unsupported_files: List of unsupported language files
        special_files: List of special files
        binary_files: List of binary files
        read_error_files: List of files with errors
    """
    # Get compressed paths and totals
    compressed_paths, totals = _prepare_instrumentation_summary(
        instrumented_files,
        instr_failed_files,
        excluded_files,
        already_instrumented_files,
        unsupported_files,
        special_files,
        binary_files,
        read_error_files,
    )

    # Create the detailed info structure
    detailed_info = {
        "timestamp": datetime.datetime.now().isoformat(),
        "summary": {
            "total_processed_files": totals["total_processed"],
            "instrumented_files": totals["total_instrumented"],
            "instrumentation_failed_files": len(instr_failed_files),
            "non_instrumented_files": totals["total_non_instrumented"],
        },
        "categories": {
            "1_instrumented_files": {
                "count": totals["total_instrumented"],
                "subcategories": {
                    "a_newly_instrumented_files": {
                        "count": len(instrumented_files),
                        "directories": format_instrumentation_paths(
                            compressed_paths["instr"], log_output=False
                        ),
                    },
                    "b_already_instrumented_files": {
                        "count": len(already_instrumented_files),
                        "directories": format_instrumentation_paths(
                            compressed_paths["already_instr"], log_output=False
                        ),
                    },
                },
            },
            "2_instrumentation_failed_files": {
                "count": len(instr_failed_files),
                "directories": format_instrumentation_paths(
                    compressed_paths["instr_failed"], log_output=False
                ),
            },
            "3_non_instrumented_files": {
                "count": totals["total_non_instrumented"],
                "subcategories": {
                    "a_excluded_files": {
                        "count": len(excluded_files),
                        "directories": format_instrumentation_paths(
                            compressed_paths["excluded"], log_output=False
                        ),
                    },
                    "b_other_skipped_files": {
                        "count": totals["total_other_skipped"],
                        "subcategories": {
                            "unsupported_language": {
                                "count": len(unsupported_files),
                                "directories": format_instrumentation_paths(
                                    compressed_paths["unsupported"], log_output=False
                                ),
                            },
                            "special_files": {
                                "count": len(special_files),
                                "directories": format_instrumentation_paths(
                                    compressed_paths["special"], log_output=False
                                ),
                            },
                            "binary_files": {
                                "count": len(binary_files),
                                "directories": format_instrumentation_paths(
                                    compressed_paths["binary"], log_output=False
                                ),
                            },
                            "read_error_files": {
                                "count": len(read_error_files),
                                "directories": format_instrumentation_paths(
                                    compressed_paths["read_error"], log_output=False
                                ),
                            },
                        },
                    },
                },
            },
        },
    }

    return yaml.dump(
        detailed_info, sort_keys=False, default_flow_style=False, allow_unicode=True
    )


def _instrument_file(file_path, rel_path, chunk_size):
    """Instrument a single file

    Args:
        file_path (str): Path to the output file
        rel_path (str): Relative path for marking

    Returns:
        bool: True if instrumentation was successful, False otherwise
    """
    instr_agent = InstrumentationAgent()
    logger.info(f'Instrumenting "{file_path}"')
    instrumented_content, success = instr_agent.instrument(
        source_code_file=file_path, mark=rel_path, chunk_size=chunk_size
    )
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(
            instrumented_content
        )  # If not successful, the file is INCOMPLETE (for debugging)
    return success


def instrument_code(
    src_dir: str | None,
    out_dir: str,
    instr_languages: str | None,
    exclude_dirs: str | None,
    parallel_num: int,
    chunk_size: int,
    frida_binary: str | None = None,
    frida_script_out: str | None = None,
    frida_module: str | None = None,
    frida_only: bool = False,
    frida_max_hooks: int = 500,
    frida_include_regex: str | None = None,
    frida_exclude_regex: str | None = None,
    frida_target_functions: str | None = None,
    frida_mode: str = "static_offset",
):
    """
    Code instrumentation phase

    Args:
        src_dir: Source code directory
        out_dir: Output directory
        instr_languages: Languages to instrument, comma-separated string
        exclude_dirs: Directories to exclude, comma-separated string
        parallel_num: Number of parallel workers for instrumentation
        chunk_size: Chunk size for instrumentation
        frida_binary: Optional ELF binary / shared library path for Frida script generation
        frida_script_out: Optional output path for generated Frida script
        frida_module: Optional module name used by Frida script at runtime
        frida_only: If True, generate Frida artifacts only (skip source instrumentation)
        frida_max_hooks: Max number of target symbols to include in hook generation
        frida_include_regex: Optional regex to include symbol names
        frida_exclude_regex: Optional regex to exclude symbol names
        frida_target_functions: Optional comma-separated exact function symbol names
        frida_mode: "static_offset"
    """
    logger.info("Starting code instrumentation phase...")

    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    frida_artifacts: dict[str, Any] | None = None

    if frida_binary:
        frida_script_path = frida_script_out or os.path.join(out_dir, "frida_hooks.js")
        target_functions = None
        if frida_target_functions:
            target_functions = [f.strip() for f in frida_target_functions.split(",")]
        (
            generated_script_path,
            generated_symbol_count,
            generated_symbols,
            generated_module_name,
            generated_expected_event_space,
            generated_hook_specs,
            generated_image_base,
        ) = generate_frida_hook_script(
            binary_path=frida_binary,
            script_out=frida_script_path,
            module_name=frida_module,
            max_hooks=frida_max_hooks,
            include_regex=frida_include_regex,
            exclude_regex=frida_exclude_regex,
            target_functions=target_functions,
            mode=frida_mode,
        )
        frida_artifacts = {
            "binary_path": frida_binary,
            "script_path": generated_script_path,
            "symbol_count": generated_symbol_count,
            "symbols": generated_symbols,
            "module_name": generated_module_name,
            "mode": frida_mode,
            "expected_event_space": generated_expected_event_space,
            "hook_specs": generated_hook_specs,
            "image_base": generated_image_base,
        }
        logger.info("Frida script ready: {}", generated_script_path)

    if frida_only:
        if not frida_binary:
            raise ValueError("--frida_only requires --frida_binary")
        if frida_artifacts:
            trace_map = _build_frida_trace_map(
                binary_path=frida_artifacts["binary_path"],
                module_name=frida_artifacts["module_name"],
                frida_mode=frida_artifacts["mode"],
                symbols=frida_artifacts["symbols"],
                project_dir=None,
                expected_event_space=frida_artifacts.get("expected_event_space"),
                hook_specs=frida_artifacts.get("hook_specs"),
                image_base=frida_artifacts.get("image_base"),
            )
            trace_map_path = os.path.join(out_dir, FRIDA_TRACE_MAP_FILE)
            _save_frida_trace_map(trace_map, trace_map_path)
            logger.info("Frida trace map saved at {}", trace_map_path)
        logger.info(
            "Frida-only mode enabled. Source instrumentation skipped successfully."
        )
        return

    if not src_dir:
        raise ValueError("--src_dir is required unless --frida_only is enabled")
    if not instr_languages:
        raise ValueError("--instr_languages is required unless --frida_only is enabled")

    # Convert exclude_dirs from comma-separated string to list if provided
    exclude_dirs_list = []
    if exclude_dirs:
        exclude_dirs_list = [
            os.path.normpath(dir.strip()) for dir in exclude_dirs.split(",")
        ]
        logger.info(f"Excluding directories: {exclude_dirs_list}")

    # Automatically add common directories to exclude list
    common_dirs_to_exclude = [
        ".git",
        ".github",
        ".vscode",
        "node_modules",
        "__pycache__",
        "venv",
        ".idea",
    ]
    for common_dir in common_dirs_to_exclude:
        if common_dir not in exclude_dirs_list:
            exclude_dirs_list.append(common_dir)
            logger.info(
                f"Automatically adding common directories to excluded directories: {common_dirs_to_exclude}"
            )

    # List all code files in the source directory
    all_files = list_all_files(src_dir)
    logger.info(f"Found {len(all_files)} files to process")

    # Convert instr_languages from comma-separated string to list
    supported_languages = [lang.strip() for lang in instr_languages.split(",")]
    if "c" in supported_languages:
        supported_languages.append(
            "cpp"
        )  # ".h" ".cc" files are detected as "cpp" language
    logger.info(f"Instrumenting languages: {supported_languages}")

    # Lists to track processed files for summary
    instrumented_files = []
    instr_failed_files = []
    excluded_files = []
    already_instrumented_files = []
    unsupported_files = []
    special_files = []
    binary_files = []
    read_error_files = []

    # Create task executor for parallel instrumentation
    task_executor = TaskExecutor(max_workers=parallel_num)
    futures_to_outfiles = {}

    # Process each file - files that need to be instrumented will be submitted to the thread pool
    # When the thread pool is full (10 workers), we'll wait for a worker to become available before submitting new tasks
    for file in all_files:
        # create output file with the same relative path
        rel_path = os.path.relpath(file, src_dir)
        out_file = os.path.join(out_dir, rel_path)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

        should_copy = True
        should_instrument = True
        skip_reason = None

        # 1. Check if file is in excluded directory
        if _is_in_excluded_dirs(file, exclude_dirs_list):
            skip_reason = "in excluded directory"
            should_instrument = False
            excluded_files.append(rel_path)

        # 2. Check if file is already instrumented
        elif _is_already_instrumented(out_file):
            skip_reason = "already instrumented"
            should_instrument = False
            should_copy = False  # No need to copy already instrumented files
            already_instrumented_files.append(rel_path)

        # 3. Check file type
        elif detect_language(file) not in supported_languages:
            skip_reason = f"unsupported language ({detect_language(file)})"
            should_instrument = False
            unsupported_files.append(rel_path)

        # 4. Check if it's a special file
        elif file.endswith("setup.py"):  # TODO: add other special files to skip
            skip_reason = "special file"
            should_instrument = False
            special_files.append(rel_path)

        # 5. Check if file is binary or non-UTF-8
        else:
            try:
                with open(file, encoding="utf-8") as f:
                    f.read()
            except UnicodeDecodeError:
                skip_reason = "binary or non-UTF-8 file"
                should_instrument = False
                binary_files.append(rel_path)
            except Exception as e:
                skip_reason = f"error reading file: {e}"
                should_instrument = False
                read_error_files.append(rel_path)

        # Take action based on checks
        if not should_instrument:
            if skip_reason:
                logger.info(f"Skipping {file}: {skip_reason}")

            if should_copy:
                _copy_file(file, out_file)
            continue

        # Perform instrumentation - wait for available worker if needed and process any completed tasks
        else:
            assert should_copy
            completed_results = task_executor.wait_for_available_worker()
            for future, success in completed_results:
                completed_rel_path = futures_to_outfiles.get(future)
                if completed_rel_path:
                    if success:
                        instrumented_files.append(completed_rel_path)
                    else:
                        instr_failed_files.append(completed_rel_path)

            # Copy file before submitting the task
            _copy_file(file, out_file)

            # Submit the instrumentation task
            future = task_executor.submit_task(
                func=_instrument_file,
                file_path=out_file,
                rel_path=rel_path,
                chunk_size=chunk_size,
            )
            futures_to_outfiles[future] = rel_path

    # Wait for all tasks to complete and collect results
    # This ensures all instrumentation tasks finish before we generate the summary
    for future, success in task_executor.wait_for_all_tasks():
        rel_path = futures_to_outfiles.get(future)
        if rel_path:
            if success:  # Success
                instrumented_files.append(rel_path)
            else:  # Failed
                instr_failed_files.append(rel_path)

    ### final check
    # check instrumentation or not
    for file in instrumented_files + already_instrumented_files:
        if not _is_already_instrumented(os.path.join(out_dir, file)):
            logger.error(
                f"File {file} was not instrumented but recorded as instrumented"
            )

    for file in instr_failed_files:
        if _is_already_instrumented(os.path.join(out_dir, file)):
            logger.error(
                f"File {file} was recorded as instrumentation failed but was instrumented"
            )

    # ensure src_dir and out_dir have the same files (use relative path!)
    src_files = [
        os.path.relpath(file, src_dir)
        for file in list_all_files(src_dir, recursive=True)
    ]
    out_files = [
        os.path.relpath(file, out_dir)
        for file in list_all_files(out_dir, recursive=True)
    ]
    for file in src_files:
        if file not in out_files:
            logger.error(
                f"File {file} is in source dir ({src_dir}) but not in out dir ({out_dir})"
            )

    # Generate summary grouped by directories
    logger.info("=" * 50)
    logger.info("INSTRUMENTATION SUMMARY")
    logger.info("=" * 50)

    # Get compressed paths and totals
    compressed_paths, totals = _prepare_instrumentation_summary(
        instrumented_files,
        instr_failed_files,
        excluded_files,
        already_instrumented_files,
        unsupported_files,
        special_files,
        binary_files,
        read_error_files,
    )

    # Show totals first
    logger.info(f"Total files processed: {totals['total_processed']} files")

    # 1. Instrumented files category
    logger.info(f"1) Instrumented files: {totals['total_instrumented']} files")

    # 1.1 Newly instrumented files
    logger.info(f"  a) Newly instrumented files: {len(instrumented_files)} files")
    if instrumented_files:
        for line in format_instrumentation_paths(
            compressed_paths["instr"], indent_level=2
        ):
            logger.info(line)

    # 1.2 Already instrumented files
    logger.info(
        f"  b) Already instrumented files: {len(already_instrumented_files)} files"
    )
    if already_instrumented_files:
        for line in format_instrumentation_paths(
            compressed_paths["already_instr"], indent_level=2
        ):
            logger.info(line)

    # 2. Instrumentation failed files category
    logger.info(f"2) Instrumentation failed files: {len(instr_failed_files)} files")
    if instr_failed_files:
        for line in format_instrumentation_paths(
            compressed_paths["instr_failed"], indent_level=1
        ):
            logger.info(line)

    # 3. Non-instrumented files category
    logger.info(f"3) Non-instrumented files: {totals['total_non_instrumented']} files")

    # 3.1 Excluded files subcategory - use the helper function
    excluded_lines = format_file_category(
        excluded_files,
        compressed_paths["excluded"],
        "a) Excluded files (in excluded directories)",
        indent_level=1,
        max_dirs_to_show=None,  # No limit for these main categories
    )
    for line in excluded_lines:
        logger.info(line)

    # 3.2 Other skipped files subcategory - use the format_other_skipped_files helper
    skipped_summary = format_other_skipped_files(
        unsupported_files,
        compressed_paths["unsupported"],
        special_files,
        compressed_paths["special"],
        binary_files,
        compressed_paths["binary"],
        read_error_files,
        compressed_paths["read_error"],
        out_dir,
    )

    # Log all skipped summary lines
    for line in skipped_summary:
        logger.info(line)

    logger.info("=" * 50)
    logger.info(f"Instrumentation completed. Output in {out_dir}")

    # Generate detailed instrumentation info and save to file
    content = generate_detailed_instrumentation_info(
        instrumented_files,
        instr_failed_files,
        excluded_files,
        already_instrumented_files,
        unsupported_files,
        special_files,
        binary_files,
        read_error_files,
    )
    # Save detailed info to file
    info_file_path = os.path.join(out_dir, INSTRUMENTATION_INFO_FILE)
    with open(info_file_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"Detailed instrumentation info saved to {info_file_path}")

    if frida_artifacts:
        trace_map = _build_frida_trace_map(
            binary_path=frida_artifacts["binary_path"],
            module_name=frida_artifacts["module_name"],
            frida_mode=frida_artifacts["mode"],
            symbols=frida_artifacts["symbols"],
            project_dir=out_dir,
            expected_event_space=frida_artifacts.get("expected_event_space"),
            hook_specs=frida_artifacts.get("hook_specs"),
            image_base=frida_artifacts.get("image_base"),
        )
        trace_map_path = os.path.join(out_dir, FRIDA_TRACE_MAP_FILE)
        _save_frida_trace_map(trace_map, trace_map_path)
        logger.info("Frida trace map saved at {}", trace_map_path)

    logger.info("=" * 50)


def setup_instrument_parser(subparsers):
    """Setup the instrument subcommand parser"""
    instrument_parser = subparsers.add_parser(
        "instrument", help="perform code instrumentation"
    )
    instrument_parser.add_argument(
        "--src_dir",
        type=str,
        help="source code directory",
        required=False,
        default=None,
    )
    instrument_parser.add_argument(
        "--out_dir", type=str, help="output directory", required=True
    )
    instrument_parser.add_argument(
        "--instr_languages",
        type=str,
        help="comma-separated list of languages to instrument (e.g. 'python,java,c,cpp')",
        required=False,
        default="c,cpp,python,java",
    )
    instrument_parser.add_argument(
        "--exclude_dirs",
        type=str,
        help="comma-separated list of directories to exclude from instrumentation (e.g. 'deps,tests,examples')",
        required=False,
        default=None,
    )
    instrument_parser.add_argument(
        "--parallel_num",
        type=int,
        help="number of parallel workers for instrumentation",
        required=False,
        default=10,
    )
    instrument_parser.add_argument(
        "--chunk_size",
        type=int,
        help="chunk size for instrumentation",
        required=False,
        default=800,
    )
    instrument_parser.add_argument(
        "--frida_binary",
        type=str,
        help="path to ELF binary/shared library to generate a Frida hook script from symbols",
        required=False,
        default=None,
    )
    instrument_parser.add_argument(
        "--frida_script_out",
        type=str,
        help="output path for generated Frida script (default: <out_dir>/frida_hooks.js)",
        required=False,
        default=None,
    )
    instrument_parser.add_argument(
        "--frida_module",
        type=str,
        help="module name used at runtime by Frida (default: basename of --frida_binary)",
        required=False,
        default=None,
    )
    instrument_parser.add_argument(
        "--frida_only",
        action="store_true",
        help="only generate Frida script artifacts; skip source instrumentation",
    )
    instrument_parser.add_argument(
        "--frida_max_hooks",
        type=int,
        help="maximum number of target symbols to include when generating static Frida hooks",
        required=False,
        default=500,
    )
    instrument_parser.add_argument(
        "--frida_include_regex",
        type=str,
        help="optional regex: only include symbol names matching this pattern",
        required=False,
        default=None,
    )
    instrument_parser.add_argument(
        "--frida_exclude_regex",
        type=str,
        help="optional regex: exclude symbol names matching this pattern",
        required=False,
        default=None,
    )
    instrument_parser.add_argument(
        "--frida_target_functions",
        type=str,
        help='optional comma-separated exact function symbols to hook (e.g. "main,_Z17validate_bracketsPKc")',
        required=False,
        default=None,
    )
    instrument_parser.add_argument(
        "--frida_mode",
        type=str,
        choices=["static_offset"],
        help="Frida generation mode: static_offset (deterministic function/basic-block/branch offset hooks)",
        required=False,
        default="static_offset",
    )
    return instrument_parser
