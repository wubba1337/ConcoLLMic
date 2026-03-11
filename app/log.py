import re
import time
from os import get_terminal_size

from loguru import logger
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule


def terminal_width():
    try:
        return get_terminal_size().columns
    except OSError:
        return 80


WIDTH = min(120, terminal_width() - 10)
TOOL_WIDTH = min(90, WIDTH - 20)
TOOL_INDENT = max(WIDTH - TOOL_WIDTH, 10)

console = Console()

print_stdout = True

# Thread-local active agent style for tool call panels — set via set_active_agent()
import threading

_thread_local = threading.local()

_AGENT_STYLES = {
    "summarizer": ("blue", "Summarizer"),
    "solver": ("yellow", "Solver"),
}


def set_active_agent(agent: str) -> None:
    """Set the active agent for the current thread to control tool call panel styling.

    Args:
        agent: One of "summarizer", "solver", or "" to reset.
    """
    style, label = _AGENT_STYLES.get(agent, ("dim cyan", ""))
    _thread_local.agent_style = style
    _thread_local.agent_label = label


def _get_active_agent_style() -> tuple[str, str]:
    return (
        getattr(_thread_local, "agent_style", "dim cyan"),
        getattr(_thread_local, "agent_label", ""),
    )


def log_exception(exception):
    logger.exception(exception)


def print_banner(msg: str) -> None:
    if not print_stdout:
        return

    banner = f" {msg} ".center(WIDTH, "=")
    console.print()
    console.print(banner, style="bold")
    console.print()


def replace_html_tags(content: str):
    """
    Helper method to process the content before printing to markdown.
    """
    replace_dict = {
        "<file>": "[file]",
        "<class>": "[class]",
        "<func>": "[func]",
        "<method>": "[method]",
        "<code>": "[code]",
        "<original>": "[original]",
        "<patched>": "[patched]",
        "</file>": "[/file]",
        "</class>": "[/class]",
        "</func>": "[/func]",
        "</method>": "[/method]",
        "</code>": "[/code]",
        "</original>": "[/original]",
        "</patched>": "[/patched]",
    }
    for key, value in replace_dict.items():
        content = content.replace(key, value)
    return content


def _compact_prompt(msg: str) -> str:
    """Strip XML wrapper tags and collapse blank lines for cleaner display."""

    # Replace known XML tags with readable section labels
    # Handle function_call_chain specially: convert [file](func) markdown links to readable text
    def _replace_call_chain(m):
        chain = m.group(1)
        chain = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1:\2()", chain)
        return f"\n\n**Function Call Chain:** {chain}\n"

    msg = re.sub(
        r"<function_call_chain>(.*?)</function_call_chain>",
        _replace_call_chain,
        msg,
        flags=re.DOTALL,
    )
    _tag_labels = {
        "execution_information": "Execution Information",
        "execution_trace": "Execution Trace",
        "target_path_constraint": "Target Path Constraint",
        "new_execution_information": "New Execution Information",
        "new_execution_trace": "New Execution Trace",
    }
    for tag, label in _tag_labels.items():
        msg = re.sub(rf"<{tag}>", f"\n\n**{label}:**\n", msg)
        msg = re.sub(rf"</{tag}>", "", msg)
    # Remove any remaining XML wrapper tags with underscores
    msg = re.sub(r"</?[a-z]+(?:_[a-z]+)+>", "", msg)
    # Collapse 2+ consecutive blank lines into a single blank line
    msg = re.sub(r"\n{3,}", "\n\n", msg)

    # Within code fences, collapse double blank lines to single
    def _compact_code_block(m):
        return re.sub(r"\n\n", "\n", m.group(0))

    msg = re.sub(r"```\w*\n.*?```", _compact_code_block, msg, flags=re.DOTALL)
    return msg.strip()


def print_ace(msg: str, desc="") -> None:

    msg = replace_html_tags(msg)
    display_msg = _compact_prompt(msg)
    markdown = Markdown(display_msg)

    title = desc if desc else "AgenticConcolicExecution"

    panel = Panel(
        markdown,
        title=title,
        title_align="left",
        border_style="magenta",
        width=WIDTH,
    )
    if not print_stdout:
        logger.debug(f"{title}:\n{msg}")
    else:
        console.print(panel)


def print_summarize(msg: str, desc="") -> None:

    msg = replace_html_tags(msg)
    markdown = Markdown(msg)

    name = "Constraint Summarize Agent"
    if desc:
        title = f"{name} ({desc})"
    else:
        title = name

    panel = Panel(
        markdown,
        title=title,
        title_align="left",
        border_style="blue",
        width=WIDTH,
    )
    if not print_stdout:
        logger.debug(f"{title}:\n{msg}")
    else:
        console.print(panel)


def print_solve(msg: str, desc="") -> None:

    msg = replace_html_tags(msg)
    markdown = Markdown(msg)

    name = "Constraint Solving Agent"
    if desc:
        title = f"{name} ({desc})"
    else:
        title = name

    panel = Panel(
        markdown,
        title=title,
        title_align="left",
        border_style="yellow",
        width=WIDTH,
    )
    if not print_stdout:
        logger.debug(f"{title}:\n{msg}")
    else:
        console.print(panel)


def print_selection(content: str, desc="") -> None:

    name = "Selection Agent"
    if desc:
        title = f"{name} ({desc})"
    else:
        title = name

    panel = Panel(
        escape(content),
        title=title,
        title_align="left",
        border_style="green",
    )
    if not print_stdout:
        logger.debug(f"{title}:\n{content}")
    else:
        console.print(panel)


def print_instrument(content: str, desc="") -> None:

    name = "Instrumentation Agent"
    if desc:
        title = f"{name} ({desc})"
    else:
        title = name

    panel = Panel(
        escape(content),
        title=title,
        title_align="left",
        border_style="red",
    )
    if not print_stdout:
        logger.debug(f"{title}:\n{content}")
    else:
        console.print(panel)


def print_reproducer(msg: str, desc="") -> None:
    if not print_stdout:
        return

    markdown = Markdown(msg)

    name = "Reproducer Test Generation"
    if desc:
        title = f"{name} ({desc})"
    else:
        title = name

    panel = Panel(
        markdown,
        title=title,
        title_align="left",
        border_style="green",
        width=WIDTH,
    )
    console.print(panel)


def print_exec_reproducer(msg: str, desc="") -> None:
    if not print_stdout:
        return

    markdown = Markdown(msg)

    name = "Reproducer Execution Result"
    if desc:
        title = f"{name} ({desc})"
    else:
        title = name

    panel = Panel(
        markdown,
        title=title,
        title_align="left",
        border_style="blue",
        width=WIDTH,
    )
    console.print(panel)


def print_review(msg: str, desc="") -> None:
    if not print_stdout:
        return

    markdown = Markdown(msg)

    name = "Review"
    if desc:
        title = f"{name} ({desc})"
    else:
        title = name

    panel = Panel(
        markdown,
        title=title,
        title_align="left",
        border_style="purple",
        width=WIDTH,
    )
    console.print(panel)


def log_and_print(msg):
    logger.info(msg)
    if print_stdout:
        console.print(msg)


def log_and_cprint(msg, **kwargs):
    logger.info(msg)
    if print_stdout:
        console.print(msg, **kwargs)


def log_and_always_print(msg):
    """
    A mode which always print to stdout, no matter what.
    Useful when running multiple tasks and we just want to see the important information.
    """
    logger.info(msg)
    # always include time for important messages
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    console.print(f"\n[{t}] {msg}")


def print_with_time(msg):
    """
    Print a msg to console with timestamp.
    """
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    console.print(f"\n[{t}] {msg}")


def print_phase(phase: str, round_num: int = None) -> None:
    """Print a prominent phase header for demo visibility."""
    if not print_stdout:
        return
    if round_num is not None:
        console.print()
        console.print(
            Rule(
                f"[bold white] ROUND {round_num} [/]",
                style="bright_blue",
                characters="═",
            )
        )
    console.print()
    console.print(Rule(f"[bold] {phase} [/]", style="dim", align="left"))
    console.print()


def print_step(msg: str) -> None:
    """Print a compact step indicator."""
    if not print_stdout:
        return
    console.print(f"  [dim]▸[/] {msg}")


def print_tool_call(
    tool_name: str, content: str, icon: str = "", func_name: str = None
) -> None:
    """Right-aligned panel for agent tool calls."""
    if not print_stdout:
        logger.debug(f"Tool call [{func_name or tool_name}]: {content}")
        return

    # Show tool call indicator line above the panel
    agent_style, agent_label = _get_active_agent_style()
    agent_prefix = f"[{agent_style}]{agent_label}[/] " if agent_label else ""
    call_label = f"{agent_prefix}[dim]calls tool 🔧[/] [bold {agent_style}]{func_name or tool_name}[/]"
    console.print(Padding(call_label, (0, 0, 0, TOOL_INDENT)))

    content = replace_html_tags(content)
    # Convert markdown headers (## Header) to bold text to prevent Rich centering them
    content = re.sub(r"^#{1,6}\s+(.+)$", r"**\1**", content, flags=re.MULTILINE)
    renderable = Markdown(content)

    title = f"{icon} {tool_name}" if icon else tool_name
    panel = Panel(
        renderable,
        title=title,
        title_align="left",
        border_style=agent_style,
        width=TOOL_WIDTH,
        padding=(0, 1),
    )
    console.print(Padding(panel, (0, 0, 0, TOOL_INDENT)))


def print_usage_compact(model_name: str, cost: float, latency: float) -> None:
    """Print compact model usage info on one line."""
    if not print_stdout:
        return
    console.print(f"  [dim]⏱  {model_name}  |  ${cost:.4f}  |  {latency:.1f}s[/]")
