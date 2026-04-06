import os
import textwrap

from app.agents.trace import TraceCollector


def test_init():
    # get the absolute path of the project root
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    # build the path relative to the project root
    code_example_dir = os.path.join(project_root, "code_example/")

    # get all the files in the code_example_dir
    files = os.listdir(code_example_dir)
    for file in files:
        file_path = os.path.join(code_example_dir, file)
        if os.path.isfile(file_path):
            trace_collector = TraceCollector(file_path)
            assert trace_collector is not None


def test_collect_trace_ignores_unknown_blocks(tmp_path):
    source = textwrap.dedent(
        """
        #include <stdio.h>
        int f() {
            fprintf(stderr, "enter f 1\\n");
            int x = 0;
            x += 1;
            fprintf(stderr, "exit f 1\\n");
            return x;
        }
        """
    ).strip()

    test_file = tmp_path / "trace_unknown.c"
    test_file.write_text(source, encoding="utf-8")

    collector = TraceCollector(str(test_file))
    trace = "enter f 1\nenter f 9999\n"

    new_lines, _target_cov, _summary = collector.collect_trace(trace)
    assert new_lines > 0
