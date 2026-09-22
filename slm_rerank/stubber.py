"""
AST Ghost Stub Generation for Codebase Reranking.
Provides concise context skeletons for frontier models: keeps imports, interfaces,
and the target chunk while collapsing non-relevant function bodies into single-line stubs.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional


def generate_ghost_stub(
    file_path: str,
    chunk: Any,
    content: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate an AST ghost stub of the enclosing file around the targeted chunk.
    """
    start_line = getattr(chunk, "start_line", 1)
    end_line = getattr(chunk, "end_line", 1)

    if content is None:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception:
            return {
                "stub": getattr(chunk, "content", ""),
                "folded_lines": 0,
                "original_lines": 0,
            }

    lines = content.split("\n")
    total_lines = len(lines)
    result_lines = []
    folded_lines = 0

    # Function/class patterns for Python, TS, JS, Go, Rust
    func_pattern = re.compile(
        r"^\s*(export\s+)?(async\s+)?(def|class|function\*?)\s+([a-zA-Z0-9_$]+)"
    )

    in_foldable = False
    block_start_line = -1
    brace_depth = 0
    python_indent = -1
    is_python = file_path.endswith(".py")
    block_header = ""

    for i, line in enumerate(lines):
        line_num = i + 1

        # Inside target chunk: always output verbatim
        if start_line <= line_num <= end_line:
            in_foldable = False
            result_lines.append(f"{line_num}: {line}")
            continue

        # Keep top-of-file imports and types
        if line_num < start_line and re.match(
            r"^\s*(import\s|from\s+|export\s+(type|interface)\s|const\s+.*=\s*require\()", line
        ):
            result_lines.append(f"{line_num}: {line}")
            continue

        if is_python:
            # Python indentation-based folding
            m = re.match(r"^(\s*)(def|class)\s+([a-zA-Z0-9_]+)", line)
            if m and not in_foldable:
                in_foldable = True
                block_start_line = line_num
                python_indent = len(m.group(1))
                block_header = line.strip()
                continue

            if in_foldable:
                current_indent = len(line) - len(line.lstrip())
                if line.strip() and current_indent <= python_indent:
                    # Block finished
                    count = line_num - block_start_line
                    folded_lines += count
                    result_lines.append(f"{block_start_line}: {block_header} ... [folded {count} lines]")
                    in_foldable = False
                    result_lines.append(f"{line_num}: {line}")
                continue
        else:
            # Brace-based folding
            m = func_pattern.match(line)
            if m and not in_foldable and "{" in line:
                opens = line.count("{")
                closes = line.count("}")
                if opens > closes:
                    in_foldable = True
                    block_start_line = line_num
                    brace_depth = opens - closes
                    block_header = line[: line.find("{")].strip()
                    continue

            if in_foldable:
                brace_depth += line.count("{") - line.count("}")
                if brace_depth <= 0:
                    count = line_num - block_start_line
                    folded_lines += count
                    result_lines.append(f"{block_start_line}: {block_header} {{ /* ... [folded {count} lines] ... */ }}")
                    in_foldable = False
                continue

        # Keep type/interface lines outside target chunk
        if re.match(r"^\s*(export\s+)?(type|interface|enum)\s+", line):
            result_lines.append(f"{line_num}: {line}")
        elif not line.strip() and result_lines and not result_lines[-1].endswith(": "):
            result_lines.append(f"{line_num}: ")

    if in_foldable:
        count = total_lines - block_start_line
        folded_lines += count
        result_lines.append(f"{block_start_line}: {block_header} ... [folded {count} lines]")

    return {
        "stub": "\n".join(result_lines),
        "folded_lines": folded_lines,
        "original_lines": total_lines,
    }
