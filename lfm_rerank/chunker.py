"""Tree-Sitter and Syntax-Aware AST Code Chunker for lfm-rerank.

Extracts exact physical node spans for functions, methods, classes, and route handlers
across Python, TypeScript, JavaScript, C/C++, Rust, and Go using tree-sitter ASTs.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .models import CandidateChunk

# Lazy tree-sitter language cache
_LANGUAGES: Dict[str, Any] = {}


def is_test_path(path_str: str) -> bool:
    """Determine if a file path belongs to a test or specification suite."""
    p = path_str.lower()
    return any(
        term in p
        for term in [
            ".test.",
            "_test.",
            ".spec.",
            "_spec.",
            "/test/",
            "/tests/",
            "__tests__",
            "/testing/",
        ]
    )


def is_binary_file(filepath: Path) -> bool:
    """Check if file appears to be binary by sampling the first 2048 bytes."""
    try:
        with open(filepath, "rb") as f:
            chunk = f.read(2048)
            if b"\x00" in chunk:
                return True
            chunk.decode("utf-8")
            return False
    except (UnicodeDecodeError, OSError):
        return True


def estimate_tokens(text: str) -> int:
    """Fast approximation of token count (~3.6 characters per token)."""
    return max(1, int(len(text) / 3.6))


def format_lines_with_numbers(lines: Sequence[str], start_line: int = 1) -> str:
    """Format lines with physical line numbers for anchor-based citation."""
    formatted = []
    for i, line in enumerate(lines, start=start_line):
        line_content = line.rstrip("\r\n")
        formatted.append(f"{i}: {line_content}")
    return "\n".join(formatted)


def compute_chunk_hash(content: str) -> str:
    """Compute SHA-256 hash of chunk content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def get_tree_sitter_language(ext: str) -> Optional[Any]:
    """Retrieve or initialize tree-sitter Language for given file extension."""
    if ext in _LANGUAGES:
        return _LANGUAGES[ext]

    try:
        import tree_sitter
        lang = None
        if ext == ".py":
            import tree_sitter_python
            lang = tree_sitter.Language(tree_sitter_python.language())
        elif ext in {".ts"}:
            import tree_sitter_typescript
            lang = tree_sitter.Language(tree_sitter_typescript.language_typescript())
        elif ext in {".tsx"}:
            import tree_sitter_typescript
            lang = tree_sitter.Language(tree_sitter_typescript.language_tsx())
        elif ext in {".js", ".jsx", ".mjs", ".cjs"}:
            import tree_sitter_javascript
            lang = tree_sitter.Language(tree_sitter_javascript.language())
        elif ext in {".c", ".h"}:
            import tree_sitter_c
            lang = tree_sitter.Language(tree_sitter_c.language())
        elif ext in {".cpp", ".cc", ".cxx", ".hpp", ".hxx"}:
            import tree_sitter_cpp
            lang = tree_sitter.Language(tree_sitter_cpp.language())
        elif ext == ".rs":
            import tree_sitter_rust
            lang = tree_sitter.Language(tree_sitter_rust.language())
        elif ext == ".go":
            import tree_sitter_go
            lang = tree_sitter.Language(tree_sitter_go.language())

        if lang is not None:
            _LANGUAGES[ext] = lang
        return lang
    except Exception:
        return None


def get_tree_sitter_parser(ext: str) -> Optional[Any]:
    """Retrieve fresh tree-sitter parser instance for given file extension."""
    lang = get_tree_sitter_language(ext)
    if lang is None:
        return None
    try:
        import tree_sitter
        return tree_sitter.Parser(lang)
    except Exception:
        return None


def extract_node_symbol(node: Any, raw_bytes: bytes) -> Optional[str]:
    """Extract symbol name from AST node safely with byte-bounds checks."""
    try:
        # Direct name field
        name_node = node.child_by_field_name("name")
        if name_node:
            sb, eb = int(name_node.start_byte), int(name_node.end_byte)
            if 0 <= sb < eb <= len(raw_bytes):
                return raw_bytes[sb:eb].decode("utf-8", errors="replace")

        # Declarator field (C/C++, JS)
        decl_node = node.child_by_field_name("declarator")
        if decl_node:
            inner_name = decl_node.child_by_field_name("name") or decl_node.child_by_field_name("declarator")
            if inner_name:
                sb, eb = int(inner_name.start_byte), int(inner_name.end_byte)
                if 0 <= sb < eb <= len(raw_bytes):
                    return raw_bytes[sb:eb].decode("utf-8", errors="replace")
            sb, eb = int(decl_node.start_byte), int(decl_node.end_byte)
            if 0 <= sb < eb <= len(raw_bytes):
                return raw_bytes[sb:eb].decode("utf-8", errors="replace").split("(")[0].strip()

        # Export statement unwrapping
        if node.type == "export_statement":
            for child in list(node.named_children):
                if child.type in {
                    "function_declaration",
                    "class_declaration",
                    "lexical_declaration",
                    "variable_declaration",
                    "type_alias_declaration",
                    "interface_declaration",
                }:
                    return extract_node_symbol(child, raw_bytes)

        # Lexical / Variable declaration
        if node.type in {"lexical_declaration", "variable_declaration"}:
            for child in list(node.named_children):
                if child.type == "variable_declarator":
                    name = child.child_by_field_name("name")
                    if name:
                        sb, eb = int(name.start_byte), int(name.end_byte)
                        if 0 <= sb < eb <= len(raw_bytes):
                            return raw_bytes[sb:eb].decode("utf-8", errors="replace")

        # Fallback to first identifier child
        for child in list(node.named_children):
            if child.type in {"identifier", "type_identifier", "field_identifier"}:
                sb, eb = int(child.start_byte), int(child.end_byte)
                if 0 <= sb < eb <= len(raw_bytes):
                    return raw_bytes[sb:eb].decode("utf-8", errors="replace")
    except Exception:
        return None

    return None


TARGET_AST_NODE_TYPES = {
    # Python
    "function_definition",
    "class_definition",
    "decorated_definition",
    # TypeScript / JavaScript
    "function_declaration",
    "class_declaration",
    "interface_declaration",
    "type_alias_declaration",
    "method_definition",
    "export_statement",
    "lexical_declaration",
    # C / C++
    "function_definition",
    "class_specifier",
    "struct_specifier",
    # Rust
    "function_item",
    "impl_item",
    "struct_item",
    "enum_item",
    "trait_item",
    # Go
    "function_declaration",
    "method_declaration",
    "type_declaration",
}


def chunk_with_treesitter(
    file_path: Path,
    raw_content: str,
    lines: List[str],
    parser: Any,
) -> List[CandidateChunk]:
    """Extract semantic AST chunks using tree-sitter concrete syntax trees.

    Decouples raw AST property traversal into plain Python tuples before creating
    CandidateChunk objects and applies bounds-clamping to prevent cursor invalidation
    and negative/overflow slice indices.
    """
    total_lines = len(lines)
    if total_lines == 0:
        return []

    raw_bytes = raw_content.encode("utf-8")
    tree = parser.parse(raw_bytes)
    root = tree.root_node
    is_test = is_test_path(str(file_path))

    # Precompute line byte offsets for fast, exact line mapping
    import bisect
    line_starts = [0]
    for idx, b in enumerate(raw_bytes):
        if b == 10:  # '\n'
            line_starts.append(idx + 1)

    # Phase 1: Decoupled AST traversal into plain Python primitives
    extracted_spans: List[Tuple[int, int, str]] = []
    root_children = list(root.named_children)

    for node in root_children:
        if node.type in TARGET_AST_NODE_TYPES:
            try:
                # Compute line numbers from exact byte offsets to avoid py-tree-sitter Point.row bitflag bugs
                sb = int(node.start_byte)
                eb = max(sb, int(node.end_byte) - 1)
                raw_sl = bisect.bisect_right(line_starts, sb)
                raw_el = bisect.bisect_right(line_starts, eb)
                sl = max(1, min(total_lines, raw_sl))
                el = max(sl, min(total_lines, raw_el))
                symbol = extract_node_symbol(node, raw_bytes) or node.type
                extracted_spans.append((sl, el, symbol))
            except Exception:
                continue

    # Free AST references from memory before allocating candidate chunks
    del root_children
    del root
    del tree

    # Phase 2: Materialize candidate chunks and compute module scopes
    chunks: List[CandidateChunk] = []
    covered_lines = set()

    for start_line, end_line, symbol in extracted_spans:
        # If node is massive (e.g. > 180 lines), break down into sliding windows preserving symbol
        if end_line - start_line > 180:
            for sub_start in range(start_line, end_line + 1, 100):
                sub_end = min(end_line, sub_start + 120)
                chunk_lines = lines[sub_start - 1 : sub_end]
                content = format_lines_with_numbers(chunk_lines, sub_start)
                chunk_id = f"{file_path}:{sub_start}-{sub_end}:{symbol}"
                sub_sb = line_starts[sub_start - 1]
                sub_eb = line_starts[sub_end] if sub_end < len(line_starts) else len(raw_bytes)
                chunks.append(
                    CandidateChunk(
                        id=chunk_id,
                        file_path=str(file_path),
                        start_line=sub_start,
                        end_line=sub_end,
                        start_byte=sub_sb,
                        end_byte=sub_eb,
                        symbol=f"{symbol} (part {sub_start}-{sub_end})",
                        content=content,
                        token_est=estimate_tokens(content),
                        content_hash=compute_chunk_hash(content),
                        is_test=is_test,
                    )
                )
                if sub_end >= end_line:
                    break
        else:
            chunk_lines = lines[start_line - 1 : end_line]
            content = format_lines_with_numbers(chunk_lines, start_line)
            chunk_id = f"{file_path}:{start_line}-{end_line}:{symbol}"
            c_sb = line_starts[start_line - 1]
            c_eb = line_starts[end_line] if end_line < len(line_starts) else len(raw_bytes)
            chunks.append(
                CandidateChunk(
                    id=chunk_id,
                    file_path=str(file_path),
                    start_line=start_line,
                    end_line=end_line,
                    start_byte=c_sb,
                    end_byte=c_eb,
                    symbol=symbol,
                    content=content,
                    token_est=estimate_tokens(content),
                    content_hash=compute_chunk_hash(content),
                    is_test=is_test,
                )
            )

        for ln in range(start_line, end_line + 1):
            covered_lines.add(ln)

    # Collect uncovered module-level blocks (e.g. imports, route bindings, top-level setup)
    uncovered_start = None
    for i in range(1, total_lines + 1):
        if i not in covered_lines:
            if uncovered_start is None:
                uncovered_start = i
        else:
            if uncovered_start is not None:
                block_len = i - uncovered_start
                if block_len >= 3:
                    u_lines = lines[uncovered_start - 1 : i - 1]
                    if any(l.strip() and not l.strip().startswith(("#", "//", "/*", "*")) for l in u_lines):
                        u_content = format_lines_with_numbers(u_lines, uncovered_start)
                        u_sb = line_starts[uncovered_start - 1]
                        u_eb = line_starts[i - 1] if (i - 1) < len(line_starts) else len(raw_bytes)
                        chunks.append(
                            CandidateChunk(
                                id=f"{file_path}:{uncovered_start}-{i-1}:module_scope",
                                file_path=str(file_path),
                                start_line=uncovered_start,
                                end_line=i - 1,
                                start_byte=u_sb,
                                end_byte=u_eb,
                                symbol="module_scope",
                                content=u_content,
                                token_est=estimate_tokens(u_content),
                                content_hash=compute_chunk_hash(u_content),
                                is_test=is_test,
                            )
                        )
                uncovered_start = None

    if uncovered_start is not None and (total_lines - uncovered_start >= 3):
        u_lines = lines[uncovered_start - 1 : total_lines]
        if any(l.strip() and not l.strip().startswith(("#", "//", "/*", "*")) for l in u_lines):
            u_content = format_lines_with_numbers(u_lines, uncovered_start)
            u_sb = line_starts[uncovered_start - 1]
            u_eb = len(raw_bytes)
            chunks.append(
                CandidateChunk(
                    id=f"{file_path}:{uncovered_start}-{total_lines}:module_scope",
                    file_path=str(file_path),
                    start_line=uncovered_start,
                    end_line=total_lines,
                    start_byte=u_sb,
                    end_byte=u_eb,
                    symbol="module_scope",
                    content=u_content,
                    token_est=estimate_tokens(u_content),
                    content_hash=compute_chunk_hash(u_content),
                    is_test=is_test,
                )
            )

    return sorted(chunks, key=lambda c: c.start_line)


def chunk_sliding_window(
    file_path: Optional[Path],
    lines: List[str],
    max_chunk_lines: int = 120,
    overlap_lines: int = 30,
) -> List[CandidateChunk]:
    """Fallback sliding window chunker for unstructured documents."""
    chunks: List[CandidateChunk] = []
    total_lines = len(lines)
    if total_lines == 0:
        return []

    cur_byte = 0
    line_byte_offsets = [0]
    for line in lines:
        cur_byte += len(line.encode("utf-8"))
        line_byte_offsets.append(cur_byte)

    step = max(1, max_chunk_lines - overlap_lines)
    file_str = str(file_path) if file_path else "raw_content"
    is_test = is_test_path(file_str)

    for start_idx in range(0, total_lines, step):
        end_idx = min(total_lines, start_idx + max_chunk_lines)
        start_line = start_idx + 1
        end_line = end_idx
        chunk_lines = lines[start_idx:end_idx]
        content = format_lines_with_numbers(chunk_lines, start_line)
        sw_sb = line_byte_offsets[start_line - 1]
        sw_eb = line_byte_offsets[end_line] if end_line < len(line_byte_offsets) else cur_byte

        chunks.append(
            CandidateChunk(
                id=f"{file_str}:{start_line}-{end_line}",
                file_path=file_str if file_path else None,
                start_line=start_line,
                end_line=end_line,
                start_byte=sw_sb,
                end_byte=sw_eb,
                symbol=None,
                content=content,
                token_est=estimate_tokens(content),
                content_hash=compute_chunk_hash(content),
                is_test=is_test,
            )
        )
        if end_idx >= total_lines:
            break

    return chunks


def chunk_file(file_path: Union[str, Path]) -> List[CandidateChunk]:
    """Semantically chunk a file using tree-sitter ASTs or sliding window fallback."""
    path = Path(file_path).resolve()
    if not path.is_file():
        return []

    if is_binary_file(path):
        return []

    try:
        with open(path, "r", encoding="utf-8", newline="", errors="replace") as f:
            raw_content = f.read()
            lines = raw_content.splitlines(keepends=True)
    except Exception:
        return []

    if not lines:
        return []

    ext = path.suffix.lower()
    ts_parser = get_tree_sitter_parser(ext)
    if ts_parser:
        try:
            ts_chunks = chunk_with_treesitter(path, raw_content, lines, ts_parser)
            if ts_chunks:
                return ts_chunks
        except Exception:
            pass

    return chunk_sliding_window(path, lines, max_chunk_lines=120, overlap_lines=30)


def prepare_candidates(
    candidate_inputs: Sequence[Union[str, Path]],
) -> List[CandidateChunk]:
    """Process file paths or text candidate inputs into semantic chunks."""
    all_chunks: List[CandidateChunk] = []

    for item in candidate_inputs:
        item_str = str(item).strip()
        if not item_str:
            continue

        p = Path(item_str)
        if p.exists() and p.is_file():
            all_chunks.extend(chunk_file(p))
        else:
            cwd_p = Path.cwd() / item_str
            if cwd_p.exists() and cwd_p.is_file():
                all_chunks.extend(chunk_file(cwd_p))
            else:
                raw_lines = item_str.splitlines(keepends=True)
                all_chunks.extend(chunk_sliding_window(None, raw_lines, max_chunk_lines=120, overlap_lines=30))

    return all_chunks
