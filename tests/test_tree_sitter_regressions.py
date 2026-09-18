"""Tree-sitter AST Chunker Regression Suite.

Verifies robust parsing, physical line extraction, and exact byte offset alignment
across all required edge cases:
- Multibyte UTF-8 characters and non-ASCII identifiers/comments.
- Emojis in strings and comments.
- CRLF line endings (\\r\\n) vs LF.
- JavaScript/TypeScript template literals with embedded braces: `${foo({bar: 1})}`.
- Regex literals with braces: `/[{}]/g`.
- BOM-prefixed files.
- Files with no trailing newline.
- Python triple-quoted docstrings with braces and newlines.
- Rust raw strings and Go backtick strings.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List

import pytest

from slm_rerank.chunker import chunk_file
from slm_rerank.models import CandidateChunk


def assert_chunks_align_with_physical_file(chunks: List[CandidateChunk], file_path: Path) -> None:
    """Rigorous assertion: extracted_lines == physical_file_lines[start-1:end]

    and byte offsets align exactly with the physical file on disk.
    """
    assert len(chunks) > 0, f"No chunks extracted from {file_path}"
    physical_bytes = file_path.read_bytes()
    with open(file_path, "r", encoding="utf-8", newline="") as f:
        physical_file_lines = f.readlines()

    # Precalculate physical line byte starts
    line_starts = [0]
    for idx, b in enumerate(physical_bytes):
        if b == 10:  # '\\n'
            line_starts.append(idx + 1)

    for chunk in chunks:
        # 1. Physical line range validity
        assert 1 <= chunk.start_line <= chunk.end_line <= len(physical_file_lines), (
            f"Invalid line range {chunk.start_line}-{chunk.end_line} for total {len(physical_file_lines)} lines"
        )

        # 2. extracted_lines == physical_file_lines[start-1:end]
        extracted_lines = [
            line.split(": ", 1)[1] if ": " in line else line
            for line in chunk.content.splitlines()
        ]
        expected_lines = [
            line.rstrip("\r\n")
            for line in physical_file_lines[chunk.start_line - 1 : chunk.end_line]
        ]
        assert extracted_lines == expected_lines, (
            f"Line content mismatch in {chunk.id}:\nExtracted: {extracted_lines}\nExpected: {expected_lines}"
        )

        # 3. Exact byte offset alignment
        assert chunk.start_byte is not None and chunk.end_byte is not None, (
            f"Byte offsets missing on chunk {chunk.id}"
        )
        expected_start_byte = line_starts[chunk.start_line - 1]
        expected_end_byte = (
            line_starts[chunk.end_line] if chunk.end_line < len(line_starts) else len(physical_bytes)
        )

        assert chunk.start_byte == expected_start_byte, (
            f"Start byte mismatch in {chunk.id}: {chunk.start_byte} != {expected_start_byte}"
        )
        assert chunk.end_byte == expected_end_byte, (
            f"End byte mismatch in {chunk.id}: {chunk.end_byte} != {expected_end_byte}"
        )

        # 4. Decoded physical byte slice matches extracted lines exactly
        byte_slice = physical_bytes[chunk.start_byte : chunk.end_byte]
        slice_lines = [
            line.rstrip("\r\n")
            for line in byte_slice.decode("utf-8", errors="replace").splitlines()
        ]
        assert slice_lines == extracted_lines, (
            f"Byte slice decode mismatch in {chunk.id}:\nDecoded: {slice_lines}\nExtracted: {extracted_lines}"
        )


def test_multibyte_utf8_identifiers_and_comments():
    """Case 1: Multibyte UTF-8 characters and non-ASCII identifiers/comments in Python."""
    code = (
        "# 共通ユーティリティモジュール 🌸\n"
        "def 計算_消費税(価格: int, 税率: float = 0.10) -> float:\n"
        "    # 日本語の計算コメント\n"
        "    return 価格 * (1.0 + 税率)\n"
        "\n"
        "class 会計管理システム:\n"
        "    def __init__(self, 初期残高: int):\n"
        "        self.残高 = 初期残高\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        f_path = Path(f.name)

    try:
        chunks = chunk_file(f_path)
        assert len(chunks) >= 2
        symbols = [c.symbol for c in chunks if c.symbol]
        assert any("計算_消費税" in s for s in symbols)
        assert any("会計管理システム" in s for s in symbols)
        assert_chunks_align_with_physical_file(chunks, f_path)
    finally:
        f_path.unlink(missing_ok=True)


def test_emojis_in_strings_and_comments():
    """Case 2: Emojis in strings and comments (multibyte 4-byte UTF-8 sequences)."""
    code = (
        "import sys\n"
        "\n"
        "def launch_rocket(target: str) -> bool:\n"
        "    # 🚀 Rocket launcher with status emojis 🔥 ✨\n"
        "    log_msg = f'Launching to {target} 🌌 🛸 🛰️'\n"
        "    print(log_msg)\n"
        "    return True\n"
        "\n"
        "def celebratory_banner():\n"
        "    \"\"\"Return celebratory confetti 🎉 🎊 🥳 🎈\"\"\"\n"
        "    return '🎉 Success! 🍾'\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        f_path = Path(f.name)

    try:
        chunks = chunk_file(f_path)
        assert len(chunks) >= 2
        symbols = [c.symbol for c in chunks if c.symbol]
        assert any("launch_rocket" in s for s in symbols)
        assert any("celebratory_banner" in s for s in symbols)
        assert_chunks_align_with_physical_file(chunks, f_path)
    finally:
        f_path.unlink(missing_ok=True)


def test_crlf_line_endings_vs_lf():
    """Case 3: CRLF line endings (\\r\\n) vs standard LF (\\n)."""
    # CRLF file
    crlf_content = (
        b"def service_heartbeat() -> str:\r\n"
        b"    # Windows CRLF line endings\r\n"
        b"    return \"PONG\"\r\n"
        b"\r\n"
        b"def service_shutdown():\r\n"
        b"    print(\"Stopping...\")\r\n"
    )
    with tempfile.NamedTemporaryFile("wb", suffix=".py", delete=False) as f:
        f.write(crlf_content)
        f_path = Path(f.name)

    try:
        chunks = chunk_file(f_path)
        assert len(chunks) >= 2
        assert_chunks_align_with_physical_file(chunks, f_path)
    finally:
        f_path.unlink(missing_ok=True)

    # LF counterpart
    lf_content = (
        "def service_heartbeat() -> str:\n"
        "    # Linux LF line endings\n"
        "    return \"PONG\"\n"
        "\n"
        "def service_shutdown():\n"
        "    print(\"Stopping...\")\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(lf_content)
        f_path_lf = Path(f.name)

    try:
        chunks_lf = chunk_file(f_path_lf)
        assert len(chunks_lf) >= 2
        assert_chunks_align_with_physical_file(chunks_lf, f_path_lf)
    finally:
        f_path_lf.unlink(missing_ok=True)


def test_typescript_template_literals_with_embedded_braces():
    """Case 4: JavaScript/TypeScript template literals with embedded braces: ${foo({bar: 1})}."""
    ts_code = (
        "export function generateGreeting(options: { prefix?: string; user: { name: string } }): string {\n"
        "    const formatUser = (u: { name: string }) => ({ formatted: u.name.toUpperCase() });\n"
        "    return `${options.prefix ?? 'Hello'}, ${formatUser({ name: options.user.name }).formatted}!`;\n"
        "}\n"
        "\n"
        "export const computeNested = (val: number) => {\n"
        "    return `Result: ${((x) => ({ count: x * 2 }))(val).count} units`;\n"
        "};\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".ts", delete=False, encoding="utf-8") as f:
        f.write(ts_code)
        f_path = Path(f.name)

    try:
        chunks = chunk_file(f_path)
        assert len(chunks) >= 2
        symbols = [c.symbol for c in chunks if c.symbol]
        assert any("generateGreeting" in s for s in symbols)
        assert any("computeNested" in s for s in symbols)
        assert_chunks_align_with_physical_file(chunks, f_path)
    finally:
        f_path.unlink(missing_ok=True)


def test_javascript_regex_literals_with_braces():
    """Case 5: Regex literals with braces: /[{}]/g in JavaScript/TypeScript."""
    js_code = (
        "function stripBraces(inputString) {\n"
        "    const bracePattern = /[{}]/g;\n"
        "    return inputString.replace(bracePattern, '');\n"
        "}\n"
        "\n"
        "function validateTokenFormat(token) {\n"
        "    const regexWithQuantifier = /^[a-z]{3,8}-[0-9]{4}$/;\n"
        "    return regexWithQuantifier.test(token);\n"
        "}\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
        f.write(js_code)
        f_path = Path(f.name)

    try:
        chunks = chunk_file(f_path)
        assert len(chunks) >= 2
        symbols = [c.symbol for c in chunks if c.symbol]
        assert any("stripBraces" in s for s in symbols)
        assert any("validateTokenFormat" in s for s in symbols)
        assert_chunks_align_with_physical_file(chunks, f_path)
    finally:
        f_path.unlink(missing_ok=True)


def test_bom_prefixed_files():
    """Case 6: BOM-prefixed files (UTF-8 BOM: \\xef\\xbb\\xbf)."""
    bom_content = (
        b"\xef\xbb\xbfdef parse_bom_file():\n"
        b"    # File starts with byte order mark\n"
        b"    return {'bom': True}\n"
        b"\n"
        b"def secondary_routine():\n"
        b"    return 42\n"
    )
    with tempfile.NamedTemporaryFile("wb", suffix=".py", delete=False) as f:
        f.write(bom_content)
        f_path = Path(f.name)

    try:
        chunks = chunk_file(f_path)
        assert len(chunks) >= 2
        assert_chunks_align_with_physical_file(chunks, f_path)
    finally:
        f_path.unlink(missing_ok=True)


def test_files_with_no_trailing_newline():
    """Case 7: Files with no trailing newline at EOF."""
    no_newline_py = (
        "def compute_sum(a: int, b: int) -> int:\n"
        "    # No trailing newline after return statement\n"
        "    return a + b"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(no_newline_py)
        f_path = Path(f.name)

    try:
        chunks = chunk_file(f_path)
        assert len(chunks) >= 1
        assert_chunks_align_with_physical_file(chunks, f_path)
        c = chunks[0]
        assert c.start_line == 1
        assert c.end_line == 3
    finally:
        f_path.unlink(missing_ok=True)


def test_python_triple_quoted_docstrings_with_braces_and_newlines():
    """Case 8: Python triple-quoted docstrings with braces, colons, and newlines."""
    code = (
        "def process_schema_payload(data: dict) -> bool:\n"
        "    \"\"\"\n"
        "    Validates schema against OpenAPI spec:\n"
        "    {\n"
        "        \"type\": \"object\",\n"
        "        \"properties\": {\n"
        "            \"id\": {\"type\": \"string\"}\n"
        "        },\n"
        "        \"required\": [\"id\"]\n"
        "    }\n"
        "    Returns True if compliant.\n"
        "    \"\"\"\n"
        "    return 'id' in data\n"
        "\n"
        "class ConfigParser:\n"
        "    '''\n"
        "    Multi-line single-quoted docstring with {braces: 123}\n"
        "    and formatting markers: {{ escaped }}\n"
        "    '''\n"
        "    def get_val(self):\n"
        "        return 1\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        f_path = Path(f.name)

    try:
        chunks = chunk_file(f_path)
        assert len(chunks) >= 2
        symbols = [c.symbol for c in chunks if c.symbol]
        assert any("process_schema_payload" in s for s in symbols)
        assert any("ConfigParser" in s for s in symbols)
        assert_chunks_align_with_physical_file(chunks, f_path)
    finally:
        f_path.unlink(missing_ok=True)


def test_rust_raw_strings_and_go_backtick_strings():
    """Case 9: Rust raw strings (r#\"...\"#) and Go backtick multi-line strings."""
    # Rust raw strings
    rs_code = (
        "pub fn get_embedded_schema() -> &'static str {\n"
        "    r#\"{\n"
        "      \"name\": \"test_suite\",\n"
        "      \"settings\": { \"enabled\": true }\n"
        "    }\"#\n"
        "}\n"
        "\n"
        "pub struct AppConfig {\n"
        "    pub port: u16,\n"
        "}\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".rs", delete=False, encoding="utf-8") as f:
        f.write(rs_code)
        f_path_rs = Path(f.name)

    try:
        chunks_rs = chunk_file(f_path_rs)
        assert len(chunks_rs) >= 2
        assert_chunks_align_with_physical_file(chunks_rs, f_path_rs)
    finally:
        f_path_rs.unlink(missing_ok=True)

    # Go backtick strings
    go_code = (
        "package main\n"
        "\n"
        "func GetJSONPayload() string {\n"
        "    return `{\n"
        "        \"route\": \"/api/v1/health\",\n"
        "        \"status\": 200\n"
        "    }`\n"
        "}\n"
        "\n"
        "func StartServer(port int) error {\n"
        "    return nil\n"
        "}\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".go", delete=False, encoding="utf-8") as f:
        f.write(go_code)
        f_path_go = Path(f.name)

    try:
        chunks_go = chunk_file(f_path_go)
        assert len(chunks_go) >= 2
        assert_chunks_align_with_physical_file(chunks_go, f_path_go)
    finally:
        f_path_go.unlink(missing_ok=True)


def test_nasty_combined_hazard_all_in_one():
    """Case 10: All hazards combined into a single file:

    BOM (\\xef\\xbb\\xbf) + CRLF (\\r\\n) + emojis + CJK identifiers +
    template literals with braces + regex braces + no trailing newline.
    Asserts exact byte-level and line-level reconstruction.
    """
    bom = b"\xef\xbb\xbf"
    code_text = (
        "// 共通認証モジュール 🌸 🚀\r\n"
        "export function 認証_ハンドラー(顧客_名前: string): boolean {\r\n"
        "    // Emojis and CJK in comments: 認証チェック 🔑 ✨\r\n"
        "    const regexPattern = /^[a-z0-9]{3,8}[{}]/g;\r\n"
        "    const formatUser = (name: string) => ({ formatted: `${name.toUpperCase()}` });\r\n"
        "    const debugMsg = `User: ${formatUser({ name: 顧客_名前 }).formatted} logged in 🔥`;\r\n"
        "    return regexPattern.test(顧客_名前);\r\n"
        "}\r\n"
        "\r\n"
        "export function 注文_処理(注文ID: number, 金額: number): string {\r\n"
        "    const payload = `${((x) => ({ res: `ID:${x * 2}` }))(注文ID).res} - 💰 ${金額}`;\r\n"
        "    const braceRegex = /[{}]/g;\r\n"
        "    return payload.replace(braceRegex, '');\r\n"
        "}"  # Notice: NO trailing newline at EOF!
    )
    file_bytes = bom + code_text.encode("utf-8")
    with tempfile.NamedTemporaryFile("wb", suffix=".ts", delete=False) as f:
        f.write(file_bytes)
        f_path = Path(f.name)

    try:
        chunks = chunk_file(f_path)
        assert len(chunks) >= 2, f"Expected at least 2 AST chunks, got {len(chunks)}"

        # Verify CJK symbol names were extracted by tree-sitter
        symbols = [c.symbol for c in chunks if c.symbol]
        assert any("認証_ハンドラー" in s for s in symbols)
        assert any("注文_処理" in s for s in symbols)

        # Assert exact byte-level and line-level reconstruction
        assert_chunks_align_with_physical_file(chunks, f_path)

        # Confirm the final chunk's physical bytes end at EOF without newline
        final_chunk = chunks[-1]
        assert final_chunk.end_byte == len(file_bytes)
        last_chunk_bytes = file_bytes[final_chunk.start_byte : final_chunk.end_byte]
        assert last_chunk_bytes.endswith(b"}")
        assert not last_chunk_bytes.endswith(b"\n")
        assert not last_chunk_bytes.endswith(b"\r")
    finally:
        f_path.unlink(missing_ok=True)

