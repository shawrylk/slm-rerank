"""Unit tests for semantic AST and symbol-aware chunker."""

import tempfile
from pathlib import Path
from lfm_rerank.chunker import chunk_file, prepare_candidates, estimate_tokens


def test_python_ast_chunking():
    py_code = (
        "import os\n"
        "\n"
        "def authenticate_user(token: str) -> bool:\n"
        "    if not token:\n"
        "        return False\n"
        "    return token == 'secret'\n"
        "\n"
        "class SessionManager:\n"
        "    def __init__(self):\n"
        "        self.active = {}\n"
        "\n"
        "    def close_all(self):\n"
        "        self.active.clear()\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(py_code)
        f_path = f.name

    try:
        chunks = chunk_file(f_path)
        assert len(chunks) >= 2
        symbols = [c.symbol for c in chunks if c.symbol]
        assert any("authenticate_user" in s for s in symbols)
        assert any("SessionManager" in s for s in symbols)

        # Verify physical line boundaries
        auth_chunk = next(c for c in chunks if "authenticate_user" in (c.symbol or ""))
        assert auth_chunk.start_line == 3
        assert auth_chunk.end_line == 6
        assert "3: def authenticate_user" in auth_chunk.content
    finally:
        Path(f_path).unlink(missing_ok=True)


def test_typescript_symbol_chunking():
    ts_code = (
        "import { Request, Response } from 'express';\n"
        "\n"
        "export function authMiddleware(req: Request, res: Response, next: any) {\n"
        "    const authHeader = req.headers.authorization;\n"
        "    if (!authHeader) return res.status(401).send();\n"
        "    next();\n"
        "}\n"
        "\n"
        "export const trigger = defineTrigger({\n"
        "    route: '/api/v1/work-types'\n"
        "});\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".ts", delete=False) as f:
        f.write(ts_code)
        f_path = f.name

    try:
        chunks = chunk_file(f_path)
        assert len(chunks) >= 1
        symbols = [c.symbol for c in chunks if c.symbol]
        assert any("authMiddleware" in s for s in symbols)
        auth_chunk = next(c for c in chunks if "authMiddleware" in (c.symbol or ""))
        assert auth_chunk.start_line == 3
        assert "3: export function authMiddleware" in auth_chunk.content
    finally:
        Path(f_path).unlink(missing_ok=True)


def test_prepare_candidates_mix():
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("line 1\nline 2\n")
        f_path = f.name

    try:
        raw_text = "raw candidate code\nwith two lines"
        chunks = prepare_candidates([f_path, raw_text])
        assert len(chunks) == 2
        assert chunks[0].file_path is not None
        assert chunks[1].file_path is None
    finally:
        Path(f_path).unlink(missing_ok=True)


def test_typescript_test_file_chunking_with_multibyte():
    ts_code = (
        "import { describe, it, expect } from 'vitest';\n"
        "const VIEWER_ROW = { role: 'viewer' };\n"
        "describe('buildings', () => {\n"
        "  it('tests japanese multibyte name', () => {\n"
        "    const item = { name: 'A棟', type: '土工事' };\n"
        "    expect(item.name).toBe('A棟');\n"
        "  });\n"
        "});\n"
        "const ENTRY_FIXTURE = { test: true };\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".test.ts", delete=False, encoding="utf-8") as f:
        f.write(ts_code)
        f_path = f.name

    try:
        chunks = chunk_file(f_path)
        assert len(chunks) >= 2
        for c in chunks:
            assert c.is_test is True
            assert 1 <= c.start_line <= c.end_line <= 9
        symbols = [c.symbol for c in chunks if c.symbol]
        assert any("VIEWER_ROW" in s for s in symbols)
        assert any("ENTRY_FIXTURE" in s for s in symbols)
    finally:
        Path(f_path).unlink(missing_ok=True)

