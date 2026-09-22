"""Call-Graph & Type Context Stitching: 1-hop interface context under a hard token cap.

A semantic chunk is precise but lonely: a method body rarely names the type it
hangs off, the helpers it calls, or who calls it. This module reconstructs that
one hop of structure from the surrounding source and emits a compact header that
is prepended to the *scoring prompt only* -- physical citations, snippets and
content hashes are never mutated.

The assembled block is strictly capped at ``MAX_STITCH_TOKENS`` (150) tokens.
Sections are added greedily in priority order (enclosing type, then called
interfaces, then callers) and any section that would breach the cap is dropped
rather than truncated mid-signature.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from .chunker import estimate_tokens
from .models import CandidateChunk

# Hard ceiling on the stitched block. Never exceeded.
MAX_STITCH_TOKENS: int = 150

# Per-signature character clamp so one pathological declaration cannot eat the budget.
MAX_SIGNATURE_CHARS: int = 160

MAX_INTERFACES: int = 4
MAX_CALLERS: int = 3

STITCH_HEADER = "[Stitched Context]"

TYPE_KINDS = frozenset({"class", "struct", "interface", "enum", "trait", "impl", "namespace"})

# Keyword-led definitions across Python, TS/JS, Rust, Go, C++, Java.
_KEYWORD_DEF_RE = re.compile(
    r"^(?P<indent>[ \t]*)"
    r"(?:(?:export|public|private|protected|internal|static|final|abstract|async|inline|"
    r"virtual|explicit|extern|unsafe|declare|default)\s+)*"
    r"(?:pub(?:\([^)]*\))?\s+)?"
    r"(?P<kind>class|struct|interface|enum|trait|impl|namespace|type|typedef|def|fn|func|function)\s+"
    r"(?P<name>[A-Za-z_~][A-Za-z0-9_]*)"
)

# Brace-language function definitions without a leading keyword (C/C++ free functions,
# TS class methods): `uint32_t parse_token(const char * src) {`.
_BARE_FUNC_DEF_RE = re.compile(
    r"^(?P<indent>[ \t]*)"
    r"(?P<sig>[A-Za-z_][A-Za-z0-9_:<>,*&\[\]\s]*?"
    r"(?P<name>[A-Za-z_~][A-Za-z0-9_]*)\s*"
    r"\([^;]*\)\s*(?:const\s*)?(?:noexcept\s*)?(?:->[^{;]+)?)\s*\{\s*$"
)

_CALL_RE = re.compile(r"\b(?P<name>[A-Za-z_][A-Za-z0-9_]{2,})\s*\(")

# Control-flow keywords that look like calls but are not.
_NON_CALL_KEYWORDS = frozenset({
    "if", "for", "while", "switch", "catch", "return", "sizeof", "assert",
    "print", "printf", "def", "fn", "func", "function", "class", "struct",
    "and", "not", "or", "in", "is", "elif", "else", "do", "try", "with", "lambda",
})

# Line-number prefixes injected by the chunker (`  12: code`) are stripped before parsing.
_LINE_PREFIX_RE = re.compile(r"^\s*\d+:\s?")


class Definition(BaseModel):
    """A single definition site discovered in a source file."""

    line: int = Field(description="1-indexed line where the definition starts")
    end_line: int = Field(description="1-indexed line where the definition's block ends")
    indent: int = Field(default=0, description="Leading whitespace width of the definition line")
    kind: str = Field(default="function", description="class/struct/interface/enum/trait/impl/type/function")
    name: str = Field(default="", description="Declared symbol name")
    signature: str = Field(default="", description="Normalized single-line signature")

    @property
    def is_type(self) -> bool:
        return self.kind in TYPE_KINDS


class StitchedContext(BaseModel):
    """Assembled 1-hop context for a chunk, capped at ``MAX_STITCH_TOKENS``."""

    text: str = Field(default="", description="Prompt-ready context block (empty when nothing was found)")
    token_est: int = Field(default=0, description="Estimated token cost of `text`")
    enclosing_type: Optional[str] = Field(default=None, description="Signature of the enclosing type, if any")
    interfaces: List[str] = Field(default_factory=list, description="Signatures of 1-hop callees")
    callers: List[str] = Field(default_factory=list, description="Signatures of 1-hop callers")
    truncated: bool = Field(default=False, description="True if a section was dropped to respect the cap")

    @property
    def is_empty(self) -> bool:
        return not self.text


def strip_line_numbers(content: str) -> str:
    """Remove the chunker's `NN: ` display prefixes so chunk bodies parse as source."""
    return "\n".join(_LINE_PREFIX_RE.sub("", line) for line in content.splitlines())


def normalize_signature(raw: str) -> str:
    """Collapse a declaration to a single clamped line."""
    sig = " ".join(str(raw).split())
    sig = sig.rstrip("{ ").rstrip()
    if len(sig) > MAX_SIGNATURE_CHARS:
        sig = sig[: MAX_SIGNATURE_CHARS - 3].rstrip() + "..."
    return sig


def _block_end_line(lines: Sequence[str], start_idx: int, indent: int) -> int:
    """Resolve where a definition's block ends, by brace depth or by dedent."""
    opener_idx = None
    for probe in range(start_idx, min(start_idx + 3, len(lines))):
        if "{" in lines[probe]:
            opener_idx = probe
            break

    if opener_idx is not None:
        depth = 0
        for idx in range(opener_idx, len(lines)):
            depth += lines[idx].count("{") - lines[idx].count("}")
            if depth <= 0:
                return idx + 1
        return len(lines)

    # Indentation-scoped languages (Python): block ends at the next line that
    # dedents back to or past the definition's own indentation.
    for idx in range(start_idx + 1, len(lines)):
        line = lines[idx]
        if not line.strip():
            continue
        line_indent = len(line) - len(line.lstrip())
        if line_indent <= indent:
            return idx
    return len(lines)


def extract_definitions(source: str) -> List[Definition]:
    """Parse a source file into definition records using language-agnostic patterns."""
    lines = source.splitlines()
    definitions: List[Definition] = []

    for idx, line in enumerate(lines):
        if not line.strip() or line.lstrip().startswith(("//", "#", "*", "/*")):
            continue

        match = _KEYWORD_DEF_RE.match(line)
        kind: Optional[str] = None
        name = ""
        if match:
            kind = match.group("kind")
            name = match.group("name")
            indent = len(match.group("indent").expandtabs(4))
            if kind in ("def", "fn", "func", "function"):
                kind = "function"
        else:
            bare = _BARE_FUNC_DEF_RE.match(line)
            if not bare:
                continue
            name = bare.group("name")
            if name in _NON_CALL_KEYWORDS:
                continue
            kind = "function"
            indent = len(bare.group("indent").expandtabs(4))

        definitions.append(
            Definition(
                line=idx + 1,
                end_line=_block_end_line(lines, idx, indent),
                indent=indent,
                kind=kind or "function",
                name=name,
                signature=normalize_signature(line),
            )
        )

    return definitions


def find_enclosing_type(definitions: Sequence[Definition], start_line: int) -> Optional[Definition]:
    """Return the innermost type definition whose block contains ``start_line``."""
    enclosing: Optional[Definition] = None
    for d in definitions:
        if not d.is_type:
            continue
        if d.line < start_line <= d.end_line:
            if enclosing is None or d.indent >= enclosing.indent:
                enclosing = d
    return enclosing


def extract_called_identifiers(content: str) -> List[str]:
    """List identifiers invoked inside a chunk body, in first-appearance order."""
    body = strip_line_numbers(content)
    seen: set = set()
    calls: List[str] = []
    for m in _CALL_RE.finditer(body):
        name = m.group("name")
        if name in _NON_CALL_KEYWORDS or name in seen:
            continue
        seen.add(name)
        calls.append(name)
    return calls


def find_callers(
    definitions: Sequence[Definition],
    source_lines: Sequence[str],
    symbol: str,
    exclude_span: Tuple[int, int],
) -> List[Definition]:
    """Find definitions outside ``exclude_span`` whose body calls ``symbol``."""
    if not symbol or symbol == "module_scope":
        return []

    call_re = re.compile(r"\b" + re.escape(symbol) + r"\s*\(")
    start, end = exclude_span
    callers: List[Definition] = []
    seen: set = set()

    for idx, line in enumerate(source_lines):
        line_no = idx + 1
        if start <= line_no <= end:
            continue
        if not call_re.search(line):
            continue
        owner = None
        for d in definitions:
            if d.is_type or d.name == symbol:
                continue
            if d.line <= line_no <= d.end_line:
                if owner is None or d.indent >= owner.indent:
                    owner = d
        if owner is not None and owner.name not in seen:
            seen.add(owner.name)
            callers.append(owner)

    return callers


def stitch_chunk_context(
    chunk: CandidateChunk,
    source: Optional[str] = None,
    corpus: Optional[Dict[str, str]] = None,
    max_tokens: int = MAX_STITCH_TOKENS,
) -> StitchedContext:
    """Build the 1-hop context block for ``chunk`` within its hard token budget.

    ``source`` is the chunk's own file text; ``corpus`` optionally maps other
    file paths to their text so callee signatures can be resolved across files.
    """
    file_source = source
    if file_source is None and chunk.file_path:
        if corpus and chunk.file_path in corpus:
            file_source = corpus[chunk.file_path]
        else:
            try:
                file_source = Path(chunk.file_path).read_text(encoding="utf-8", errors="replace")
            except Exception:
                file_source = None

    if not file_source:
        return StitchedContext()

    source_lines = file_source.splitlines()
    definitions = extract_definitions(file_source)
    span = (chunk.start_line, chunk.end_line)

    # 1. Enclosing type / namespace.
    enclosing = find_enclosing_type(definitions, chunk.start_line)

    # 2. 1-hop callees defined in this file or elsewhere in the corpus.
    called = extract_called_identifiers(chunk.content)
    own_symbol = (chunk.symbol or "").split(" ")[0]
    definitions_by_name: Dict[str, Definition] = {}
    for d in definitions:
        if d.name and d.name not in definitions_by_name:
            definitions_by_name[d.name] = d

    external_definitions: Dict[str, Definition] = {}
    if corpus:
        for path, text in corpus.items():
            if chunk.file_path and path == chunk.file_path:
                continue
            for d in extract_definitions(text):
                if d.name and d.name not in external_definitions:
                    external_definitions[d.name] = d

    interfaces: List[str] = []
    for name in called:
        if name == own_symbol or len(interfaces) >= MAX_INTERFACES:
            continue
        target = definitions_by_name.get(name) or external_definitions.get(name)
        if target is None or (span[0] <= target.line <= span[1]):
            continue
        interfaces.append(target.signature)

    # 3. 1-hop callers of this chunk's own symbol.
    callers = [
        d.signature
        for d in find_callers(definitions, source_lines, own_symbol, span)[:MAX_CALLERS]
    ]

    return _assemble(enclosing, interfaces, callers, max_tokens)


def _assemble(
    enclosing: Optional[Definition],
    interfaces: Sequence[str],
    callers: Sequence[str],
    max_tokens: int,
) -> StitchedContext:
    """Greedily assemble sections in priority order without ever breaching the cap."""
    budget = max(0, int(max_tokens))
    lines: List[str] = [STITCH_HEADER]
    truncated = False

    kept_enclosing: Optional[str] = None
    kept_interfaces: List[str] = []
    kept_callers: List[str] = []

    def fits(candidate_lines: Sequence[str]) -> bool:
        return estimate_tokens("\n".join(candidate_lines)) <= budget

    if not fits(lines):
        return StitchedContext(truncated=True)

    if enclosing is not None:
        candidate = lines + [f"Enclosing type: {enclosing.signature}"]
        if fits(candidate):
            lines = candidate
            kept_enclosing = enclosing.signature
        else:
            truncated = True

    for sig in interfaces:
        candidate = lines + [f"Calls: {sig}"]
        if fits(candidate):
            lines = candidate
            kept_interfaces.append(sig)
        else:
            truncated = True
            break

    for sig in callers:
        candidate = lines + [f"Called by: {sig}"]
        if fits(candidate):
            lines = candidate
            kept_callers.append(sig)
        else:
            truncated = True
            break

    if len(lines) == 1:
        # Header only: nothing useful was found or nothing fit.
        return StitchedContext(truncated=truncated)

    text = "\n".join(lines)
    return StitchedContext(
        text=text,
        token_est=estimate_tokens(text),
        enclosing_type=kept_enclosing,
        interfaces=kept_interfaces,
        callers=kept_callers,
        truncated=truncated,
    )


def apply_context_stitching(
    chunks: Sequence[CandidateChunk],
    max_tokens: int = MAX_STITCH_TOKENS,
) -> Tuple[int, int]:
    """Attach stitched context to every chunk that yields one.

    Mutates ``chunk.stitched_context`` in place and returns
    ``(chunks_stitched, total_context_tokens)``. Source files are read once and
    shared as a corpus so cross-file callee signatures resolve.
    """
    corpus: Dict[str, str] = {}
    for chunk in chunks:
        if not chunk.file_path or chunk.file_path in corpus:
            continue
        try:
            corpus[chunk.file_path] = Path(chunk.file_path).read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

    stitched_count = 0
    total_tokens = 0
    for chunk in chunks:
        ctx = stitch_chunk_context(
            chunk,
            source=corpus.get(chunk.file_path or ""),
            corpus=corpus,
            max_tokens=max_tokens,
        )
        if ctx.is_empty:
            chunk.stitched_context = None
            continue
        chunk.stitched_context = ctx.text
        stitched_count += 1
        total_tokens += ctx.token_est

    return stitched_count, total_tokens
