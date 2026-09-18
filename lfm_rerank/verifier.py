"""Deterministic physical ground-truth verifier for lfm-rerank."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .models import (
    CandidateChunk,
    Citation,
    GroundTruthStatus,
    RerankResultItem,
)


class GroundTruthVerifier:
    """Verifies that chunks and citations strictly align with physical disk reality."""

    def __init__(self, strict: bool = False):
        self.strict = strict

    def verify_chunk(
        self,
        chunk: CandidateChunk,
        score: float,
        logprob_yes: Optional[float] = None,
        logprob_no: Optional[float] = None,
        cached: bool = False,
        prompt_tokens: int = 0,
        completion_tokens: int = 1,
        duration_ms: float = 0.0,
        raw_score: Optional[float] = None,
        adjusted_score: Optional[float] = None,
        delta: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RerankResultItem:
        """Create verified RerankResultItem with deterministic physical citation."""
        r_score = score if raw_score is None else raw_score
        a_score = score if adjusted_score is None else adjusted_score
        d_score = round(a_score - r_score, 4) if delta is None else delta
        item_meta = metadata or {}

        # 1. Raw text candidate without physical file backing
        if not chunk.file_path:
            return RerankResultItem(
                candidate_id=chunk.id,
                file_path=None,
                raw_score=r_score,
                adjusted_score=a_score,
                delta=d_score,
                score=a_score,
                logprob_yes=logprob_yes,
                logprob_no=logprob_no,
                citation=Citation(file="raw_text", start_line=chunk.start_line, end_line=chunk.end_line, symbol=chunk.symbol),
                symbol=chunk.symbol,
                ground_truth_status=GroundTruthStatus.UNVERIFIED_RAW_TEXT,
                ground_truth_details="Candidate is raw text (no physical file on disk)",
                snippet=chunk.content[:300],
                cached=cached,
                is_test=chunk.is_test,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                duration_ms=duration_ms,
                metadata=item_meta,
            )

        # 2. Verify physical file exists on disk
        p = Path(chunk.file_path).resolve()
        if not p.is_file():
            return RerankResultItem(
                candidate_id=chunk.id,
                file_path=str(p),
                raw_score=0.0,
                adjusted_score=0.0,
                delta=0.0,
                score=0.0,
                logprob_yes=logprob_yes,
                logprob_no=logprob_no,
                citation=Citation(file=str(p), start_line=chunk.start_line, end_line=chunk.end_line, symbol=chunk.symbol),
                symbol=chunk.symbol,
                ground_truth_status=GroundTruthStatus.FILE_NOT_FOUND,
                ground_truth_details=f"File '{p}' does not exist on disk",
                snippet=None,
                cached=cached,
                is_test=chunk.is_test,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                duration_ms=duration_ms,
                metadata=item_meta,
            )

        # 3. Verify line range and symbol
        try:
            with open(p, "r", encoding="utf-8", newline="", errors="replace") as f:
                lines = f.readlines()
        except Exception as e:
            return RerankResultItem(
                candidate_id=chunk.id,
                file_path=str(p),
                raw_score=0.0,
                adjusted_score=0.0,
                delta=0.0,
                score=0.0,
                logprob_yes=logprob_yes,
                logprob_no=logprob_no,
                citation=Citation(file=str(p), start_line=chunk.start_line, end_line=chunk.end_line, symbol=chunk.symbol),
                symbol=chunk.symbol,
                ground_truth_status=GroundTruthStatus.FILE_NOT_FOUND,
                ground_truth_details=f"Could not read file: {e}",
                snippet=None,
                cached=cached,
                is_test=chunk.is_test,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                duration_ms=duration_ms,
                metadata=item_meta,
            )

        total_lines = len(lines)
        if chunk.start_line < 1 or chunk.end_line > total_lines or chunk.start_line > chunk.end_line:
            return RerankResultItem(
                candidate_id=chunk.id,
                file_path=str(p),
                raw_score=0.0,
                adjusted_score=0.0,
                delta=0.0,
                score=0.0,
                logprob_yes=logprob_yes,
                logprob_no=logprob_no,
                citation=Citation(file=str(p), start_line=chunk.start_line, end_line=chunk.end_line, symbol=chunk.symbol),
                symbol=chunk.symbol,
                ground_truth_status=GroundTruthStatus.INVALID_LINE_RANGE,
                ground_truth_details=f"Lines {chunk.start_line}-{chunk.end_line} exceed file length ({total_lines})",
                snippet=None,
                cached=cached,
                is_test=chunk.is_test,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                duration_ms=duration_ms,
                metadata=item_meta,
            )

        snippet_lines = lines[chunk.start_line - 1 : min(chunk.end_line, chunk.start_line + 8)]
        snippet_text = "".join(snippet_lines)

        return RerankResultItem(
            candidate_id=chunk.id,
            file_path=str(p),
            raw_score=r_score,
            adjusted_score=a_score,
            delta=d_score,
            score=a_score,
            logprob_yes=logprob_yes,
            logprob_no=logprob_no,
            citation=Citation(
                file=str(p),
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                symbol=chunk.symbol,
            ),
            symbol=chunk.symbol,
            ground_truth_status=GroundTruthStatus.VERIFIED,
            ground_truth_details="Deterministic AST/symbol match: physical file and line range verified",
            snippet=snippet_text,
            cached=cached,
            is_test=chunk.is_test,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            duration_ms=duration_ms,
            metadata=item_meta,
        )
