"""Unit tests for deterministic physical ground-truth verifier."""

import tempfile
from pathlib import Path
import pytest

from slm_rerank.models import CandidateChunk, GroundTruthStatus
from slm_rerank.verifier import GroundTruthVerifier


@pytest.fixture
def sample_code_file():
    content = (
        "import os\n"
        "\n"
        "def authenticate_request(token: str) -> bool:\n"
        "    if not token:\n"
        "        return False\n"
        "    return token == 'secret'\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(content)
        f_path = f.name
    yield Path(f_path)
    Path(f_path).unlink(missing_ok=True)


def test_verifier_exact_match(sample_code_file):
    verifier = GroundTruthVerifier()
    chunk = CandidateChunk(
        id="c1",
        file_path=str(sample_code_file),
        start_line=3,
        end_line=6,
        symbol="authenticate_request",
        content="3: def authenticate_request...",
    )

    result = verifier.verify_chunk(
        chunk=chunk,
        score=0.92,
        logprob_yes=-0.15,
        logprob_no=-2.10,
    )

    assert result.ground_truth_status == GroundTruthStatus.VERIFIED
    assert result.score == 0.92
    assert result.citation.file == str(sample_code_file)
    assert result.citation.start_line == 3
    assert result.citation.end_line == 6
    assert result.citation.symbol == "authenticate_request"
    assert "authenticate_request" in result.snippet


def test_verifier_file_not_found():
    verifier = GroundTruthVerifier()
    chunk = CandidateChunk(
        id="c2",
        file_path="/non/existent/fake_auth.py",
        start_line=1,
        end_line=5,
        symbol="auth",
        content="def auth(): pass",
    )

    result = verifier.verify_chunk(chunk=chunk, score=0.85)
    assert result.ground_truth_status == GroundTruthStatus.FILE_NOT_FOUND
    assert result.score == 0.0


def test_verifier_invalid_line_range(sample_code_file):
    verifier = GroundTruthVerifier()
    # File only has 6 lines, chunk claims lines 100-150
    chunk = CandidateChunk(
        id="c3",
        file_path=str(sample_code_file),
        start_line=100,
        end_line=150,
        symbol="func",
        content="...",
    )

    result = verifier.verify_chunk(chunk=chunk, score=0.80)
    assert result.ground_truth_status == GroundTruthStatus.INVALID_LINE_RANGE
    assert result.score == 0.0
