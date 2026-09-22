"""v0.6.6: candidate recall is a ranked union of path and body matches.

The reranker can only score what recall hands it. Gating filename matching behind
"content search found nothing" made a file named after the thing you asked for
unreachable the moment any other file mentioned the words -- which is exactly what
test files do. These tests mirror src/index.test.mjs so both implementations stay
in step.
"""
import subprocess

import pytest

from slm_rerank.discovery import discover_candidate_files, extract_query_terms, is_test_path


def test_query_terms_keep_stems_beside_their_roots():
    terms = extract_query_terms("interop harness migration fixture")
    assert terms == ["interop", "harness", "migration", "migrat", "migrate", "fixture"]

    # stems must follow their own root, or a term cap severs them
    assert terms.index("migrat") > terms.index("migration")
    assert terms.index("fixture") > terms.index("migrate")

    # -ss is not a plural, -es is
    assert extract_query_terms("harness") == ["harness"]
    assert extract_query_terms("classes") == ["classes", "class"]
    assert extract_query_terms("exports") == ["exports", "export"]
    assert extract_query_terms("chunking") == ["chunking", "chunk"]

    # stop words only -> falls back to the first raw word
    assert extract_query_terms("the and of") == ["the"]
    assert extract_query_terms("") == []


@pytest.mark.parametrize(
    "path,expected",
    [
        ("tests/unit-1.test.ts", True),
        ("src/__tests__/thing.ts", True),
        ("tests/test_client.py", True),
        ("pkg/client_test.go", True),
        ("src/interop/migrate-fixture.ts", False),
        ("src/fixtures/rows.ts", False),
        ("src/latest.ts", False),
    ],
)
def test_is_test_path(path, expected):
    assert is_test_path(path) is expected


@pytest.fixture()
def recall_repo(tmp_path):
    """A file whose *name* carries the query vocabulary, drowned in tests that say it too."""
    (tmp_path / "src" / "interop").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "interop" / "migrate-fixture.ts").write_text(
        "export function buildLegacyBridge(rows) { return rows.map(normalizeRow); }\n"
    )
    for i in range(1, 6):
        (tmp_path / "tests" / f"unit-{i}.test.ts").write_text(
            'describe("interop harness", () => { it("runs the interop harness migration", () => {}); });\n'
        )

    # discovery shells out to rg or git; make the tree visible to either
    subprocess.run(["git", "init", "-q", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=test", "commit", "-qm", "fixture"],
        cwd=tmp_path, check=True,
    )
    return tmp_path


@pytest.mark.parametrize(
    "query", ["fixture", "interop fixture", "interop harness migration fixture"]
)
def test_filename_match_outranks_tests_mentioning_the_words(recall_repo, query):
    files = discover_candidate_files(query, cwd=str(recall_repo))
    assert any("migrate-fixture" in f for f in files), f"{query!r} lost the fixture file: {files}"
    assert "migrate-fixture" in files[0], f"{query!r} ranked {files[0]} above the fixture file"


def test_unmatched_query_still_returns_a_sample(recall_repo):
    """Nothing matching is not the same as nothing to look at."""
    files = discover_candidate_files("zzzznotpresent", cwd=str(recall_repo))
    assert files
    assert all(f.endswith(".ts") for f in files)
