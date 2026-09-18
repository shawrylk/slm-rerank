"""Unit tests for persistent SQLite caching."""

import tempfile
from pathlib import Path
from lfm_rerank.cache import RerankCache


def test_cache_put_and_get():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "test_cache.db"
        cache = RerankCache(db_path=db_file)

        query = "auth middleware handler"
        chunk_content = "1: def auth(): pass"

        # Initially miss
        assert cache.get(query, chunk_content) is None

        # Write entry
        cache.put_batch(query, [(chunk_content, 0.88, -0.15, -2.10)])

        # Read back
        cached = cache.get(query, chunk_content)
        assert cached is not None
        assert abs(cached["score"] - 0.88) < 1e-4
        assert abs(cached["logprob_yes"] - (-0.15)) < 1e-4

        # Query with different casing and whitespace hits the same cache key
        cached_variant = cache.get("   AUTH   middleware   handler  ", chunk_content)
        assert cached_variant is not None
        assert abs(cached_variant["score"] - 0.88) < 1e-4


def test_cache_batch_get():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "test_cache.db"
        cache = RerankCache(db_path=db_file)

        query = "search query"
        chunks = ["chunk_a", "chunk_b", "chunk_c"]

        cache.put_batch(query, [
            ("chunk_a", 0.90, -0.1, -2.5),
            ("chunk_c", 0.30, -2.5, -0.2),
        ])

        batch_res = cache.get_batch(query, chunks)
        assert len(batch_res) == 2
        assert 0 in batch_res
        assert batch_res[2]["score"] == 0.30


def test_robust_cache_key_invalidation():
    """Verify that cache key is invalidated by changes in any of the 8 parameters:

    model_id, prompt_version, query_text, chunk_hash, chunk_byte_range,
    query_intent, prior_version, length_exponent.
    """
    base_args = {
        "query": "find trigger handler",
        "chunk_content": "1: def trigger(): pass",
        "model_id": "lfm",
        "prompt_version": "v1",
        "chunk_byte_range": "0-100",
        "query_intent": "IMPLEMENTATION",
        "prior_version": "v0.3.2",
        "length_exponent": 0.15,
    }
    base_key, _, _ = RerankCache.compute_cache_key(**base_args)

    # 1. model_id
    k_model, _, _ = RerankCache.compute_cache_key(**{**base_args, "model_id": "qwen"})
    assert k_model != base_key

    # 2. prompt_version
    k_prompt, _, _ = RerankCache.compute_cache_key(**{**base_args, "prompt_version": "v2"})
    assert k_prompt != base_key

    # 3. query_text
    k_query, _, _ = RerankCache.compute_cache_key(**{**base_args, "query": "find different trigger"})
    assert k_query != base_key

    # 4. chunk_hash / content
    k_chunk, _, _ = RerankCache.compute_cache_key(**{**base_args, "chunk_content": "1: def other(): pass"})
    assert k_chunk != base_key

    # 5. chunk_byte_range
    k_range, _, _ = RerankCache.compute_cache_key(**{**base_args, "chunk_byte_range": "100-250"})
    assert k_range != base_key

    # 6. query_intent
    k_intent, _, _ = RerankCache.compute_cache_key(**{**base_args, "query_intent": "SPECIFICATION"})
    assert k_intent != base_key

    # 7. prior_version
    k_prior, _, _ = RerankCache.compute_cache_key(**{**base_args, "prior_version": "v0.4.0"})
    assert k_prior != base_key

    # 8. length_exponent
    k_len, _, _ = RerankCache.compute_cache_key(**{**base_args, "length_exponent": 0.12})
    assert k_len != base_key

