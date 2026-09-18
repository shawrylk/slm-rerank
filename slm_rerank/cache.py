"""Persistent hash-based SQLite caching for lfm-rerank with model isolation."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class RerankCache:
    """Persistent SQLite cache keyed by sha256(model_id : prompt_version : normalized_query : chunk_hash : chunk_byte_range : query_intent : prior_version : length_exponent)."""

    def __init__(self, db_path: Optional[Path] = None, model_id: str = "lfm"):
        self.model_id = model_id
        if db_path is None:
            cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "lfm-rerank"
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = cache_dir / "cache.db"
        else:
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rerank_cache (
                    cache_key TEXT PRIMARY KEY,
                    model_id TEXT,
                    prompt_version TEXT,
                    normalized_query TEXT NOT NULL,
                    chunk_hash TEXT NOT NULL,
                    chunk_byte_range TEXT,
                    query_intent TEXT,
                    prior_version TEXT,
                    length_exponent REAL,
                    score REAL NOT NULL,
                    logprob_yes REAL,
                    logprob_no REAL,
                    created_at REAL NOT NULL
                )
                """
            )
            # Safe migration for older tables missing v0.3.2 columns
            for col, col_type in [
                ("model_id", "TEXT"),
                ("prompt_version", "TEXT"),
                ("chunk_byte_range", "TEXT"),
                ("query_intent", "TEXT"),
                ("prior_version", "TEXT"),
                ("length_exponent", "REAL"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE rerank_cache ADD COLUMN {col} {col_type}")
                except sqlite3.OperationalError:
                    pass

            conn.execute("CREATE INDEX IF NOT EXISTS idx_query ON rerank_cache (normalized_query)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_model ON rerank_cache (model_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunk_hash ON rerank_cache (chunk_hash)")
            conn.commit()

    @staticmethod
    def compute_cache_key(
        query: str,
        chunk_content: Optional[str] = None,
        model_id: str = "lfm",
        prompt_version: str = "v1",
        chunk_hash: Optional[str] = None,
        chunk_byte_range: Optional[str] = None,
        query_intent: Optional[str] = None,
        prior_version: str = "v0.3.2",
        length_exponent: Optional[float] = None,
    ) -> Tuple[str, str, str]:
        """Compute robust cache key incorporating model_id, prompt_version, query_text, chunk_hash,

        chunk_byte_range, query_intent, prior_version, and length_exponent.
        Returns (cache_key, normalized_query, chunk_hash).
        """
        normalized_query = " ".join(query.strip().lower().split())
        c_hash = chunk_hash or (hashlib.sha256(chunk_content.encode("utf-8")).hexdigest() if chunk_content is not None else "")
        byte_range_str = str(chunk_byte_range) if chunk_byte_range is not None else ""
        intent_str = str(query_intent) if query_intent is not None else ""
        p_version = str(prompt_version) if prompt_version is not None else "v1"
        pr_version = str(prior_version) if prior_version is not None else "v0.3.2"
        len_exp_str = f"{length_exponent:.4f}" if length_exponent is not None else ""

        key_raw = f"{model_id}:{p_version}:{normalized_query}:{c_hash}:{byte_range_str}:{intent_str}:{pr_version}:{len_exp_str}"
        cache_key = hashlib.sha256(key_raw.encode("utf-8")).hexdigest()
        return cache_key, normalized_query, c_hash

    def get(
        self,
        query: str,
        chunk_content: Optional[str] = None,
        model_id: Optional[str] = None,
        prompt_version: str = "v1",
        chunk_hash: Optional[str] = None,
        chunk_byte_range: Optional[str] = None,
        query_intent: Optional[str] = None,
        prior_version: str = "v0.3.2",
        length_exponent: Optional[float] = None,
    ) -> Optional[Dict[str, float]]:
        """Retrieve cached score for a single chunk."""
        mid = model_id or self.model_id
        cache_key, norm_q, c_hash = self.compute_cache_key(
            query=query,
            chunk_content=chunk_content,
            model_id=mid,
            prompt_version=prompt_version,
            chunk_hash=chunk_hash,
            chunk_byte_range=chunk_byte_range,
            query_intent=query_intent,
            prior_version=prior_version,
            length_exponent=length_exponent,
        )
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT score, logprob_yes, logprob_no FROM rerank_cache WHERE cache_key = ?",
                (cache_key,),
            )
            row = cursor.fetchone()
            if row:
                return {
                    "score": float(row["score"]),
                    "logprob_yes": float(row["logprob_yes"]) if row["logprob_yes"] is not None else None,
                    "logprob_no": float(row["logprob_no"]) if row["logprob_no"] is not None else None,
                }
            # Fallback for legacy cached entries matching (normalized_query, chunk_hash) with matching or NULL model_id
            cursor = conn.execute(
                "SELECT score, logprob_yes, logprob_no FROM rerank_cache WHERE normalized_query = ? AND chunk_hash = ? AND (model_id = ? OR model_id IS NULL) LIMIT 1",
                (norm_q, c_hash, mid),
            )
            row = cursor.fetchone()
            if row:
                return {
                    "score": float(row["score"]),
                    "logprob_yes": float(row["logprob_yes"]) if row["logprob_yes"] is not None else None,
                    "logprob_no": float(row["logprob_no"]) if row["logprob_no"] is not None else None,
                }
        return None

    def get_batch(
        self,
        query: str,
        chunk_contents: Optional[List[str]] = None,
        model_id: Optional[str] = None,
        prompt_version: str = "v1",
        query_intent: Optional[str] = None,
        prior_version: str = "v0.3.2",
        length_exponent: Optional[float] = None,
        chunks: Optional[List[Any]] = None,
    ) -> Dict[int, Dict[str, float]]:
        """Batch retrieve cached scores. Returns map of chunk_index -> score_dict."""
        candidate_items = chunks if chunks is not None else (chunk_contents or [])
        if not candidate_items:
            return {}

        mid = model_id or self.model_id
        results: Dict[int, Dict[str, float]] = {}
        keys_to_indices: Dict[str, int] = {}
        missing_by_hash: Dict[str, int] = {}

        for idx, item in enumerate(candidate_items):
            if hasattr(item, "content"):
                c_content = item.content
                c_hash = item.content_hash or hashlib.sha256(c_content.encode("utf-8")).hexdigest()
                b_range = f"{item.start_byte}-{item.end_byte}" if getattr(item, "start_byte", None) is not None else None
            else:
                c_content = str(item)
                c_hash = hashlib.sha256(c_content.encode("utf-8")).hexdigest()
                b_range = None

            cache_key, norm_q, _ = self.compute_cache_key(
                query=query,
                chunk_content=c_content,
                model_id=mid,
                prompt_version=prompt_version,
                chunk_hash=c_hash,
                chunk_byte_range=b_range,
                query_intent=query_intent,
                prior_version=prior_version,
                length_exponent=length_exponent,
            )
            keys_to_indices[cache_key] = idx
            missing_by_hash[c_hash] = idx

        all_keys = list(keys_to_indices.keys())
        normalized_query = " ".join(query.strip().lower().split())

        # 1. Primary lookup by exact robust 8-tuple cache_key
        with self._get_connection() as conn:
            for i in range(0, len(all_keys), 500):
                batch_keys = all_keys[i : i + 500]
                placeholders = ",".join("?" for _ in batch_keys)
                cursor = conn.execute(
                    f"SELECT cache_key, score, logprob_yes, logprob_no FROM rerank_cache WHERE cache_key IN ({placeholders})",
                    batch_keys,
                )
                for row in cursor.fetchall():
                    c_key = row["cache_key"]
                    idx = keys_to_indices[c_key]
                    results[idx] = {
                        "score": float(row["score"]),
                        "logprob_yes": float(row["logprob_yes"]) if row["logprob_yes"] is not None else None,
                        "logprob_no": float(row["logprob_no"]) if row["logprob_no"] is not None else None,
                    }
                    # Remove from missing
                    if idx in candidate_items:
                        pass

            # 2. Legacy fallback lookup by (normalized_query, chunk_hash) for missing indices
            unresolved_indices = [idx for idx in range(len(candidate_items)) if idx not in results]
            if unresolved_indices:
                unresolved_hashes = []
                for idx in unresolved_indices:
                    item = candidate_items[idx]
                    h = item.content_hash if hasattr(item, "content_hash") and item.content_hash else hashlib.sha256((item.content if hasattr(item, "content") else str(item)).encode("utf-8")).hexdigest()
                    unresolved_hashes.append((h, idx))

                for i in range(0, len(unresolved_hashes), 500):
                    batch_pairs = unresolved_hashes[i : i + 500]
                    batch_h_list = [p[0] for p in batch_pairs]
                    h_to_idx = {p[0]: p[1] for p in batch_pairs}
                    placeholders = ",".join("?" for _ in batch_h_list)
                    cursor = conn.execute(
                        f"SELECT chunk_hash, score, logprob_yes, logprob_no FROM rerank_cache WHERE normalized_query = ? AND (model_id = ? OR model_id IS NULL) AND chunk_hash IN ({placeholders})",
                        [normalized_query, mid] + batch_h_list,
                    )
                    for row in cursor.fetchall():
                        ch = row["chunk_hash"]
                        if ch in h_to_idx:
                            t_idx = h_to_idx[ch]
                            results[t_idx] = {
                                "score": float(row["score"]),
                                "logprob_yes": float(row["logprob_yes"]) if row["logprob_yes"] is not None else None,
                                "logprob_no": float(row["logprob_no"]) if row["logprob_no"] is not None else None,
                            }

        return results

    def put_batch(
        self,
        query: str,
        entries: List[Tuple[Any, float, Optional[float], Optional[float]]],
        model_id: Optional[str] = None,
        prompt_version: str = "v1",
        query_intent: Optional[str] = None,
        prior_version: str = "v0.3.2",
        length_exponent: Optional[float] = None,
    ) -> None:
        """Batch write entries isolated by robust 8-tuple cache key."""
        if not entries:
            return

        mid = model_id or self.model_id
        now = time.time()
        rows = []
        for item, score, lp_yes, lp_no in entries:
            if hasattr(item, "content"):
                c_content = item.content
                c_hash = item.content_hash or hashlib.sha256(c_content.encode("utf-8")).hexdigest()
                b_range = f"{item.start_byte}-{item.end_byte}" if getattr(item, "start_byte", None) is not None else None
            else:
                c_content = str(item)
                c_hash = hashlib.sha256(c_content.encode("utf-8")).hexdigest()
                b_range = None

            cache_key, norm_q, _ = self.compute_cache_key(
                query=query,
                chunk_content=c_content,
                model_id=mid,
                prompt_version=prompt_version,
                chunk_hash=c_hash,
                chunk_byte_range=b_range,
                query_intent=query_intent,
                prior_version=prior_version,
                length_exponent=length_exponent,
            )
            rows.append((
                cache_key,
                mid,
                prompt_version,
                norm_q,
                c_hash,
                b_range,
                str(query_intent or ""),
                prior_version,
                length_exponent,
                score,
                lp_yes,
                lp_no,
                now,
            ))

        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO rerank_cache 
                (cache_key, model_id, prompt_version, normalized_query, chunk_hash, chunk_byte_range, query_intent, prior_version, length_exponent, score, logprob_yes, logprob_no, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

    def clear(self, model_id: Optional[str] = None) -> None:
        """Clear cached entries, optionally filtered by model_id."""
        with self._get_connection() as conn:
            if model_id is not None:
                conn.execute("DELETE FROM rerank_cache WHERE model_id = ?", (model_id,))
            else:
                conn.execute("DELETE FROM rerank_cache")
            conn.commit()
