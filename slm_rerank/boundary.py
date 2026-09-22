"""
Architecture Boundary Slicing for Codebase Reranking.
Detects vertical slices (features/<slice>, modules/<slice>, packages/<slice>)
and clusters reranker results by architectural domain boundary.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List


def detect_slice(file_path: str) -> str:
    """
    Detect the architectural vertical slice or module for a file path.
    Conforms to standard bounded-context and vertical-slice architecture patterns.
    """
    if not file_path:
        return "root"

    normalized = file_path.replace("\\", "/")
    parts = [p for p in normalized.split("/") if p]

    boundary_keywords = {"features", "modules", "packages", "apps", "workers", "services", "domains"}

    for i in range(len(parts) - 1):
        if parts[i].lower() in boundary_keywords:
            return parts[i + 1]

    # Fallback to src/<slice>/
    if "src" in parts:
        src_idx = parts.index("src")
        if src_idx < len(parts) - 2:
            return parts[src_idx + 1]

    if len(parts) > 1:
        return parts[0]
    return "root"


def group_by_slice(results: List[Any]) -> Dict[str, Dict[str, Any]]:
    """
    Group reranking results by architectural vertical slice.
    """
    groups: Dict[str, Dict[str, Any]] = {}

    for item in results:
        file_path = getattr(item, "file_path", None)
        if not file_path and hasattr(item, "chunk"):
            file_path = getattr(item.chunk, "file_path", None)

        slice_name = getattr(item, "slice", None) or detect_slice(file_path)

        score = getattr(item, "score", 0.0)

        if slice_name not in groups:
            groups[slice_name] = {
                "slice": slice_name,
                "max_score": score,
                "items": [],
            }

        if score > groups[slice_name]["max_score"]:
            groups[slice_name]["max_score"] = score
        groups[slice_name]["items"].append(item)

    # Sort slices descending by max_score
    sorted_keys = sorted(groups.keys(), key=lambda k: groups[k]["max_score"], reverse=True)
    return {k: groups[k] for k in sorted_keys}
