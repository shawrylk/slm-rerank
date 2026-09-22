"""Data models for LFM Semantic Co-Processor Wide Reranker."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QueryIntent(str, Enum):
    IMPLEMENTATION = "IMPLEMENTATION"
    SPECIFICATION = "SPECIFICATION"
    BUG_DIAGNOSIS = "BUG_DIAGNOSIS"
    REFACTOR = "REFACTOR"


class GroundTruthStatus(str, Enum):
    VERIFIED = "VERIFIED"
    SYMBOL_NOT_FOUND = "SYMBOL_NOT_FOUND"
    INVALID_LINE_RANGE = "INVALID_LINE_RANGE"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    PARSE_ERROR = "PARSE_ERROR"
    AMBIGUOUS_COMPLETION = "AMBIGUOUS_COMPLETION"
    UNVERIFIED_RAW_TEXT = "UNVERIFIED_RAW_TEXT"


class Citation(BaseModel):
    file: str = Field(description="Relative or absolute path of the cited file")
    start_line: int = Field(default=1, description="Start line number (1-indexed)")
    end_line: int = Field(default=1, description="End line number (1-indexed)")
    symbol: Optional[str] = Field(default=None, description="Exact function, class, or symbol name")


class CandidateChunk(BaseModel):
    id: str = Field(description="Unique chunk ID (file:start-end:symbol)")
    file_path: Optional[str] = Field(default=None, description="Source file path on disk")
    start_line: int = Field(default=1, description="Starting line in source file")
    end_line: int = Field(default=1, description="Ending line in source file")
    start_byte: Optional[int] = Field(default=None, description="Starting byte offset in physical source file")
    end_byte: Optional[int] = Field(default=None, description="Ending byte offset in physical source file")
    symbol: Optional[str] = Field(default=None, description="Extracted AST/symbol definition name")
    content: str = Field(description="Code snippet formatted with physical line numbers")
    token_est: int = Field(default=0, description="Estimated token count")
    content_hash: str = Field(default="", description="SHA-256 hash of content")
    is_test: bool = Field(default=False, description="Whether this chunk originates from a test file")
    stitched_context: Optional[str] = Field(
        default=None,
        description="1-hop call-graph/type context prepended to the scoring prompt only (<= 150 tokens)",
    )


class CandidateManifestEntry(BaseModel):
    file_path: str = Field(description="Physical file path evaluated")
    chunks_count: int = Field(default=0, description="Number of semantic chunks evaluated")
    max_score: float = Field(default=0.0, description="Highest relevance probability for this file")
    status: str = Field(default="omitted", description="'included' if passed threshold, else 'omitted'")
    symbols: List[str] = Field(default_factory=list, description="Top-level symbols found in file")
    is_test: bool = Field(default=False, description="Whether the file is a test specification")


class RerankResultItem(BaseModel):
    candidate_id: str
    file_path: Optional[str] = None
    raw_score: float = 0.0  # Pure model probability before priors (model diagnostics)
    decision_score: float = 0.0  # Log-odds intent-adjusted score (ranking & thresholding)
    calibrated_score: float = 0.0  # Post-hoc calibrated probability via Platt/Temperature scaling
    adjusted_score: float = 0.0  # Alias for decision_score (backward compatibility)
    delta: float = 0.0  # decision_score - raw_score
    score: float = 0.0  # Normalized probability (alias for decision_score)
    logprob_yes: Optional[float] = None
    logprob_no: Optional[float] = None
    citation: Citation  # Deterministic physical citation
    symbol: Optional[str] = None
    ground_truth_status: GroundTruthStatus = GroundTruthStatus.VERIFIED
    ground_truth_details: str = ""
    snippet: Optional[str] = None
    cached: bool = False
    below_threshold: bool = False  # True if retained via fallback floor
    ambiguous: bool = False  # True if top-1 token was neither yes nor no
    is_test: bool = False
    redundancy_penalized: bool = False  # True if MMR redundancy penalty applied
    symbol_boosted: bool = False  # True if symbol match prior applied
    matched_symbols: List[str] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 1
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Utility Trap Resolution & Context Stitching (v0.5.0)
    domain_hits: int = 0  # Distinct core domain entities matched anywhere in the chunk
    structural_domain_hits: int = 0  # Domain entities matched in the symbol or file path
    symptom_hits: int = 0  # Distinct symptom modifiers (crash, exception, ...) matched
    matched_domain_entities: List[str] = Field(default_factory=list)
    symbol_role: str = "UNKNOWN"  # PUBLIC_API | INTERNAL_HELPER | UNKNOWN
    centrality_boosted: bool = False  # Architectural centrality boost applied
    utility_penalized: bool = False  # Small internal helper penalty applied
    utility_trap_demoted: bool = False  # Clamped below the best domain-aligned chunk
    context_stitched: bool = False  # Scored with 1-hop stitched context
    tier1_lexical_score: Optional[float] = None  # Tier-1 hybrid pre-filter score, if it ran


class AmbiguityEvent(BaseModel):
    token_text: str = ""
    token_id: Optional[int] = None
    logprob: Optional[float] = None
    prompt_suffix: str = ""
    prompt_state: str = ""
    chunk_id: Optional[str] = None
    reason: str = ""
    top_candidates: List[Dict[str, Any]] = Field(default_factory=list)


class Telemetry(BaseModel):
    candidate_files_count: int = 0
    chunks_evaluated: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    ambiguous_completions: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    prefill_tokens_per_sec: float = 0.0
    decode_tokens_per_sec: float = 0.0
    total_wall_time_s: float = 0.0
    total_input_bytes: int = 0
    top_k_input_tokens: int = 0
    reduction_ratio: float = 0.0
    reduction_percentage: float = 0.0
    concurrency: int = 1
    threshold: float = 0.65
    fallback_floor_triggered: bool = False
    top_ceiling: Optional[int] = None
    query_intent: Optional[QueryIntent] = None

    # Operational Telemetry Tracking (v0.3.1)
    ambiguous_completion_rate: float = 0.0
    yes_variant_rate: float = 0.0
    no_variant_rate: float = 0.0
    cache_hit_rate: float = 0.0
    score_latency_p95: float = 0.0
    model_id: Optional[str] = None

    # Abstention & Prior Telemetry (v0.3.2)
    abstained: bool = False
    abstain_threshold: float = 0.20
    intent_confidence: float = 1.0

    # Calibration & Failure-Mode Telemetry (v0.3.3)
    pre_calibration_brier: Optional[float] = None
    post_calibration_brier: Optional[float] = None
    pre_calibration_ece: Optional[float] = None
    post_calibration_ece: Optional[float] = None
    calibrator_type: Optional[str] = None
    redundancy_penalties_applied: int = 0
    symbol_boosts_applied: int = 0
    margin_threshold_used: float = 0.15
    ambiguity_events: List[Dict[str, Any]] = Field(default_factory=list)

    # Two-Tier Hybrid Search (v0.5.0)
    tier1_applied: bool = False
    tier1_candidates_in: int = 0
    tier1_candidates_out: int = 0
    tier1_reason: str = ""
    full_evaluation: bool = True

    # Call-Graph & Type Context Stitching (v0.5.0)
    context_stitching_enabled: bool = False
    chunks_context_stitched: int = 0
    stitched_context_tokens: int = 0
    max_stitched_context_tokens: int = 150

    # Utility Trap Resolution (v0.5.0)
    domain_entities: List[str] = Field(default_factory=list)
    symptom_modifiers: List[str] = Field(default_factory=list)
    centrality_boosts_applied: int = 0
    utility_penalties_applied: int = 0
    utility_trap_demotions: int = 0


class RerankResponse(BaseModel):
    query: str
    threshold: float
    top_k: Optional[int] = None
    intent: QueryIntent = QueryIntent.IMPLEMENTATION
    results: List[RerankResultItem]
    candidate_manifest: List[CandidateManifestEntry]
    telemetry: Telemetry
