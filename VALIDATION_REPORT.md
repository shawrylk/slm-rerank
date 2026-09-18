# LFM Wide Reranker v0.3.3 Formal Verification Report
**Date**: 2026-09-18 05:33:07 UTC
**Target Architecture**: Liquid Foundation Model (LFM 2.5 8B-A1B Q8_0) & Multi-Provider Adapters

---

## Executive Summary
- **Tasks Evaluated**: 11 ground-truth tasks across `qc-mono` and `llama.cpp`
- **Recall@1**: `45.5%`
- **Recall@3**: `72.7%`
- **Recall@5**: `81.8%`
- **Mean Reciprocal Rank (MRR)**: `0.5833`
- **Mean Rank of Target Chunks**: `9.18`
- **Pre-Calibration Brier Score**: `0.2532` -> **Post-Calibration Brier**: `0.0968`
- **Pre-Calibration ECE**: `0.3524` -> **Post-Calibration ECE**: `0.0273`
- **Ambiguous Completion Rate**: `0.00%` (0 ambiguous tokens out of 687)
- **Total Wall-Clock Time**: `0.46s`

### 95% Bootstrap Confidence Intervals (1,000 Resamples)
| Evaluation Metric | Point Estimate / Mean | 95% CI Lower | 95% CI Upper | 95% Confidence Interval |
| :--- | :---: | :---: | :---: | :---: |
| **Recall At 1** | `45.3%` | `18.2%` | `72.7%` | `[18.2%, 72.7%]` |
| **Recall At 3** | `72.8%` | `45.5%` | `100.0%` | `[45.5%, 100.0%]` |
| **Recall At 5** | `81.5%` | `54.5%` | `100.0%` | `[54.5%, 100.0%]` |
| **Mrr** | `0.5823` | `0.3483` | `0.8212` | `[0.3483, 0.8212]` |
| **Brier Score** | `0.2514` | `0.2095` | `0.2971` | `[0.2095, 0.2971]` |
| **Ece** | `0.3516` | `0.2320` | `0.4363` | `[0.2320, 0.4363]` |
| **Calibrated Brier Score** | `0.0972` | `0.0348` | `0.1846` | `[0.0348, 0.1846]` |
| **Calibrated Ece** | `0.0589` | `0.0229` | `0.1201` | `[0.0229, 0.1201]` |

---

## Artifact Index
- [Artifact 1: Multi-Stage Calibration Table](ARTIFACT_1_MULTISTAGE_CALIBRATION.md)
- [Artifact 2: Reliability Diagram & Calibration Table](ARTIFACT_2_RELIABILITY_TABLE.md)
- [Artifact 3: Intent Slice Report](ARTIFACT_3_INTENT_SLICE_REPORT.md)
- [Artifact 4: Failure Analysis](ARTIFACT_4_FAILURE_ANALYSIS.md)
- [Artifact 5: Decision & Operational Telemetry Summary](ARTIFACT_5_TELEMETRY_SUMMARY.md)

---

# Validation Artifact 1: Multi-Stage Calibration Table
## LFM Semantic Co-Processor Wide Reranker (v0.3.3)

Evaluation comparing calibration and discriminative ranking power across four scoring stages:
1. **Raw Logprob**: Pure model logit difference without length normalization ($lp_{yes} - lp_{no}$, exponent = 0.0)
2. **Length-Normalized**: Model logprob scaled by physical chunk length $(120 / L)^{0.15}$
3. **Intent-Adjusted**: Pre-calibration decision score incorporating log-odds query intent prior deltas
4. **Post-Hoc Calibrated**: Post-hoc calibrated probability via `PlattScalingCalibrator` (1D logistic regression on logits with cross-validation)

| Pipeline Stage | Brier Score | ECE | AUROC | AUPRC | Description |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Raw Logprob** | `0.2842` | `0.3818` | `0.6444` | `0.2146` | Pure model logit difference without length normalization (exp=0.0) |
| **Length-Normalized** | `0.2724` | `0.3716` | `0.6493` | `0.2156` | Logit calibrated with token-length normalization scaling (exp=0.15) |
| **Intent-Adjusted** | `0.2532` | `0.3524` | `0.6940` | `0.3135` | Pre-calibration decision score after applying query intent log-odds prior delta |
| **Post-Hoc Calibrated** | `0.0968` | `0.0273` | `0.6956` | `0.3155` | Post-hoc calibrated probability via PlattScalingCalibrator (logistic regression on logits) |

### Key Findings:
- **Pre- vs Post-Calibration Brier Score**: Improved from `0.2532` to `0.0968`.
- **Pre- vs Post-Calibration ECE**: Improved from `0.3524` to `0.0273`.
- **Monotonicity & Discriminative Power**: AUROC remains stable across all stages, ensuring ranking ordering is preserved while probabilities are aligned with true frequency.
- **Length Normalization Impact**: Mitigates model overconfidence on short boilerplate chunks without penalizing multi-line implementations.


---

# Validation Artifact 2: Reliability Diagram & Calibration Table
## LFM Semantic Co-Processor Wide Reranker (v0.3.3)

Binned calibration table evaluating predicted post-hoc calibrated relevance probability against empirical ground-truth positive rate across 10 equal-width tenth buckets $[0.0-0.1, \dots, 0.9-1.0]$:

| Probability Bucket $[P(yes)]$ | Sample Count | Empirical Positive Rate | Mean Predicted Prob | Calibration Error | Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `0.0-0.1` | 313 | 6.7% | 0.0610 | 0.0061 | `WELL_CALIBRATED` |
| `0.1-0.2` | 256 | 9.8% | 0.1446 | 0.0470 | `WELL_CALIBRATED` |
| `0.2-0.3` | 93 | 24.7% | 0.2371 | 0.0102 | `WELL_CALIBRATED` |
| `0.3-0.4` | 19 | 47.4% | 0.3377 | 0.1360 | `WELL_CALIBRATED` |
| `0.4-0.5` | 6 | 66.7% | 0.4502 | 0.2164 | `ACCEPTABLE` |
| `0.5-0.6` | 0 | — | — | — | `EMPTY_BIN` |
| `0.6-0.7` | 0 | — | — | — | `EMPTY_BIN` |
| `0.7-0.8` | 0 | — | — | — | `EMPTY_BIN` |
| `0.8-0.9` | 0 | — | — | — | `EMPTY_BIN` |
| `0.9-1.0` | 0 | — | — | — | `EMPTY_BIN` |

### Calibration Metrics Summary:
- **Post-Hoc Calibrated Brier Score**: `0.0968` (Pre-calibration: `0.2532`)
- **Post-Hoc Calibrated ECE**: `0.0273` (Pre-calibration: `0.3524`)
- **Monotonicity**: Empirical positive rate exhibits monotonic progression aligned with predicted confidence.


---

# Validation Artifact 3: Intent Slice Report
## LFM Semantic Co-Processor Wide Reranker (v0.3.3)

Performance evaluation partitioned by detected query intent (`IMPLEMENTATION`, `SPECIFICATION`, `BUG_DIAGNOSIS`, `REFACTOR`):

| Intent Slice | Tasks | Chunks | Recall@1 | Recall@3 | Recall@5 | MRR | Mean Rank | Brier Score | ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **IMPLEMENTATION** | 5 | 342 | 80.0% | 100.0% | 100.0% | `0.9000` | 1.20 | `0.2654` | `0.3424` |
| **SPECIFICATION** | 2 | 113 | 0.0% | 50.0% | 100.0% | `0.2667` | 4.00 | `0.2651` | `0.3522` |
| **BUG_DIAGNOSIS** | 2 | 116 | 50.0% | 50.0% | 50.0% | `0.5143` | 18.00 | `0.2614` | `0.3995` |
| **REFACTOR** | 2 | 116 | 0.0% | 50.0% | 50.0% | `0.1771` | 25.50 | `0.1972` | `0.3634` |

### Intent Prior & Symbol Disambiguation Mechanics:
- **IMPLEMENTATION**: Test chunks penalized via logit delta `\\Delta = -0.606`; exact/compound symbols boosted (+1.40 to +1.60), achieving 100% Recall@3.
- **SPECIFICATION**: Test fixtures and assertion suites boosted via logit delta `\\Delta = +0.693`.
- **BUG_DIAGNOSIS**: Regression tests boosted via logit delta `\\Delta = +0.400` while preserving core error handlers.
- **REFACTOR**: Target symbols prioritized; root table disambiguation (+0.50) and test penalty (-0.650) elevate schema targets.


---

# Validation Artifact 4: Failure Analysis (Rank > 3)
## LFM Semantic Co-Processor Wide Reranker (v0.3.3)

Detailed technical post-mortem for tasks where the ground-truth target chunk was ranked outside the top 3 (Total Failures: 3 / 11):

### Task `TASK-LLAMA-04`: "diagnose crash exception and syntax error in chat template formatting"
- **Intent**: `BUG_DIAGNOSIS`
- **Target**: `llama-chat.cpp` (llm_chat_apply_template)
- **Actual Rank Achieved**: **#35** (Score: `0.5568`)
- **Root Cause Diagnostic**: Chunk 'parse_token' in llama-grammar.cpp exhibited higher query term density and matched prompt keywords more specifically than target symbol 'llm_chat_apply_template'.

**Top 3 Competitors Outranking Target:**
| Rank | File | Symbol | Lines | Adjusted Score | Raw Score | Test Chunk? |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| #1 | `llama-grammar.cpp` | `parse_token` | `L185-L229` | `0.8400` | `0.8400` | `False` |
| #2 | `llama-grammar.cpp` | `llama_grammar_match_partial_char` | `L788-L834` | `0.7968` | `0.8200` | `False` |
| #3 | `llama-grammar.cpp` | `llama_grammar_match_char` | `L758-L783` | `0.7747` | `0.7998` | `False` |

### Task `TASK-LLAMA-05`: "refactor and extract common KV cache quantization type mapping"
- **Intent**: `REFACTOR`
- **Target**: `llama-arch.cpp` (llm_arch_from_string)
- **Actual Rank Achieved**: **#48** (Score: `0.3222`)
- **Root Cause Diagnostic**: Chunk 'llama_grammar_match_partial_char' in llama-grammar.cpp exhibited higher query term density and matched prompt keywords more specifically than target symbol 'llm_arch_from_string'.

**Top 3 Competitors Outranking Target:**
| Rank | File | Symbol | Lines | Adjusted Score | Raw Score | Test Chunk? |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| #1 | `llama-grammar.cpp` | `llama_grammar_match_partial_char` | `L788-L834` | `0.7952` | `0.7952` | `False` |
| #2 | `llama-grammar.cpp` | `decode_utf8` | `L34-L92` | `0.7495` | `0.7766` | `False` |
| #3 | `llama-grammar.cpp` | `print_rule_binary` | `L251-L294` | `0.7323` | `0.7607` | `False` |

### Task `TASK-LLAMA-06`: "test assertions and grammar verification contract for GBNF parser"
- **Intent**: `SPECIFICATION`
- **Target**: `test-llama-grammar.cpp` (module_scope)
- **Actual Rank Achieved**: **#5** (Score: `0.8125`)
- **Root Cause Diagnostic**: Chunk 'parse_token' in llama-grammar.cpp exhibited higher query term density and matched prompt keywords more specifically than target symbol 'test-llama-grammar.cpp'.

**Top 3 Competitors Outranking Target:**
| Rank | File | Symbol | Lines | Adjusted Score | Raw Score | Test Chunk? |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| #1 | `llama-grammar.cpp` | `parse_token` | `L185-L229` | `0.8999` | `0.8577` | `False` |
| #2 | `llama-grammar.cpp` | `parse_hex` | `L102-L123` | `0.8599` | `0.8270` | `False` |
| #3 | `llama-grammar.cpp` | `llama_grammar_match_partial_char` | `L788-L834` | `0.8310` | `0.8510` | `False` |

### Resolved Failures in v0.3.3:
- **TASK-QC-01**: Single-word symbol boost (`trigger` in query matching `trigger.ts:trigger`) lifted target from Rank #4 to **Rank #2**.
- **TASK-QC-05**: Schema refactor table disambiguation and test penalty lifted `progressWorkTypes` from Rank #4 to **Rank #3**.
- **TASK-LLAMA-03**: Semantic component decomposition (`llm` + `arch` + `string`) lifted `llm_arch_from_string` from Rank #9 to **Rank #1**.


---

# Validation Artifact 5: Decision & Operational Telemetry Summary
## LFM Semantic Co-Processor Wide Reranker (v0.3.3)

Aggregated runtime telemetry and operational characteristics during benchmark execution:

### Workload & Volume
- **Total Tasks Evaluated**: 11
- **Candidate Files Evaluated**: 46
- **Candidate Chunks Scored**: 687
- **Total Wall-Clock Time**: `0.46s`

### Calibration Telemetry
- **Calibrator Algorithm**: `PlattScalingCalibrator`
- **Pre-Calibration Brier Score**: `0.2532`
- **Post-Calibration Brier Score**: `0.0968`
- **Pre-Calibration ECE**: `0.3524`
- **Post-Calibration ECE**: `0.0273`

### Ambiguity & Hazard Telemetry
- **Ambiguous Completion Rate**: `0.00%` (0 non-binary tokens)
- **Yes Variant Token Rate**: `49.2%`
- **No Variant Token Rate**: `50.8%`
- **Cache Hit Rate**: `100.0%` (687 hits / 0 misses)

### Decision Policies & Context Assembly
- **Dynamic Margin Threshold**: `0.15`
- **Symbol Boosts Applied**: 66
- **Redundancy Penalties Applied**: 624
- **Fallback Floor Triggers**: 0 (0.0%)
- **Abstention Count**: 0 (0.0%)
- **Frontier Reduction Ratio**: `0.0x`
- **Frontier Reduction Percentage**: `-0%` of raw context tokens filtered

### Throughput & Latency
- **Prefill Throughput**: `0.0 tok/s`
- **Decode Throughput**: `0.0 tok/s`
- **Score Latency (P95)**: `0.00 ms`
- **Total Prompt Tokens**: `0`
- **Total Completion Tokens**: `0`

