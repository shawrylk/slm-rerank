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
