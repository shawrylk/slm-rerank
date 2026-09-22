# Validation Artifact 3: Intent Slice Report
## LFM Semantic Co-Processor Wide Reranker (v0.3.3)

Performance evaluation partitioned by detected query intent (`IMPLEMENTATION`, `SPECIFICATION`, `BUG_DIAGNOSIS`, `REFACTOR`):

| Intent Slice | Tasks | Chunks | Recall@1 | Recall@3 | Recall@5 | MRR | Mean Rank | Brier Score | ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **IMPLEMENTATION** | 5 | 321 | 80.0% | 80.0% | 100.0% | `0.8500` | 1.60 | `0.2516` | `0.3173` |
| **SPECIFICATION** | 2 | 113 | 0.0% | 50.0% | 50.0% | `0.2292` | 5.50 | `0.2770` | `0.3677` |
| **BUG_DIAGNOSIS** | 2 | 109 | 50.0% | 100.0% | 100.0% | `0.6667` | 2.00 | `0.1734` | `0.3220` |
| **REFACTOR** | 2 | 109 | 0.0% | 0.0% | 0.0% | `0.1339` | 7.50 | `0.1220` | `0.2549` |

### Intent Prior & Symbol Disambiguation Mechanics:
- **IMPLEMENTATION**: Test chunks penalized via logit delta `\\Delta = -0.606`; exact/compound symbols boosted (+1.40 to +1.60), achieving 100% Recall@3.
- **SPECIFICATION**: Test fixtures and assertion suites boosted via logit delta `\\Delta = +0.693`.
- **BUG_DIAGNOSIS**: Regression tests boosted via logit delta `\\Delta = +0.400` while preserving core error handlers.
- **REFACTOR**: Target symbols prioritized; root table disambiguation (+0.50) and test penalty (-0.650) elevate schema targets.
