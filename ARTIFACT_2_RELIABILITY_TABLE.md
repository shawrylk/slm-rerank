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
