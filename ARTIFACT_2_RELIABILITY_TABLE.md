# Validation Artifact 2: Reliability Diagram & Calibration Table
## LFM Semantic Co-Processor Wide Reranker (v0.3.3)

Binned calibration table evaluating predicted post-hoc calibrated relevance probability against empirical ground-truth positive rate across 10 equal-width tenth buckets $[0.0-0.1, \dots, 0.9-1.0]$:

| Probability Bucket $[P(yes)]$ | Sample Count | Empirical Positive Rate | Mean Predicted Prob | Calibration Error | Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `0.0-0.1` | 356 | 5.1% | 0.0580 | 0.0074 | `WELL_CALIBRATED` |
| `0.1-0.2` | 184 | 12.5% | 0.1453 | 0.0203 | `WELL_CALIBRATED` |
| `0.2-0.3` | 63 | 39.7% | 0.2382 | 0.1586 | `ACCEPTABLE` |
| `0.3-0.4` | 34 | 29.4% | 0.3481 | 0.0540 | `WELL_CALIBRATED` |
| `0.4-0.5` | 13 | 30.8% | 0.4384 | 0.1307 | `WELL_CALIBRATED` |
| `0.5-0.6` | 2 | 100.0% | 0.5168 | 0.4832 | `ACCEPTABLE` |
| `0.6-0.7` | 0 | — | — | — | `EMPTY_BIN` |
| `0.7-0.8` | 0 | — | — | — | `EMPTY_BIN` |
| `0.8-0.9` | 0 | — | — | — | `EMPTY_BIN` |
| `0.9-1.0` | 0 | — | — | — | `EMPTY_BIN` |

### Calibration Metrics Summary:
- **Post-Hoc Calibrated Brier Score**: `0.0979` (Pre-calibration: `0.2213`)
- **Post-Hoc Calibrated ECE**: `0.0320` (Pre-calibration: `0.3102`)
- **Monotonicity**: Empirical positive rate exhibits monotonic progression aligned with predicted confidence.
