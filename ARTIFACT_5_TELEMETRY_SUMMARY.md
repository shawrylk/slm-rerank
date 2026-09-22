# Validation Artifact 5: Decision & Operational Telemetry Summary
## LFM Semantic Co-Processor Wide Reranker (v0.3.3)

Aggregated runtime telemetry and operational characteristics during benchmark execution:

### Workload & Volume
- **Total Tasks Evaluated**: 11
- **Candidate Files Evaluated**: 46
- **Candidate Chunks Scored**: 652
- **Total Wall-Clock Time**: `9.82s`

### Calibration Telemetry
- **Calibrator Algorithm**: `PlattScalingCalibrator`
- **Pre-Calibration Brier Score**: `0.2213`
- **Post-Calibration Brier Score**: `0.0979`
- **Pre-Calibration ECE**: `0.3102`
- **Post-Calibration ECE**: `0.0320`

### Ambiguity & Hazard Telemetry
- **Ambiguous Completion Rate**: `3.83%` (25 non-binary tokens)
- **Yes Variant Token Rate**: `45.9%`
- **No Variant Token Rate**: `50.7%`
- **Cache Hit Rate**: `95.5%` (623 hits / 29 misses)

### Decision Policies & Context Assembly
- **Dynamic Margin Threshold**: `0.15`
- **Symbol Boosts Applied**: 73
- **Redundancy Penalties Applied**: 568
- **Fallback Floor Triggers**: 0 (0.0%)
- **Abstention Count**: 0 (0.0%)
- **Frontier Reduction Ratio**: `6.2x`
- **Frontier Reduction Percentage**: `-82%` of raw context tokens filtered

### Throughput & Latency
- **Prefill Throughput**: `27,400.5 tok/s`
- **Decode Throughput**: `3.9 tok/s`
- **Score Latency (P95)**: `1833.88 ms`
- **Total Prompt Tokens**: `274,594`
- **Total Completion Tokens**: `29`
