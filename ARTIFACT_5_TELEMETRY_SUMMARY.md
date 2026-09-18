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
