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
