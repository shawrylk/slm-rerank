"""Unit tests for formal calibration metrics: Brier score, ECE, and reliability table."""

import pytest
from lfm_rerank.eval import (
    calculate_auprc,
    calculate_auroc,
    calculate_brier_score,
    calculate_ece,
    generate_reliability_table,
)


def test_brier_score_perfect_predictions():
    y_true = [1, 0, 1, 0]
    y_prob = [1.0, 0.0, 1.0, 0.0]
    assert calculate_brier_score(y_true, y_prob) == 0.0


def test_brier_score_worst_predictions():
    y_true = [1, 0]
    y_prob = [0.0, 1.0]
    assert calculate_brier_score(y_true, y_prob) == 1.0


def test_brier_score_intermediate():
    y_true = [1, 0]
    y_prob = [0.8, 0.2]
    # ((0.8 - 1.0)^2 + (0.2 - 0.0)^2) / 2 = (0.04 + 0.04) / 2 = 0.04
    assert calculate_brier_score(y_true, y_prob) == 0.04


def test_brier_score_empty_and_mismatch():
    assert calculate_brier_score([], []) == 0.0
    with pytest.raises(ValueError, match="Length mismatch"):
        calculate_brier_score([1], [0.5, 0.8])


def test_ece_perfect_calibration():
    # 5 samples at 0.8 where 4 are positive -> acc=0.8, conf=0.8 -> error=0.0
    # 5 samples at 0.2 where 1 is positive -> acc=0.2, conf=0.2 -> error=0.0
    y_true = [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]
    y_prob = [0.8, 0.8, 0.8, 0.8, 0.8, 0.2, 0.2, 0.2, 0.2, 0.2]
    ece = calculate_ece(y_true, y_prob, n_bins=10)
    assert ece == 0.0


def test_ece_miscalibrated():
    # Model predicts 0.9 confidence for everything, but all targets are negative (0)
    y_true = [0, 0, 0, 0]
    y_prob = [0.95, 0.95, 0.95, 0.95]
    ece = calculate_ece(y_true, y_prob, n_bins=10)
    assert ece == 0.95


def test_ece_empty_and_mismatch():
    assert calculate_ece([], [], n_bins=10) == 0.0
    with pytest.raises(ValueError, match="Length mismatch"):
        calculate_ece([1], [0.5, 0.2], n_bins=10)


def test_generate_reliability_table_structure():
    y_true = [1, 0, 1, 0]
    y_prob = [0.15, 0.15, 0.85, 0.85]
    table = generate_reliability_table(y_true, y_prob, n_bins=10)

    assert len(table) == 10
    bucket_labels = [row["bucket"] for row in table]
    assert bucket_labels == [
        "0.0-0.1",
        "0.1-0.2",
        "0.2-0.3",
        "0.3-0.4",
        "0.4-0.5",
        "0.5-0.6",
        "0.6-0.7",
        "0.7-0.8",
        "0.8-0.9",
        "0.9-1.0",
    ]

    # Check 0.1-0.2 bucket (2 samples: [1, 0], prob=[0.15, 0.15])
    bin_1 = table[1]
    assert bin_1["sample_count"] == 2
    assert bin_1["empirical_positive_rate"] == 0.5
    assert bin_1["mean_predicted_prob"] == 0.15
    assert bin_1["calibration_error"] == 0.35

    # Check 0.8-0.9 bucket (2 samples: [1, 0], prob=[0.85, 0.85])
    bin_8 = table[8]
    assert bin_8["sample_count"] == 2
    assert bin_8["empirical_positive_rate"] == 0.5
    assert bin_8["mean_predicted_prob"] == 0.85
    assert bin_8["calibration_error"] == 0.35

    # Check an empty bucket
    empty_bin = table[0]
    assert empty_bin["sample_count"] == 0
    assert empty_bin["empirical_positive_rate"] is None
    assert empty_bin["calibration_error"] is None


def test_auroc_metrics():
    # Perfect discrimination
    y_true = [1, 0, 1, 0]
    y_prob = [0.9, 0.1, 0.8, 0.2]
    assert calculate_auroc(y_true, y_prob) == 1.0

    # Inverted discrimination
    assert calculate_auroc([1, 0], [0.1, 0.9]) == 0.0

    # Tied discrimination -> 0.5
    assert calculate_auroc([1, 0], [0.5, 0.5]) == 0.5

    # Empty / mismatch edge cases
    assert calculate_auroc([], []) == 0.0
    with pytest.raises(ValueError, match="Length mismatch"):
        calculate_auroc([1], [0.5, 0.8])


def test_auprc_metrics():
    # Perfect discrimination
    y_true = [1, 0, 1, 0]
    y_prob = [0.9, 0.1, 0.8, 0.2]
    assert calculate_auprc(y_true, y_prob) == 1.0

    # Inverted discrimination
    assert calculate_auprc([1, 0], [0.1, 0.9]) == 0.5

    # Empty / mismatch edge cases
    assert calculate_auprc([], []) == 0.0
    assert calculate_auprc([0, 0], [0.5, 0.8]) == 0.0
    with pytest.raises(ValueError, match="Length mismatch"):
        calculate_auprc([1], [0.5, 0.8])

