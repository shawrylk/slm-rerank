"""Post-hoc calibration layer for LFM Semantic Co-Processor Wide Reranker.

Implements TemperatureScalingCalibrator and PlattScalingCalibrator (1D logistic regression
on logits with cross-validation and L2 regularization).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


def sigmoid(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Numerically stable sigmoid function."""
    if isinstance(z, (int, float)):
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        else:
            ez = math.exp(z)
            return ez / (1.0 + ez)
    z_arr = np.asarray(z, dtype=np.float64)
    return np.where(z_arr >= 0, 1.0 / (1.0 + np.exp(-z_arr)), np.exp(z_arr) / (1.0 + np.exp(z_arr)))


def logit(p: Union[float, np.ndarray], eps: float = 1e-6) -> Union[float, np.ndarray]:
    """Numerically clamped log-odds (logit) function."""
    if isinstance(p, (int, float)):
        clamped = max(eps, min(1.0 - eps, float(p)))
        return math.log(clamped / (1.0 - clamped))
    p_arr = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p_arr / (1.0 - p_arr))


class BaseCalibrator:
    """Base interface for post-hoc probability calibrators."""

    def fit(self, logits: Sequence[float], y_true: Sequence[Union[int, float]]) -> "BaseCalibrator":
        raise NotImplementedError

    def calibrate(self, logit_val: float) -> float:
        raise NotImplementedError

    def calibrate_proba(self, raw_prob: float) -> float:
        """Calibrate a probability by converting to logit first."""
        z = float(logit(raw_prob))
        return self.calibrate(z)

    def predict_proba(self, logits: Sequence[float]) -> List[float]:
        return [round(self.calibrate(float(z)), 4) for z in logits]


class TemperatureScalingCalibrator(BaseCalibrator):
    """Post-hoc calibrator using single temperature parameter: p_cal = sigmoid(z / T)."""

    def __init__(self, temperature: float = 1.0):
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        self.temperature: float = float(temperature)

    def calibrate(self, logit_val: float) -> float:
        z = logit_val / self.temperature
        return round(float(sigmoid(z)), 4)

    def fit(
        self,
        logits: Sequence[float],
        y_true: Sequence[Union[int, float]],
        t_bounds: Tuple[float, float] = (0.05, 10.0),
        num_steps: int = 200,
    ) -> "TemperatureScalingCalibrator":
        """Optimize temperature T on (logits, y_true) by minimizing binary cross-entropy loss."""
        z_arr = np.asarray(logits, dtype=np.float64)
        y_arr = np.asarray(y_true, dtype=np.float64)

        if len(z_arr) == 0:
            return self

        best_t = 1.0
        best_nll = float("inf")

        # Grid search with fine golden-section refinement
        t_grid = np.linspace(t_bounds[0], t_bounds[1], num_steps)
        for t in t_grid:
            p = sigmoid(z_arr / t)
            p_clamped = np.clip(p, 1e-12, 1.0 - 1e-12)
            nll = -np.mean(y_arr * np.log(p_clamped) + (1.0 - y_arr) * np.log(1.0 - p_clamped))
            if nll < best_nll:
                best_nll = nll
                best_t = float(t)

        self.temperature = round(best_t, 4)
        return self

    def fit_cv(
        self,
        logits: Sequence[float],
        y_true: Sequence[Union[int, float]],
        cv: int = 5,
        t_bounds: Tuple[float, float] = (0.05, 10.0),
    ) -> "TemperatureScalingCalibrator":
        """K-fold cross-validation to optimize temperature T."""
        z_arr = np.asarray(logits, dtype=np.float64)
        y_arr = np.asarray(y_true, dtype=np.float64)
        n = len(z_arr)
        if n < cv or cv <= 1:
            return self.fit(logits, y_true, t_bounds=t_bounds)

        indices = np.arange(n)
        folds = np.array_split(indices, cv)

        fold_temperatures = []
        for i in range(cv):
            val_idx = folds[i]
            train_idx = np.setdiff1d(indices, val_idx)
            cal_fold = TemperatureScalingCalibrator()
            cal_fold.fit(z_arr[train_idx], y_arr[train_idx], t_bounds=t_bounds)
            fold_temperatures.append(cal_fold.temperature)

        self.temperature = round(float(np.median(fold_temperatures)), 4)
        return self


    def predict_proba_oof(
        self,
        logits: Sequence[float],
        y_true: Sequence[Union[int, float]],
        cv: int = 5,
        seed: int = 42,
    ) -> List[float]:
        """Compute out-of-fold calibrated probabilities across K-fold CV."""
        return compute_oof_predictions(logits, y_true, cv=cv, seed=seed, calibrator_factory=TemperatureScalingCalibrator)


class PlattScalingCalibrator(BaseCalibrator):
    """Post-hoc calibrator using 1D logistic regression on logits: p_cal = sigmoid(a * z + b).

    Optimized via Newton-Raphson / IRLS with L2 regularization and cross-validation support.
    """

    def __init__(self, a: float = 1.0, b: float = 0.0):
        self.a: float = float(a)
        self.b: float = float(b)

    def calibrate(self, logit_val: float) -> float:
        z = self.a * logit_val + self.b
        return round(float(sigmoid(z)), 4)

    def fit(
        self,
        logits: Sequence[float],
        y_true: Sequence[Union[int, float]],
        l2_reg: float = 1e-3,
        max_iter: int = 50,
        tol: float = 1e-6,
    ) -> "PlattScalingCalibrator":
        """Fit slope a and intercept b on (logits, y_true) using Newton-Raphson."""
        z_arr = np.asarray(logits, dtype=np.float64)
        y_arr = np.asarray(y_true, dtype=np.float64)
        n = len(z_arr)
        if n == 0:
            return self

        # Design matrix X = [z, 1]
        X = np.column_stack([z_arr, np.ones(n, dtype=np.float64)])
        theta = np.array([1.0, 0.0], dtype=np.float64)  # initial [a=1.0, b=0.0]

        reg_matrix = np.diag([l2_reg, 0.0])  # Do not regularize bias term heavily

        for _ in range(max_iter):
            z_pred = X @ theta
            p = sigmoid(z_pred)
            p = np.clip(p, 1e-12, 1.0 - 1e-12)

            # Gradient: g = X^T (p - y) + reg * theta
            g = X.T @ (p - y_arr) + reg_matrix @ theta

            # Hessian: H = X^T W X + reg_matrix
            w = p * (1.0 - p)
            H = (X.T * w) @ X + reg_matrix

            try:
                delta = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                break

            theta_new = theta - delta
            if np.max(np.abs(delta)) < tol:
                theta = theta_new
                break
            theta = theta_new

        # Ensure slope is non-negative to maintain ranking monotonicity
        self.a = max(0.01, round(float(theta[0]), 4))
        self.b = round(float(theta[1]), 4)
        return self

    def fit_cv(
        self,
        logits: Sequence[float],
        y_true: Sequence[Union[int, float]],
        cv: int = 5,
        l2_regs: Sequence[float] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0),
    ) -> "PlattScalingCalibrator":
        """K-fold cross-validation to select optimal L2 regularization and fit parameters."""
        z_arr = np.asarray(logits, dtype=np.float64)
        y_arr = np.asarray(y_true, dtype=np.float64)
        n = len(z_arr)
        if n < cv or cv <= 1:
            return self.fit(logits, y_true)

        indices = np.arange(n)
        folds = np.array_split(indices, cv)

        best_l2 = l2_regs[0]
        best_brier = float("inf")

        for reg in l2_regs:
            fold_briers = []
            for i in range(cv):
                val_idx = folds[i]
                train_idx = np.setdiff1d(indices, val_idx)
                cal = PlattScalingCalibrator()
                cal.fit(z_arr[train_idx], y_arr[train_idx], l2_reg=reg)
                preds = [cal.calibrate(z) for z in z_arr[val_idx]]
                brier = float(np.mean((np.array(preds) - y_arr[val_idx]) ** 2))
                fold_briers.append(brier)
            avg_brier = float(np.mean(fold_briers))
            if avg_brier < best_brier:
                best_brier = avg_brier
                best_l2 = reg

        # Refit on full dataset with best regularization
        return self.fit(logits, y_true, l2_reg=best_l2)

    def predict_proba_oof(
        self,
        logits: Sequence[float],
        y_true: Sequence[Union[int, float]],
        cv: int = 5,
        seed: int = 42,
    ) -> List[float]:
        """Compute out-of-fold calibrated probabilities across K-fold CV."""
        return compute_oof_predictions(logits, y_true, cv=cv, seed=seed, calibrator_factory=PlattScalingCalibrator)


def compute_brier_skill_score(brier_model: float, y_true: Sequence[Any]) -> float:
    """Compute Brier Skill Score (BSS) against trivial base-rate predictor:
    BSS = 1 - (brier_model / brier_baseline)
    where brier_baseline = p * (1 - p) with p = mean(y_true).
    """
    if len(y_true) == 0:
        return 0.0
    p_base = float(np.mean([float(y) for y in y_true]))
    brier_baseline = p_base * (1.0 - p_base)
    if brier_baseline <= 1e-12:
        return 0.0
    return round(1.0 - (float(brier_model) / brier_baseline), 4)


def compute_sharpness_distribution(y_prob: Sequence[float]) -> Dict[str, float]:
    """Compute sharpness distribution metrics for probability predictions:
    min, median, p75, max, % chunks > 0.30, % > 0.50, % > 0.70.
    """
    if len(y_prob) == 0:
        return {
            "min": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "max": 0.0,
            "pct_above_30": 0.0,
            "pct_above_50": 0.0,
            "pct_above_70": 0.0,
        }
    arr = np.asarray(y_prob, dtype=np.float64)
    n = len(arr)
    return {
        "min": round(float(np.min(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4),
        "max": round(float(np.max(arr)), 4),
        "pct_above_30": round(float(np.sum(arr > 0.30) / n * 100.0), 2),
        "pct_above_50": round(float(np.sum(arr > 0.50) / n * 100.0), 2),
        "pct_above_70": round(float(np.sum(arr > 0.70) / n * 100.0), 2),
    }


def compute_oof_predictions(
    logits: Sequence[float],
    y_true: Sequence[Union[int, float]],
    cv: int = 5,
    seed: int = 42,
    calibrator_factory: Optional[Any] = None,
) -> List[float]:
    """Compute Out-of-Fold (OOF) calibrated predictions via K-fold cross-validation.
    Predictions for each fold are generated strictly by a calibrator fitted on remaining folds.
    """
    z_arr = np.asarray(logits, dtype=np.float64)
    y_arr = np.asarray(y_true, dtype=np.float64)
    n = len(z_arr)
    if n == 0:
        return []
    if n < cv or cv <= 1:
        factory = calibrator_factory or PlattScalingCalibrator
        cal = factory()
        cal.fit(z_arr, y_arr)
        return cal.predict_proba(z_arr)

    oof = np.zeros(n, dtype=np.float64)
    pos_idx = np.where(y_arr == 1.0)[0]
    neg_idx = np.where(y_arr == 0.0)[0]

    rng = np.random.RandomState(seed)
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    pos_folds = np.array_split(pos_idx, cv)
    neg_folds = np.array_split(neg_idx, cv)

    factory = calibrator_factory or PlattScalingCalibrator

    for i in range(cv):
        val_idx = np.sort(np.concatenate([pos_folds[i], neg_folds[i]]))
        train_idx = np.setdiff1d(np.arange(n), val_idx)

        cal = factory()
        cal.fit(z_arr[train_idx], y_arr[train_idx])
        preds = [cal.calibrate(z) for z in z_arr[val_idx]]
        oof[val_idx] = preds

    return [round(float(p), 4) for p in oof]


def compute_task_heldout_predictions(
    task_ids: Sequence[str],
    logits: Sequence[float],
    y_true: Sequence[Union[int, float]],
    train_task_ids: Sequence[str],
    test_task_ids: Sequence[str],
    calibrator_factory: Optional[Any] = None,
) -> Tuple[List[float], List[float], List[float]]:
    """Fit calibrator on train_task_ids and predict strictly on unseen test_task_ids.
    Returns (test_y_true, test_preds, test_logits).
    """
    z_arr = np.asarray(logits, dtype=np.float64)
    y_arr = np.asarray(y_true, dtype=np.float64)
    tasks = list(task_ids)

    train_mask = np.array([t in train_task_ids for t in tasks])
    test_mask = np.array([t in test_task_ids for t in tasks])

    factory = calibrator_factory or PlattScalingCalibrator
    cal = factory()
    cal.fit(z_arr[train_mask], y_arr[train_mask])

    test_preds = [round(float(cal.calibrate(z)), 4) for z in z_arr[test_mask]]
    test_y = [float(y) for y in y_arr[test_mask]]
    test_z = [float(z) for z in z_arr[test_mask]]

    return test_y, test_preds, test_z


_DEFAULT_CALIBRATOR: Optional[BaseCalibrator] = None


def get_default_calibrator() -> BaseCalibrator:
    """Retrieve default PlattScalingCalibrator for LFM wide reranker."""
    global _DEFAULT_CALIBRATOR
    if _DEFAULT_CALIBRATOR is None:
        # Pre-calibrated parameters for LFM 2.5 on code retrieval tasks
        _DEFAULT_CALIBRATOR = PlattScalingCalibrator(a=0.60, b=-2.10)
    return _DEFAULT_CALIBRATOR


def set_default_calibrator(calibrator: BaseCalibrator) -> None:
    """Set global default calibrator."""
    global _DEFAULT_CALIBRATOR
    _DEFAULT_CALIBRATOR = calibrator
