from __future__ import annotations


import warnings


from typing import Dict, Optional

import numpy as np

from scipy.stats import multivariate_normal, shapiro
from sklearn.covariance import LedoitWolf


# NumPy 2.0+ compatibility: trapz was removed and replaced with trapezoid
# Add backward compatibility for older NumPy versions
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # type: ignore


class GaussianStateDistribution:
    """Gaussian model with Ledoit-Wolf shrinkage and diagnostics."""

    def __init__(self, label: str, data: np.ndarray, test_normality: bool = True):
        self.label = label
        self.data = np.asarray(data, dtype=np.float64)

        if self.data.ndim != 2:
            raise ValueError("State data must be 2D: (samples, features)")

        self.n_samples, self.n_features = self.data.shape
        self.normality_test_results: Optional[Dict] = None

        if test_normality:
            self._test_normality()

        self._fit()

    def _test_normality(self) -> None:
        """Test multivariate normality using Shapiro-Wilk on marginals."""
        non_normal_features = []

        # Test first 10 features (full test is expensive)
        for feat_idx in range(min(10, self.n_features)):
            if self.n_samples >= 3:  # shapiro requires n >= 3
                try:
                    stat, p = shapiro(self.data[:, feat_idx])
                    if p < 0.01:  # strict threshold
                        non_normal_features.append((feat_idx, p))
                except Exception:
                    pass

        if non_normal_features:
            warnings.warn(
                f"State '{self.label}': {len(non_normal_features)}/{min(10, self.n_features)} "
                f"features deviate from normality (p<0.01). "
                f"Gaussian assumptions may be violated."
            )

        self.normality_test_results = {"non_normal": non_normal_features}

    def _fit(self) -> None:
        """Fit Gaussian with shrinkage covariance."""
        if self.n_samples < 2 * self.n_features:
            warnings.warn(
                f"State '{self.label}': n_samples={self.n_samples} < "
                f"2*n_features={2 * self.n_features}. "
                f"Covariance estimation may be unreliable."
            )

        self.mean = self.data.mean(axis=0)

        try:
            lw = LedoitWolf().fit(self.data)
            self.cov = lw.covariance_
            # Add jitter for numerical stability
            self.cov = self.cov + 1e-8 * np.eye(self.n_features)
        except Exception as e:
            warnings.warn(
                f"Ledoit-Wolf failed for '{self.label}': {e}. Using diagonal covariance."
            )
            self.cov = np.diag(np.var(self.data, axis=0) + 1e-8)

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        """Log probability density."""
        return multivariate_normal.logpdf(
            x, mean=self.mean, cov=self.cov, allow_singular=False
        )

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Probability density."""
        return multivariate_normal.pdf(
            x, mean=self.mean, cov=self.cov, allow_singular=False
        )

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sample from distribution."""
        return rng.multivariate_normal(self.mean, self.cov, size=n)
