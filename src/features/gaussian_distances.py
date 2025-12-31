from __future__ import annotations
from typing import Tuple
import numpy as np

from representation.gaussian_state_distribution import GaussianStateDistribution

# NumPy 2.0+ compatibility: trapz was removed and replaced with trapezoid
# Add backward compatibility for older NumPy versions
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # type: ignore


class GaussianDistances:
    """Information-theoretic distances between Gaussians."""

    @staticmethod
    def _slogdet(cov: np.ndarray) -> Tuple[float, float]:
        """Safe log-determinant."""
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            raise ValueError("Covariance must be positive definite for log-det")
        return sign, logdet

    @staticmethod
    def kl_gaussian(
        p: GaussianStateDistribution, q: GaussianStateDistribution
    ) -> float:
        """KL(P||Q) in closed form for multivariate Gaussians."""
        k = p.n_features
        inv_q = np.linalg.inv(q.cov)
        diff = (q.mean - p.mean).reshape(-1, 1)

        _, logdet_p = GaussianDistances._slogdet(p.cov)
        _, logdet_q = GaussianDistances._slogdet(q.cov)

        term_trace = np.trace(inv_q @ p.cov)
        term_quad = float(
            (diff.T @ inv_q @ diff).item()
        )  # Extract scalar from 1x1 array
        term_logdet = logdet_q - logdet_p

        return 0.5 * (term_trace + term_quad - k + term_logdet)

    @staticmethod
    def symmetric_kl(
        p: GaussianStateDistribution, q: GaussianStateDistribution
    ) -> float:
        """Symmetric KL divergence: 0.5 * (KL(P||Q) + KL(Q||P))."""
        return 0.5 * (
            GaussianDistances.kl_gaussian(p, q) + GaussianDistances.kl_gaussian(q, p)
        )

    @staticmethod
    def hellinger(p: GaussianStateDistribution, q: GaussianStateDistribution) -> float:
        """Hellinger distance for multivariate Gaussians."""
        cov_avg = 0.5 * (p.cov + q.cov)
        inv_avg = np.linalg.inv(cov_avg)
        diff = (p.mean - q.mean).reshape(-1, 1)

        _, logdet_p = GaussianDistances._slogdet(p.cov)
        _, logdet_q = GaussianDistances._slogdet(q.cov)
        _, logdet_avg = GaussianDistances._slogdet(cov_avg)

        log_bc = (
            0.25 * (logdet_p + logdet_q)
            - 0.5 * logdet_avg
            - 0.125
            * float((diff.T @ inv_avg @ diff).item())  # Extract scalar from 1x1 array
        )

        bc = float(np.exp(log_bc))
        h2 = max(0.0, 1.0 - bc)
        return float(np.sqrt(h2))

    @staticmethod
    def jsd_monte_carlo(
        p: GaussianStateDistribution,
        q: GaussianStateDistribution,
        rng: np.random.Generator,
        n_samples: int = 5000,
        n_bootstrap: int = 50,
    ) -> Tuple[float, float, float]:
        """Monte Carlo JSD with bootstrap confidence interval.

        Returns:
            (jsd_mean, jsd_ci_low, jsd_ci_high)
        """
        jsd_estimates = []

        for _ in range(n_bootstrap):
            sp = p.sample(n_samples, rng)
            sq = q.sample(n_samples, rng)

            logp_sp = p.log_pdf(sp)
            logq_sp = q.log_pdf(sp)
            logm_sp = np.logaddexp(logp_sp, logq_sp) - np.log(2.0)
            kl_p_m = float(np.mean(logp_sp - logm_sp))

            logp_sq = p.log_pdf(sq)
            logq_sq = q.log_pdf(sq)
            logm_sq = np.logaddexp(logp_sq, logq_sq) - np.log(2.0)
            kl_q_m = float(np.mean(logq_sq - logm_sq))

            jsd = 0.5 * (kl_p_m + kl_q_m)
            jsd_estimates.append(max(jsd, 0.0))

        jsd_estimates = np.array(jsd_estimates)
        return (
            float(np.mean(jsd_estimates)),
            float(np.percentile(jsd_estimates, 2.5)),
            float(np.percentile(jsd_estimates, 97.5)),
        )
