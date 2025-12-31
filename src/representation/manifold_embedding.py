from __future__ import annotations
from typing import Dict, Literal, Optional
import numpy as np
from sklearn.manifold import MDS


from representation.gaussian_state_distribution import GaussianStateDistribution
from features.gaussian_distances import GaussianDistances

# NumPy 2.0+ compatibility: trapz was removed and replaced with trapezoid
# Add backward compatibility for older NumPy versions
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # type: ignore


class ManifoldEmbedding:
    """MDS embedding with distance computation."""

    def __init__(
        self,
        distributions: Dict[str, GaussianStateDistribution],
        metric: Literal["jsd", "skl", "hellinger"] = "jsd",
        rng_seed: int = 42,
        jsd_n_samples: int = 5000,
        jsd_n_bootstrap: int = 50,
    ):
        self.distributions = distributions
        self.metric = metric
        self.labels = list(distributions.keys())
        self.rng = np.random.default_rng(rng_seed)
        self.jsd_n_samples = jsd_n_samples
        self.jsd_n_bootstrap = jsd_n_bootstrap
        self.distance_matrix: Optional[np.ndarray] = None
        self.distance_ci_low: Optional[np.ndarray] = None
        self.distance_ci_high: Optional[np.ndarray] = None
        self.embedding_2d: Optional[np.ndarray] = None

    def compute_distance_matrix(self) -> np.ndarray:
        """Compute pairwise distances between states."""
        n = len(self.labels)
        D = np.zeros((n, n), dtype=np.float64)
        D_ci_low = np.zeros((n, n), dtype=np.float64)
        D_ci_high = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            for j in range(i + 1, n):
                p = self.distributions[self.labels[i]]
                q = self.distributions[self.labels[j]]

                if self.metric == "skl":
                    d = GaussianDistances.symmetric_kl(p, q)
                    d_ci_low = d_ci_high = d  # deterministic
                elif self.metric == "hellinger":
                    d = GaussianDistances.hellinger(p, q)
                    d_ci_low = d_ci_high = d  # deterministic
                elif self.metric == "jsd":
                    d, d_ci_low, d_ci_high = GaussianDistances.jsd_monte_carlo(
                        p,
                        q,
                        rng=self.rng,
                        n_samples=self.jsd_n_samples,
                        n_bootstrap=self.jsd_n_bootstrap,
                    )
                else:
                    raise ValueError(f"Unknown metric: {self.metric}")

                D[i, j] = D[j, i] = d
                D_ci_low[i, j] = D_ci_low[j, i] = d_ci_low
                D_ci_high[i, j] = D_ci_high[j, i] = d_ci_high

        self.distance_matrix = D
        self.distance_ci_low = D_ci_low
        self.distance_ci_high = D_ci_high
        return D

    def embed_2d(self, random_state: int = 42) -> np.ndarray:
        """Embed distance matrix in 2D using MDS."""
        if self.distance_matrix is None:
            raise ValueError("Call compute_distance_matrix() first")

        mds = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=random_state,
            n_init=10,
            max_iter=300,
        )
        self.embedding_2d = mds.fit_transform(self.distance_matrix)
        return self.embedding_2d

    def stress(self) -> float:
        """Compute MDS stress (normalized RMSE)."""
        if self.distance_matrix is None or self.embedding_2d is None:
            raise ValueError("Need both distance_matrix and embedding_2d")

        n = self.distance_matrix.shape[0]
        embedded_dist = np.zeros_like(self.distance_matrix)

        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(self.embedding_2d[i] - self.embedding_2d[j]))
                embedded_dist[i, j] = embedded_dist[j, i] = d

        numerator = np.sum((self.distance_matrix - embedded_dist) ** 2)
        denominator = np.sum(self.distance_matrix**2)

        return float(np.sqrt(numerator / denominator)) if denominator > 0 else 0.0
