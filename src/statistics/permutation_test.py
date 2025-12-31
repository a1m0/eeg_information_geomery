from __future__ import annotations
import logging
from typing import Dict, Optional
import numpy as np
from sklearn.decomposition import PCA

from data.state_builder import StateScheme, LabelTarget, StateBuilder
from features.gaussian_distances import GaussianDistances
from representation.gaussian_state_distribution import GaussianStateDistribution

# NumPy 2.0+ compatibility: trapz was removed and replaced with trapezoid
# Add backward compatibility for older NumPy versions
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # type: ignore


class PermutationTest:
    """Statistical significance testing via permutation without preprocessing leakage."""

    @staticmethod
    def test_state_separation(
        features: np.ndarray,
        labels: np.ndarray,
        scheme: StateScheme,
        target: LabelTarget,
        threshold: float,
        metric: str,
        n_iterations: int = 1000,
        rng_seed: int = 42,
        use_pca: bool = False,
        pca_components: int = 15,
        logger: Optional[logging.Logger] = None,
    ) -> Dict[str, float]:
        """Test if state separation is significant via permutation.

        CRITICAL FIX: If PCA is used, it is refitted for each permutation to avoid
        leakage from the true label structure.

        Args:
            features: (n_samples, n_features) - RAW features (not PCA-transformed)
            labels: (n_samples, 4)
            use_pca: Whether to apply PCA (will be fitted per permutation)
            pca_components: Number of PCA components

        Returns:
            Dictionary with observed distance and p-value.
        """
        rng = np.random.default_rng(rng_seed)

        # Helper function to compute distance with optional PCA
        def compute_distance_with_optional_pca(feats, labs, fit_pca=False):
            # Apply PCA if requested
            if use_pca:
                n_components = min(pca_components, feats.shape[1], feats.shape[0])
                if fit_pca:
                    pca = PCA(n_components=n_components, random_state=rng_seed)
                    feats_transformed = pca.fit_transform(feats)
                else:
                    # This shouldn't be called in corrected version
                    raise ValueError("Must fit PCA for each permutation")
            else:
                feats_transformed = feats

            # Build states
            states = StateBuilder.build_states(
                feats_transformed,
                labs,
                scheme=scheme,
                target=target,
                threshold=threshold,
                min_samples=1,
            )

            if len(states) != 2:
                return np.nan

            # Fit distributions
            state_names = list(states.keys())
            p_dist = GaussianStateDistribution(
                state_names[0], states[state_names[0]], test_normality=False
            )
            q_dist = GaussianStateDistribution(
                state_names[1], states[state_names[1]], test_normality=False
            )

            # Compute distance
            if metric == "skl":
                return GaussianDistances.symmetric_kl(p_dist, q_dist)
            elif metric == "hellinger":
                return GaussianDistances.hellinger(p_dist, q_dist)
            elif metric == "jsd":
                d, _, _ = GaussianDistances.jsd_monte_carlo(
                    p_dist, q_dist, rng=rng, n_samples=2000, n_bootstrap=1
                )
                return d
            else:
                raise ValueError(f"Unknown metric: {metric}")

        # Observed distance
        try:
            d_obs = compute_distance_with_optional_pca(features, labels, fit_pca=True)
            if np.isnan(d_obs):
                if logger:
                    logger.warning(
                        "Permutation test: Could not compute observed distance"
                    )
                return {"observed_distance": np.nan, "p_value": np.nan}
        except Exception as e:
            if logger:
                logger.warning(f"Permutation test observed distance failed: {e}")
            return {"observed_distance": np.nan, "p_value": np.nan}

        # Null distribution
        null_distances = []
        for iter_idx in range(n_iterations):
            try:
                # Shuffle labels
                labels_shuffled = labels.copy()
                rng.shuffle(labels_shuffled)

                # Compute distance with REFITTED PCA
                d_null = compute_distance_with_optional_pca(
                    features, labels_shuffled, fit_pca=True
                )

                if not np.isnan(d_null):
                    null_distances.append(d_null)

            except Exception:
                continue

        if len(null_distances) == 0:
            if logger:
                logger.warning("No valid permutations generated")
            return {
                "observed_distance": float(d_obs),
                "p_value": np.nan,
                "n_permutations": 0,
            }

        null_distances = np.array(null_distances)
        p_value = float(np.mean(null_distances >= d_obs))

        if logger:
            logger.info(
                f"Permutation test: observed={d_obs:.4f}, "
                f"null_mean={np.mean(null_distances):.4f}, "
                f"p={p_value:.4f} ({len(null_distances)}/{n_iterations} valid iterations)"
            )

        return {
            "observed_distance": float(d_obs),
            "p_value": float(p_value),
            "null_mean": float(np.mean(null_distances)),
            "null_std": float(np.std(null_distances)),
            "null_median": float(np.median(null_distances)),
            "n_permutations": len(null_distances),
        }
