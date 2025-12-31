from __future__ import annotations


from dataclasses import dataclass

import numpy as np

# NumPy 2.0+ compatibility: trapz was removed and replaced with trapezoid
# Add backward compatibility for older NumPy versions
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # type: ignore


@dataclass
class SubjectData:
    """Container for single subject's data with proper tracking."""

    subject_id: str
    features: np.ndarray  # (n_trials, n_features)
    labels: np.ndarray  # (n_trials, 4)
    trial_indices: np.ndarray  # (n_trials,) original trial numbers

    def __len__(self) -> int:
        return len(self.features)

    def normalize_features(self) -> SubjectData:
        """Return copy with z-scored features (within-subject normalization)."""
        # Handle empty feature arrays
        if len(self.features) == 0:
            return SubjectData(
                subject_id=self.subject_id,
                features=self.features.copy(),
                labels=self.labels.copy(),
                trial_indices=self.trial_indices.copy(),
            )

        mean = self.features.mean(axis=0, keepdims=True)
        std = self.features.std(axis=0, keepdims=True)
        std = np.where(std < 1e-10, 1.0, std)  # avoid division by zero

        features_normalized = (self.features - mean) / std

        return SubjectData(
            subject_id=self.subject_id,
            features=features_normalized,
            labels=self.labels.copy(),
            trial_indices=self.trial_indices.copy(),
        )
