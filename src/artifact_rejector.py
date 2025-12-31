from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from scipy import stats

# NumPy 2.0+ compatibility: trapz was removed and replaced with trapezoid
# Add backward compatibility for older NumPy versions
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # type: ignore


class ArtifactRejector:
    """Detects and rejects artifact-contaminated trials."""

    def __init__(
        self,
        voltage_threshold_uv: float = 100.0,
        kurtosis_threshold: float = 5.0,
        logger: Optional[logging.Logger] = None,
    ):
        self.voltage_threshold = voltage_threshold_uv
        self.kurtosis_threshold = kurtosis_threshold
        self.logger = logger

    def is_clean(self, trial_eeg: np.ndarray) -> bool:
        """Check if trial is artifact-free.

        Args:
            trial_eeg: (n_channels, n_samples)

        Returns:
            True if clean, False if contaminated
        """
        # Check voltage threshold
        max_voltage = np.max(np.abs(trial_eeg))
        if max_voltage > self.voltage_threshold:
            return False

        # Check kurtosis (spikes produce high kurtosis)
        kurtosis = stats.kurtosis(trial_eeg, axis=1, fisher=True)
        if np.any(np.abs(kurtosis) > self.kurtosis_threshold):
            return False

        return True

    def filter_trials(
        self, data: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filter out artifact trials.

        Args:
            data: (n_trials, n_channels, n_samples)
            labels: (n_trials, 4)

        Returns:
            clean_data, clean_labels, clean_mask
        """
        n_trials = data.shape[0]
        clean_mask = np.zeros(n_trials, dtype=bool)

        for trial_idx in range(n_trials):
            if self.is_clean(data[trial_idx]):
                clean_mask[trial_idx] = True

        n_rejected = n_trials - np.sum(clean_mask)
        if self.logger and n_rejected > 0:
            self.logger.info(
                f"  Artifact rejection: {n_rejected}/{n_trials} trials rejected "
                f"({100 * n_rejected / n_trials:.1f}%)"
            )

        return data[clean_mask], labels[clean_mask], clean_mask
