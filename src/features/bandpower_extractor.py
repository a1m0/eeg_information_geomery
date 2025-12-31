from __future__ import annotations

import logging
from typing import List, Optional, Tuple
import numpy as np
from scipy.signal import welch

from utils.config import Bands, DEAPConfig
from artifact_rejector import ArtifactRejector

# NumPy 2.0+ compatibility: trapz was removed and replaced with trapezoid
# Add backward compatibility for older NumPy versions
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # type: ignore


class BandpowerExtractor:
    """Extracts bandpower features with proper baseline correction."""

    def __init__(
        self,
        bands: Bands = Bands(),
        sfreq: int = 128,
        welch_nperseg: int = 256,
        eps: float = 1e-12,
        logger: Optional[logging.Logger] = None,
    ):
        self.bands = bands.as_dict()
        self.sfreq = sfreq
        self.welch_nperseg = welch_nperseg
        self.eps = eps
        self.logger = logger

    @staticmethod
    def _band_integral(
        f: np.ndarray, pxx: np.ndarray, band: Tuple[float, float]
    ) -> float:
        """Integrate PSD over frequency band using trapezoidal rule."""
        lo, hi = band
        mask = (f >= lo) & (f < hi)

        if not np.any(mask):
            raise ValueError(
                f"Band {band} Hz outside PSD frequency range [{f[0]:.2f}, {f[-1]:.2f}] Hz"
            )

        # Use np.trapezoid (NumPy 2.0+) with backward compatibility
        return float(np.trapezoid(pxx[mask], f[mask]))

    def extract_trial(
        self,
        trial_eeg: np.ndarray,
        baseline_samples: int,
        baseline_correct: bool = True,
    ) -> np.ndarray:
        """Extract bandpower features from single trial.

        Baseline correction: Computes log(stim_power / baseline_power) which is
        equivalent to log(stim) - log(baseline). This is a log-ratio, representing
        proportional change in dB-like units.

        Args:
            trial_eeg: (n_channels, n_samples)
            baseline_samples: number of baseline samples
            baseline_correct: whether to apply baseline correction

        Returns:
            features: (n_channels * n_bands,) ordered as [ch0_theta, ch0_alpha, ...]
        """
        if trial_eeg.ndim != 2:
            raise ValueError("trial_eeg must have shape (channels, samples)")

        n_ch, n_samp = trial_eeg.shape
        if n_samp <= baseline_samples:
            raise ValueError("Not enough samples for baseline/stimulus split")

        baseline = trial_eeg[:, :baseline_samples]
        stimulus = trial_eeg[:, baseline_samples:]

        feat_list: List[float] = []

        for ch in range(n_ch):
            # Welch PSD with consistent parameters
            f_b, pxx_b = welch(
                baseline[ch],
                fs=self.sfreq,
                nperseg=min(self.welch_nperseg, baseline.shape[1]),
                detrend="constant",
            )
            f_s, pxx_s = welch(
                stimulus[ch],
                fs=self.sfreq,
                nperseg=min(self.welch_nperseg, stimulus.shape[1]),
                detrend="constant",
            )

            for band_name, band in self.bands.items():
                pb = self._band_integral(f_b, pxx_b, band)
                ps = self._band_integral(f_s, pxx_s, band)

                if baseline_correct:
                    # Log-ratio: log(stimulus/baseline) in dB-like units
                    feat = np.log((ps + self.eps) / (pb + self.eps))
                else:
                    feat = np.log(ps + self.eps)

                feat_list.append(float(feat))

        return np.asarray(feat_list, dtype=np.float32)

    def extract_subject(
        self,
        subject_data: np.ndarray,
        subject_labels: np.ndarray,
        config: DEAPConfig,
        artifact_rejector: Optional[ArtifactRejector] = None,
        baseline_correct: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract features from all trials of a single subject.

        Returns:
            features: (n_clean_trials, n_features)
            labels: (n_clean_trials, 4)
            clean_mask: (n_trials,) boolean mask of clean trials
        """
        # Artifact rejection
        if artifact_rejector is not None:
            # Only use EEG channels for artifact detection
            eeg_data = subject_data[:, : config.n_eeg_channels, :]
            clean_data, clean_labels, clean_mask = artifact_rejector.filter_trials(
                eeg_data, subject_labels
            )
        else:
            clean_data = subject_data[:, : config.n_eeg_channels, :]
            clean_labels = subject_labels
            clean_mask = np.ones(len(subject_labels), dtype=bool)

        # Handle case where all trials are rejected
        if len(clean_data) == 0:
            if self.logger:
                self.logger.warning("  All trials rejected - returning empty arrays")
            # Return empty arrays with correct shape
            n_bands = len(self.bands)
            n_features = config.n_eeg_channels * n_bands
            return (
                np.empty((0, n_features), dtype=np.float32),
                np.empty((0, 4), dtype=np.float32),
                clean_mask,
            )

        # Extract features from clean trials
        features = []
        for trial_eeg in clean_data:
            feat = self.extract_trial(
                trial_eeg,
                baseline_samples=config.baseline_samples,
                baseline_correct=baseline_correct,
            )
            features.append(feat)

        return np.vstack(features), clean_labels, clean_mask
