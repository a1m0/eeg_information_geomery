from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple


@dataclass(frozen=True)
class Bands:
    """EEG frequency bands.

    NOTE: Delta (0.5-4 Hz) REMOVED - DEAP is high-pass filtered at 4 Hz.
    Theta lower bound at 4 Hz may have filter edge artifacts.
    """

    theta: Tuple[float, float] = (4.0, 8.0)
    alpha: Tuple[float, float] = (8.0, 13.0)
    beta: Tuple[float, float] = (13.0, 30.0)

    def as_dict(self) -> Dict[str, Tuple[float, float]]:
        return {
            "theta": self.theta,
            "alpha": self.alpha,
            "beta": self.beta,
        }


@dataclass(frozen=True)
class DEAPConfig:
    """DEAP dataset configuration based on Koelstra et al. (2012)."""

    sfreq: int = 128
    n_eeg_channels: int = 32
    baseline_seconds: int = 3
    stimulus_seconds: int = 60
    highpass_hz: float = 4.0  # DEAP preprocessing
    lowpass_hz: float = 45.0

    # Expected channel names (first 32 channels)
    expected_channels: Tuple[str, ...] = (
        "Fp1",
        "AF3",
        "F3",
        "F7",
        "FC5",
        "FC1",
        "C3",
        "T7",
        "CP5",
        "CP1",
        "P3",
        "P7",
        "PO3",
        "O1",
        "Oz",
        "Pz",
        "Fp2",
        "AF4",
        "Fz",
        "F4",
        "F8",
        "FC6",
        "FC2",
        "Cz",
        "C4",
        "T8",
        "CP6",
        "CP2",
        "P4",
        "P8",
        "PO4",
        "O2",
    )

    @property
    def baseline_samples(self) -> int:
        return self.baseline_seconds * self.sfreq

    @property
    def stimulus_samples(self) -> int:
        return self.stimulus_seconds * self.sfreq


@dataclass
class AnalysisConfig:
    """Configuration for the analysis pipeline."""

    scheme: Literal["binary", "quadrant_va"] = "binary"
    target: Literal["valence", "arousal", "dominance", "liking"] = "valence"
    threshold: float = 5.0
    use_pca: bool = True
    pca_components: int = 15
    baseline_correct: bool = True
    artifact_threshold_uv: float = 100.0
    artifact_kurtosis_threshold: float = 5.0
    min_samples_per_state: int = 30
    global_seed: int = 42
    jsd_n_samples: int = 5000
    jsd_n_bootstrap: int = 50
    permutation_n_iter: int = 1000
