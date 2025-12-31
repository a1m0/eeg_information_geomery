from __future__ import annotations

from typing import Dict, Literal
import numpy as np


# NumPy 2.0+ compatibility: trapz was removed and replaced with trapezoid
# Add backward compatibility for older NumPy versions
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # type: ignore

LabelTarget = Literal["valence", "arousal", "dominance", "liking"]
StateScheme = Literal["binary", "quadrant_va"]


class StateBuilder:
    """Builds emotional state labels from DEAP ratings."""

    TARGET_INDEX = {"valence": 0, "arousal": 1, "dominance": 2, "liking": 3}

    @classmethod
    def build_states(
        cls,
        features: np.ndarray,
        labels: np.ndarray,
        scheme: StateScheme = "binary",
        target: LabelTarget = "valence",
        threshold: float = 5.0,
        min_samples: int = 30,
    ) -> Dict[str, np.ndarray]:
        """Build state sample sets from features and labels.

        Args:
            features: (n_samples, n_features)
            labels: (n_samples, 4)
            scheme: "binary" or "quadrant_va"
            target: label dimension for binary scheme
            threshold: split threshold (on 1-9 scale)
            min_samples: minimum samples required per state

        Returns:
            Dictionary mapping state names to feature arrays
        """
        if features.shape[0] != labels.shape[0]:
            raise ValueError("Features and labels must have same number of samples")

        if scheme == "binary":
            idx = cls.TARGET_INDEX[target]
            y = labels[:, idx]

            low = features[y < threshold]
            high = features[y >= threshold]

            states = {f"low_{target}": low, f"high_{target}": high}

        elif scheme == "quadrant_va":
            v = labels[:, cls.TARGET_INDEX["valence"]]
            a = labels[:, cls.TARGET_INDEX["arousal"]]

            hv = v >= threshold
            ha = a >= threshold

            states = {
                "LVLA": features[(~hv) & (~ha)],
                "LVHA": features[(~hv) & (ha)],
                "HVLA": features[(hv) & (~ha)],
                "HVHA": features[(hv) & (ha)],
            }
        else:
            raise ValueError(f"Unknown scheme: {scheme}")

        # Filter states with insufficient samples
        states = {k: v for k, v in states.items() if len(v) >= min_samples}

        if len(states) < 2:
            raise ValueError(
                f"Fewer than 2 states have >={min_samples} samples. "
                f"Adjust threshold or min_samples."
            )

        return states
