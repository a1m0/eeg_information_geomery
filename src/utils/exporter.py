from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from utils.config import AnalysisConfig
from representation.gaussian_state_distribution import GaussianStateDistribution

# import src.gaussian_distances import


# NumPy 2.0+ compatibility: trapz was removed and replaced with trapezoid
# Add backward compatibility for older NumPy versions
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # type: ignore


class Exporter:
    """Export results with metadata."""

    @staticmethod
    def export_distance_matrix(
        D: np.ndarray,
        labels: List[str],
        out_csv: Path,
        ci_low: Optional[np.ndarray] = None,
        ci_high: Optional[np.ndarray] = None,
    ) -> None:
        """Export distance matrix as CSV."""
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(D, index=labels, columns=labels)
        df.to_csv(out_csv)

        # Export confidence intervals if available
        if ci_low is not None and ci_high is not None:
            df_ci_low = pd.DataFrame(ci_low, index=labels, columns=labels)
            df_ci_high = pd.DataFrame(ci_high, index=labels, columns=labels)
            df_ci_low.to_csv(out_csv.parent / (out_csv.stem + "_ci_low.csv"))
            df_ci_high.to_csv(out_csv.parent / (out_csv.stem + "_ci_high.csv"))

    @staticmethod
    def export_embedding(coords: np.ndarray, labels: List[str], out_csv: Path) -> None:
        """Export 2D embedding coordinates."""
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"state": labels, "x": coords[:, 0], "y": coords[:, 1]})
        df.to_csv(out_csv, index=False)

    @staticmethod
    def export_distribution_params(
        dists: Dict[str, GaussianStateDistribution], out_dir: Path
    ) -> None:
        """Export Gaussian parameters."""
        out_dir.mkdir(parents=True, exist_ok=True)

        means = pd.DataFrame({k: v.mean for k, v in dists.items()}).T
        means.index.name = "state"
        means.to_csv(out_dir / "gaussian_means.csv")

        vars_ = pd.DataFrame({k: np.diag(v.cov) for k, v in dists.items()}).T
        vars_.index.name = "state"
        vars_.to_csv(out_dir / "gaussian_variances_diag.csv")

    @staticmethod
    def export_metadata(config: AnalysisConfig, out_path: Path, **kwargs) -> None:
        """Export analysis metadata as JSON."""
        import numpy as np

        def convert_to_serializable(obj):
            """Convert numpy types to Python native types."""
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(i) for i in obj]
            return obj

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "scheme": config.scheme,
                "target": config.target,
                "threshold": config.threshold,
                "use_pca": config.use_pca,
                "pca_components": config.pca_components,
                "baseline_correct": config.baseline_correct,
                "artifact_threshold_uv": config.artifact_threshold_uv,
                "global_seed": config.global_seed,
            },
            **convert_to_serializable(kwargs),
        }

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metadata, f, indent=2)
