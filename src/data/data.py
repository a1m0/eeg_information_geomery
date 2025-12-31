import logging
import pickle
import re
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

from utils.config import DEAPConfig

if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # type: ignore


class DEAPLoader:
    """Loads and validates DEAP preprocessed python pickles."""

    def __init__(
        self, dataset_root: str | Path, config: DEAPConfig, logger: logging.Logger
    ):
        self.dataset_root = Path(dataset_root)
        self.config = config
        self.logger = logger
        self.data_dir = self._resolve_data_preprocessed_python_dir(self.dataset_root)

    @staticmethod
    def _resolve_data_preprocessed_python_dir(root: Path) -> Path:
        """Find data_preprocessed_python directory."""
        direct = root / "data_preprocessed_python"
        if direct.exists():
            return direct

        matches = list(root.rglob("data_preprocessed_python"))
        if matches:
            return matches[0]

        return root

    def list_subject_files(self) -> List[Path]:
        """List subject files (s01.dat ... s32.dat) in order."""
        files = sorted(self.data_dir.glob("s*.dat"))

        def key(p: Path) -> int:
            m = re.search(r"s(\d+)", p.stem)
            return int(m.group(1)) if m else 10**9

        return sorted(files, key=key)

    def load_subject(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray, str]:
        """Load single subject file with validation.

        Returns:
            data: (n_trials, n_channels, n_samples)
            labels: (n_trials, 4)
            subject_id: str
        """
        with open(file_path, "rb") as f:
            obj = pickle.load(f, encoding="iso-8859-1")

        if not isinstance(obj, dict) or "data" not in obj or "labels" not in obj:
            raise ValueError(f"Unexpected DEAP pickle structure in {file_path}")

        data = np.asarray(obj["data"], dtype=np.float32)
        labels = np.asarray(obj["labels"], dtype=np.float32)

        # Validate shapes
        expected_samples = self.config.baseline_samples + self.config.stimulus_samples
        if data.shape[2] < expected_samples:
            self.logger.warning(
                f"{file_path.name}: Expected {expected_samples} samples, got {data.shape[2]}"
            )

        if labels.shape[1] != 4:
            raise ValueError(
                f"{file_path.name}: Expected 4 label columns, got {labels.shape[1]}"
            )

        return data, labels, file_path.stem

    def load_all(
        self, max_subjects: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
        """Load all subjects."""
        files = self.list_subject_files()
        if not files:
            raise FileNotFoundError(
                f"No DEAP subject files found under: {self.data_dir}"
            )

        if max_subjects is not None:
            files = files[:max_subjects]

        all_data, all_labels, subject_ids = [], [], []

        for fp in files:
            self.logger.info(f"Loading {fp.name}")
            data, labels, subj_id = self.load_subject(fp)
            all_data.append(data)
            all_labels.append(labels)
            subject_ids.append(subj_id)

        return all_data, all_labels, subject_ids
