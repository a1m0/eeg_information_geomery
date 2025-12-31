import argparse

from utils.config import AnalysisConfig
from analysis import run_analysis


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser with artifact rejection option."""
    p = argparse.ArgumentParser(
        description="Information-geometric analysis on DEAP EEG (Leakage-Free Version)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--dataset_root",
        type=str,
        default="data/deap-dataset",
        help="Path containing data_preprocessed_python directory",
    )

    p.add_argument(
        "--output_dir",
        type=str,
        default="results_deap_revised",
        help="Output directory for results",
    )

    p.add_argument(
        "--scheme",
        type=str,
        choices=["binary", "quadrant_va"],
        default="binary",
        help="State labeling scheme",
    )

    p.add_argument(
        "--target",
        type=str,
        choices=["valence", "arousal", "dominance", "liking"],
        default="valence",
        help="Target dimension for binary scheme",
    )

    p.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Threshold for state separation (on 1-9 scale)",
    )

    p.add_argument(
        "--metrics",
        type=str,
        default="jsd,skl,hellinger",
        help="Comma-separated distance metrics",
    )

    p.add_argument(
        "--max_subjects",
        type=int,
        default=None,
        help="Limit number of subjects (for testing)",
    )

    # Artifact rejection options
    p.add_argument(
        "--skip_artifact_rejection",
        action="store_true",
        help="Skip artifact rejection (recommended for DEAP preprocessed data)",
    )

    p.add_argument(
        "--artifact_threshold",
        type=float,
        default=100.0,
        help="Voltage threshold in μV (ignored if --skip_artifact_rejection)",
    )

    p.add_argument(
        "--artifact_kurtosis_threshold",
        type=float,
        default=5.0,
        help="Kurtosis threshold for artifact detection (ignored if --skip_artifact_rejection)",
    )

    # Feature extraction
    p.add_argument(
        "--no_baseline_correct", action="store_true", help="Disable baseline correction"
    )

    p.add_argument(
        "--no_pca", action="store_true", help="Disable PCA dimensionality reduction"
    )

    p.add_argument(
        "--pca_components", type=int, default=15, help="Number of PCA components"
    )

    # Analysis options
    p.add_argument(
        "--min_samples_per_state",
        type=int,
        default=30,
        help="Minimum samples required per state",
    )

    p.add_argument(
        "--global_seed", type=int, default=42, help="Random seed for reproducibility"
    )

    p.add_argument(
        "--jsd_samples",
        type=int,
        default=5000,
        help="Number of Monte Carlo samples for JSD",
    )

    p.add_argument(
        "--jsd_bootstrap",
        type=int,
        default=50,
        help="Number of bootstrap iterations for JSD confidence intervals",
    )

    p.add_argument(
        "--permutation_iterations",
        type=int,
        default=1000,
        help="Number of permutation test iterations",
    )

    p.add_argument(
        "--no_permutation_test", action="store_true", help="Skip permutation test"
    )

    p.add_argument(
        "--no_cross_validation", action="store_true", help="Skip cross-validation"
    )

    p.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    return p


def main():
    """Main entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    # Parse metrics
    metrics = [m.strip() for m in args.metrics.split(",")]

    # Build config
    config = AnalysisConfig(
        scheme=args.scheme,
        target=args.target,
        threshold=args.threshold,
        use_pca=not args.no_pca,
        pca_components=args.pca_components,
        baseline_correct=not args.no_baseline_correct,
        artifact_threshold_uv=args.artifact_threshold,
        artifact_kurtosis_threshold=args.artifact_kurtosis_threshold,
        min_samples_per_state=args.min_samples_per_state,
        global_seed=args.global_seed,
        jsd_n_samples=args.jsd_samples,
        jsd_n_bootstrap=args.jsd_bootstrap,
        permutation_n_iter=args.permutation_iterations,
    )

    # Run analysis
    run_analysis(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        config=config,
        metrics=metrics,
        max_subjects=args.max_subjects,
        run_permutation_test=not args.no_permutation_test,
        run_cross_validation=not args.no_cross_validation,
        skip_artifact_rejection=args.skip_artifact_rejection,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
