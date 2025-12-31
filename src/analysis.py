from __future__ import annotations

import json
import scipy
import sklearn

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from utils.config import AnalysisConfig, DEAPConfig
from utils.logger import setup_logger
from utils.exporter import Exporter
from utils.cross_subject_analysis import CrossSubjectAnalysis
from data.data import DEAPLoader
from data.subject_data import SubjectData
from data.state_builder import StateBuilder
from statistics.permutation_test import PermutationTest
from representation.manifold_embedding import ManifoldEmbedding
from representation.gaussian_state_distribution import GaussianStateDistribution
from features.bandpower_extractor import BandpowerExtractor

from artifact_rejector import ArtifactRejector

# NumPy 2.0+ compatibility: trapz was removed and replaced with trapezoid
# Add backward compatibility for older NumPy versions
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # type: ignore


def run_analysis(
    dataset_root: str | Path,
    output_dir: str | Path = "results_deap_revised",
    config: Optional[AnalysisConfig] = None,
    metrics: Iterable[str] = ("jsd", "skl", "hellinger"),
    max_subjects: Optional[int] = None,
    run_permutation_test: bool = True,
    run_cross_validation: bool = True,
    skip_artifact_rejection: bool = False,  # NEW PARAMETER
    log_level: str = "INFO",
) -> None:
    """Run complete analysis pipeline with all safeguards.

    CRITICAL FIXES:
    1. Main analysis uses all data for exploration (clearly labeled)
    2. Cross-validation uses per-fold PCA (proper generalization)
    3. Permutation test refits PCA per permutation (no leakage)
    4. Optional artifact rejection (can be disabled for pre-cleaned data)

    Args:
        dataset_root: Path to DEAP dataset
        output_dir: Where to save results
        config: Analysis configuration
        metrics: Distance metrics to compute
        max_subjects: Limit number of subjects (for testing)
        run_permutation_test: Whether to run permutation test
        run_cross_validation: Whether to run cross-validation
        skip_artifact_rejection: If True, skip artifact rejection (use for DEAP)
        log_level: Logging level
    """
    # Setup
    logger = setup_logger(log_level)
    if config is None:
        config = AnalysisConfig()

    # Set global seed
    np.random.seed(config.global_seed)

    logger.info("=" * 60)
    logger.info("DEAP INFORMATION-GEOMETRIC ANALYSIS (LEAKAGE-FREE)")
    logger.info("=" * 60)
    logger.info(
        f"numpy={np.__version__}, scipy={scipy.__version__}, sklearn={sklearn.__version__}"
    )

    deap_config = DEAPConfig()

    # Load data
    logger.info("Loading DEAP dataset...")
    loader = DEAPLoader(dataset_root, deap_config, logger)
    subjects_data, subjects_labels, subject_ids = loader.load_all(
        max_subjects=max_subjects
    )
    logger.info(f"Loaded {len(subject_ids)} subjects: {subject_ids}")

    # Setup artifact rejection (optional)
    if skip_artifact_rejection:
        logger.info("⚠️  Artifact rejection DISABLED (using all trials)")
        artifact_rejector = None
    else:
        logger.info(
            f"Artifact rejection enabled: voltage={config.artifact_threshold_uv}μV, "
            f"kurtosis={config.artifact_kurtosis_threshold}"
        )
        artifact_rejector = ArtifactRejector(
            voltage_threshold_uv=config.artifact_threshold_uv,
            kurtosis_threshold=config.artifact_kurtosis_threshold,
            logger=logger,
        )

    # Extract features per subject
    extractor = BandpowerExtractor(sfreq=deap_config.sfreq, logger=logger)

    subjects: List[SubjectData] = []
    total_trials = 0
    rejected_trials = 0

    for subj_data, subj_labels, subj_id in zip(
        subjects_data, subjects_labels, subject_ids
    ):
        features, labels, clean_mask = extractor.extract_subject(
            subj_data,
            subj_labels,
            deap_config,
            artifact_rejector=artifact_rejector,
            baseline_correct=config.baseline_correct,
        )

        trial_indices = np.where(clean_mask)[0]
        n_rejected = len(subj_labels) - len(labels)
        total_trials += len(subj_labels)
        rejected_trials += n_rejected

        subjects.append(
            SubjectData(
                subject_id=subj_id,
                features=features,
                labels=labels,
                trial_indices=trial_indices,
            )
        )

    if artifact_rejector is not None:
        rejection_rate = 100 * rejected_trials / total_trials
        logger.info(
            f"Total artifact rejection: {rejected_trials}/{total_trials} trials "
            f"({rejection_rate:.1f}%)"
        )
        if rejection_rate > 80:
            logger.warning(
                f"⚠️  HIGH REJECTION RATE ({rejection_rate:.1f}%)! "
                f"Consider using --skip_artifact_rejection for DEAP preprocessed data"
            )
        elif rejection_rate > 50:
            logger.warning(
                f"⚠️  Moderate rejection rate ({rejection_rate:.1f}%). "
                f"Try increasing --artifact_threshold or --artifact_kurtosis_threshold"
            )

    total_samples = sum(len(s) for s in subjects)
    logger.info(f"Total samples across subjects: {total_samples}")

    if total_samples < 100:
        logger.warning(
            f"⚠️  LOW SAMPLE COUNT ({total_samples})! "
            f"Results may be unreliable. Need at least 200+ trials."
        )

    # Within-subject normalization (CORRECT: per-subject z-scoring)
    logger.info("Applying within-subject z-score normalization...")
    subjects_normalized = [s.normalize_features() for s in subjects]

    # Filter out subjects with no data
    subjects_normalized = [s for s in subjects_normalized if len(s.features) > 0]
    logger.info(
        f"Subjects with valid data: {len(subjects_normalized)}/{len(subject_ids)}"
    )

    if len(subjects_normalized) == 0:
        logger.error("❌ No subjects have valid data after preprocessing!")
        return

    # Setup output directory
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"subject_id": subject_ids}).to_csv(
        out_dir / "subjects_used.csv", index=False
    )

    # ========================================================================
    # STAGE 1: EXPLORATORY ANALYSIS (uses all data, for visualization only)
    # ========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 1: EXPLORATORY ANALYSIS (all data)")
    logger.info("NOTE: Results are for exploration/visualization only.")
    logger.info("      NOT for generalization claims.")
    logger.info("=" * 60)

    # Pool all data
    X_all = np.vstack([s.features for s in subjects_normalized])
    Y_all = np.vstack([s.labels for s in subjects_normalized])
    logger.info(f"Pooled feature matrix: {X_all.shape}")

    # Check if we have enough data
    if X_all.shape[0] < config.min_samples_per_state * 2:
        logger.warning(
            f"⚠️  Insufficient data ({X_all.shape[0]} samples) for reliable analysis. "
            f"Need at least {config.min_samples_per_state * 2} samples."
        )

    # Optional PCA (on all data, for exploration only)
    if config.use_pca:
        n_components = min(config.pca_components, X_all.shape[1], X_all.shape[0] - 1)
        pca_exploratory = PCA(
            n_components=n_components,
            random_state=config.global_seed,
        )
        Xr_exploratory = pca_exploratory.fit_transform(X_all)
        explained_var = pca_exploratory.explained_variance_ratio_.sum()
        logger.info(
            f"PCA (exploratory): {X_all.shape[1]} -> {Xr_exploratory.shape[1]} features "
            f"(explained variance: {explained_var:.3f})"
        )

        if explained_var < 0.7:
            logger.warning(
                f"⚠️  Low explained variance ({explained_var:.3f}). "
                f"Consider increasing --pca_components"
            )
    else:
        Xr_exploratory = X_all

    # Build states
    logger.info(
        f"Building states: scheme={config.scheme}, target={config.target}, "
        f"threshold={config.threshold}"
    )

    try:
        states_exploratory = StateBuilder.build_states(
            Xr_exploratory,
            Y_all,
            scheme=config.scheme,
            target=config.target,
            threshold=config.threshold,
            min_samples=config.min_samples_per_state,
        )
    except ValueError as e:
        logger.error(f"❌ Failed to build states: {e}")
        logger.error(
            "Try adjusting --threshold or --min_samples_per_state, "
            "or use --skip_artifact_rejection to retain more data"
        )
        return

    logger.info("State sample sizes (exploratory):")
    for name, data in states_exploratory.items():
        logger.info(f"  {name}: n={len(data)}")

    # Check class balance
    state_sizes = [len(data) for data in states_exploratory.values()]
    min_size = min(state_sizes)
    max_size = max(state_sizes)
    imbalance_ratio = max_size / min_size if min_size > 0 else np.inf

    if imbalance_ratio > 3.0:
        logger.warning(
            f"⚠️  Severe class imbalance (ratio={imbalance_ratio:.1f}). "
            f"Consider adjusting --threshold to balance classes better."
        )

    # Fit distributions
    logger.info("Fitting Gaussian distributions...")
    dists_exploratory = {
        name: GaussianStateDistribution(name, data, test_normality=True)
        for name, data in states_exploratory.items()
    }

    # Export distribution parameters
    Exporter.export_distribution_params(
        dists_exploratory, out_dir / "distributions_exploratory"
    )

    # Compute distances and embeddings for each metric
    for metric in metrics:
        metric = metric.lower().strip()
        logger.info(f"\n--- Computing {metric.upper()} distances (exploratory) ---")

        emb = ManifoldEmbedding(
            dists_exploratory,
            metric=metric,  # type: ignore
            rng_seed=config.global_seed,
            jsd_n_samples=config.jsd_n_samples,
            jsd_n_bootstrap=config.jsd_n_bootstrap,
        )

        D = emb.compute_distance_matrix()
        coords = emb.embed_2d(random_state=config.global_seed)
        stress = emb.stress()

        logger.info(f"{metric.upper()} MDS stress: {stress:.4f}")

        # Log distance matrix for binary case
        if config.scheme == "binary" and len(emb.labels) == 2:
            dist_value = D[0, 1]
            logger.info(
                f"{metric.upper()} distance between {emb.labels[0]} and {emb.labels[1]}: "
                f"{dist_value:.4f}"
            )

        # Export with "exploratory" label
        Exporter.export_distance_matrix(
            D,
            emb.labels,
            out_dir / f"{metric}_distance_matrix_exploratory.csv",
            ci_low=emb.distance_ci_low,
            ci_high=emb.distance_ci_high,
        )
        Exporter.export_embedding(
            coords, emb.labels, out_dir / f"{metric}_mds_2d_exploratory.csv"
        )

    # ========================================================================
    # STAGE 2: CROSS-VALIDATION (proper generalization assessment)
    # ========================================================================
    if run_cross_validation:
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 2: LEAVE-SUBJECT-OUT CROSS-VALIDATION")
        logger.info("PCA fitted separately per fold (no leakage).")
        logger.info("=" * 60)

        # Filter subjects with no data (already done above)
        subjects_for_cv = subjects_normalized

        if len(subjects_for_cv) < 3:
            logger.warning(
                f"⚠️  Only {len(subjects_for_cv)} subjects have data. "
                f"Skipping cross-validation (requires at least 3 subjects)."
            )
        else:
            cv_result = CrossSubjectAnalysis.run_loso_cv(
                subjects_for_cv, config, logger=logger
            )

            if cv_result["n_folds"] > 0:
                logger.info("\n" + "=" * 60)
                logger.info("CROSS-VALIDATION RESULTS (TRUE GENERALIZATION)")
                logger.info("=" * 60)
                logger.info(
                    f"Mean test accuracy:  {cv_result['mean_accuracy']:.3f} "
                    f"± {cv_result['std_accuracy']:.3f}"
                )
                logger.info(f"Median test accuracy: {cv_result['median_accuracy']:.3f}")
                logger.info(
                    f"Range: [{cv_result['min_accuracy']:.3f}, {cv_result['max_accuracy']:.3f}]"
                )
                logger.info(
                    f"Completed folds: {cv_result['n_folds']}/{len(subjects_for_cv)}"
                )

                # Check for tiny test sets
                fold_results = cv_result["fold_results"]
                test_sizes = [r["n_test_trials"] for r in fold_results]
                n_tiny_folds = sum(1 for size in test_sizes if size <= 2)

                if n_tiny_folds > len(fold_results) / 2:
                    logger.warning(
                        f"⚠️  {n_tiny_folds}/{len(fold_results)} folds have ≤2 test trials. "
                        f"CV accuracy may be unreliable (high variance from tiny test sets)."
                    )
                    logger.warning(
                        "Consider reporting permutation test p-value as primary metric."
                    )

                if not np.isnan(cv_result["mean_train_distance"]):
                    logger.info(
                        f"Mean train distance: {cv_result['mean_train_distance']:.4f} "
                        f"± {cv_result['std_train_distance']:.4f}"
                    )

                # Interpretation guidance
                if config.scheme == "binary":
                    baseline = 0.5
                    if cv_result["mean_accuracy"] < baseline + 0.05:
                        logger.info(
                            f"⚠️  Accuracy ({cv_result['mean_accuracy']:.3f}) is at chance level "
                            f"(50% for binary). No evidence of generalization."
                        )
                    elif cv_result["mean_accuracy"] < baseline + 0.15:
                        logger.info(
                            f"ℹ️  Accuracy ({cv_result['mean_accuracy']:.3f}) is slightly above "
                            f"chance but with high variance (±{cv_result['std_accuracy']:.3f}). "
                            f"Weak generalization."
                        )
                    else:
                        logger.info(
                            f"✓ Accuracy ({cv_result['mean_accuracy']:.3f}) shows meaningful "
                            f"above-chance performance."
                        )

                # Save CV results
                cv_df = pd.DataFrame(cv_result["fold_results"])
                cv_df.to_csv(out_dir / "cross_validation_results.csv", index=False)

                # Save summary
                cv_summary = {k: v for k, v in cv_result.items() if k != "fold_results"}
                with open(out_dir / "cross_validation_summary.json", "w") as f:
                    json.dump(cv_summary, f, indent=2)
            else:
                logger.warning("❌ Cross-validation produced no valid folds")

    # ========================================================================
    # STAGE 3: PERMUTATION TEST (with proper PCA handling)
    # ========================================================================
    if run_permutation_test and config.scheme == "binary":
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 3: PERMUTATION TEST")
        logger.info("PCA refitted per permutation (no leakage).")
        logger.info("=" * 60)

        perm_result = PermutationTest.test_state_separation(
            X_all,  # Use RAW features
            Y_all,
            scheme=config.scheme,
            target=config.target,
            threshold=config.threshold,
            metric="skl",  # Use deterministic metric for speed
            n_iterations=config.permutation_n_iter,
            rng_seed=config.global_seed,
            use_pca=config.use_pca,
            pca_components=config.pca_components,
            logger=logger,
        )

        # Interpretation
        if not np.isnan(perm_result.get("p_value", np.nan)):
            p_val = perm_result["p_value"]
            obs_dist = perm_result["observed_distance"]
            null_mean = perm_result.get("null_mean", np.nan)

            logger.info("\n--- Permutation Test Interpretation ---")
            if p_val < 0.001:
                logger.info(f"✓✓✓ HIGHLY SIGNIFICANT (p={p_val:.4f} < 0.001)")
                logger.info("Strong evidence of state separation.")
            elif p_val < 0.01:
                logger.info(f"✓✓ VERY SIGNIFICANT (p={p_val:.4f} < 0.01)")
                logger.info("Clear evidence of state separation.")
            elif p_val < 0.05:
                logger.info(f"✓ SIGNIFICANT (p={p_val:.4f} < 0.05)")
                logger.info("Statistically significant state separation.")
            elif p_val < 0.10:
                logger.info(f"~ MARGINAL (p={p_val:.4f} < 0.10)")
                logger.info("Weak evidence, interpret with caution.")
            else:
                logger.info(f"✗ NOT SIGNIFICANT (p={p_val:.4f} >= 0.10)")
                logger.info("No evidence of state separation above chance.")

            logger.info(
                f"Observed distance: {obs_dist:.4f}, Null mean: {null_mean:.4f}"
            )

        # Save results
        with open(out_dir / "permutation_test_result.json", "w") as f:
            json.dump(perm_result, f, indent=2)

    # Export metadata
    metadata_extras = {
        "subjects": subject_ids,
        "n_subjects": len(subject_ids),
        "n_subjects_with_data": len(subjects_normalized),
        "n_total_samples": len(X_all),
        "feature_dim_original": X_all.shape[1],
        "feature_dim_final": Xr_exploratory.shape[1],
        "states_exploratory": {
            name: len(data) for name, data in states_exploratory.items()
        },
        "artifact_rejection_enabled": not skip_artifact_rejection,
        "artifact_rejection_rate": (
            float(rejected_trials / total_trials) if artifact_rejector else 0.0
        ),
    }

    Exporter.export_metadata(
        config,
        out_dir / "analysis_metadata.json",
        **metadata_extras,
    )

    logger.info("\n" + "=" * 60)
    logger.info(f"Results saved to: {out_dir}")
    logger.info("=" * 60)
    logger.info("Analysis complete!")
    logger.info("\nIMPORTANT NOTES:")
    logger.info("  • Exploratory results: For visualization/hypothesis generation")
    logger.info("  • CV accuracy: TRUE generalization performance estimate")
    logger.info("  • Permutation test: Statistical significance of separation")

    # Final warnings
    if total_samples < 200:
        logger.warning(
            f"\n⚠️  WARNING: Low sample count ({total_samples}). "
            f"Results may be statistically underpowered."
        )

    if run_cross_validation and cv_result.get("n_folds", 0) > 0:
        fold_results = cv_result["fold_results"]
        test_sizes = [r["n_test_trials"] for r in fold_results]
        if np.median(test_sizes) <= 2:
            logger.warning(
                "\n⚠️  WARNING: Most CV folds have ≤2 test trials. "
                "Accuracy estimates are unreliable. "
                "Rely on permutation test p-value instead."
            )

    logger.info("=" * 60)
