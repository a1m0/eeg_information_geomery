from __future__ import annotations
import logging
from typing import Dict, List, Optional
import numpy as np
from sklearn.decomposition import PCA

from utils.config import AnalysisConfig
from data.subject_data import SubjectData
from data.state_builder import StateBuilder
from features.gaussian_distances import GaussianDistances
from representation.gaussian_state_distribution import GaussianStateDistribution

# NumPy 2.0+ compatibility: trapz was removed and replaced with trapezoid
# Add backward compatibility for older NumPy versions
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # type: ignore


class CrossSubjectAnalysis:
    """Leave-subject-out cross-validation with proper preprocessing isolation."""

    @staticmethod
    def run_loso_cv(
        subjects: List[SubjectData],
        config: AnalysisConfig,
        logger: Optional[logging.Logger] = None,
    ) -> Dict:
        """Run leave-one-subject-out cross-validation with per-fold PCA.

        CRITICAL FIX: PCA is fitted separately for each fold using only training data.
        Test subjects are evaluated by classifying their trials using training distributions.

        Returns:
            Dictionary with per-fold accuracies and distances
        """
        n_subjects = len(subjects)
        fold_results = []

        for test_idx in range(n_subjects):
            # 1. Split subjects
            train_subjects = [s for i, s in enumerate(subjects) if i != test_idx]
            test_subject = subjects[test_idx]

            if len(test_subject.features) == 0:
                if logger:
                    logger.warning(
                        f"Fold {test_idx}: Test subject {test_subject.subject_id} has no trials"
                    )
                continue

            # 2. Pool ONLY training subjects
            X_train = np.vstack([s.features for s in train_subjects])
            Y_train = np.vstack([s.labels for s in train_subjects])
            X_test = test_subject.features
            Y_test = test_subject.labels

            # 3. Fit PCA ONLY on training data (if enabled)
            if config.use_pca:
                n_components = min(
                    config.pca_components, X_train.shape[1], X_train.shape[0]
                )
                pca_fold = PCA(
                    n_components=n_components,
                    random_state=config.global_seed + test_idx,
                )
                X_train_transformed = pca_fold.fit_transform(X_train)
                X_test_transformed = pca_fold.transform(X_test)

                if logger:
                    explained_var = pca_fold.explained_variance_ratio_.sum()
                    logger.debug(
                        f"Fold {test_idx}: PCA {X_train.shape[1]} → {n_components} "
                        f"(var explained: {explained_var:.3f})"
                    )
            else:
                X_train_transformed = X_train
                X_test_transformed = X_test

            # 4. Build states on training data
            try:
                states_train = StateBuilder.build_states(
                    X_train_transformed,
                    Y_train,
                    scheme=config.scheme,
                    target=config.target,
                    threshold=config.threshold,
                    min_samples=config.min_samples_per_state,
                )

                if len(states_train) < 2:
                    if logger:
                        logger.warning(
                            f"Fold {test_idx}: Insufficient states "
                            f"({len(states_train)} < 2 required)"
                        )
                    continue

                # 5. Fit Gaussian distributions on training data
                dists_train = {
                    name: GaussianStateDistribution(name, data, test_normality=False)
                    for name, data in states_train.items()
                }

                state_names = list(dists_train.keys())

                # 6. Compute training distance (for reference)
                train_distance = np.nan
                if len(state_names) == 2:
                    p_train = dists_train[state_names[0]]
                    q_train = dists_train[state_names[1]]
                    train_distance = GaussianDistances.symmetric_kl(p_train, q_train)

                # 7. EVALUATE ON TEST SUBJECT'S TRIALS (NEW!)
                target_idx = StateBuilder.TARGET_INDEX[config.target]

                predictions = []
                true_labels = []
                log_likelihoods = []

                for trial_idx in range(len(X_test_transformed)):
                    trial_features = X_test_transformed[
                        trial_idx : trial_idx + 1
                    ]  # Keep 2D

                    # Get true label
                    true_value = Y_test[trial_idx, target_idx]
                    if config.scheme == "binary":
                        true_label = (
                            f"high_{config.target}"
                            if true_value >= config.threshold
                            else f"low_{config.target}"
                        )
                    else:
                        # For quadrant scheme, need valence and arousal
                        v = Y_test[trial_idx, StateBuilder.TARGET_INDEX["valence"]]
                        a = Y_test[trial_idx, StateBuilder.TARGET_INDEX["arousal"]]
                        hv = v >= config.threshold
                        ha = a >= config.threshold
                        if hv and ha:
                            true_label = "HVHA"
                        elif hv and not ha:
                            true_label = "HVLA"
                        elif not hv and ha:
                            true_label = "LVHA"
                        else:
                            true_label = "LVLA"

                    true_labels.append(true_label)

                    # Compute log-likelihood under each distribution
                    log_probs = {}
                    for state_name, dist in dists_train.items():
                        try:
                            log_prob = dist.log_pdf(trial_features)[0]
                            log_probs[state_name] = log_prob
                        except Exception:
                            log_probs[state_name] = -np.inf

                    # Predict: highest log-likelihood
                    if len(log_probs) > 0:
                        predicted_state = max(log_probs, key=log_probs.get)
                        predictions.append(predicted_state)
                        log_likelihoods.append(log_probs[predicted_state])
                    else:
                        predictions.append(state_names[0])  # Default
                        log_likelihoods.append(-np.inf)

                # Compute accuracy
                if len(predictions) > 0:
                    correct = sum(
                        1
                        for pred, true in zip(predictions, true_labels)
                        if pred == true
                    )
                    accuracy = correct / len(predictions)
                else:
                    accuracy = 0.0

                # Store fold results
                fold_results.append(
                    {
                        "test_subject": test_subject.subject_id,
                        "train_distance": float(train_distance),
                        "test_accuracy": float(accuracy),
                        "n_test_trials": len(predictions),
                        "n_train_samples": X_train.shape[0],
                        "n_states": len(state_names),
                        "state_names": state_names,
                        "mean_log_likelihood": float(np.mean(log_likelihoods))
                        if log_likelihoods
                        else np.nan,
                    }
                )

                if logger:
                    logger.info(
                        f"Fold {test_idx:2d} ({test_subject.subject_id}): "
                        f"accuracy={accuracy:.3f}, n_test={len(predictions)}, "
                        f"train_dist={train_distance:.4f}"
                    )

            except Exception as e:
                if logger:
                    logger.warning(f"Fold {test_idx} failed: {e}")
                continue

        # Aggregate results
        if len(fold_results) == 0:
            if logger:
                logger.warning("Cross-validation produced no valid folds")
            return {
                "fold_results": [],
                "mean_accuracy": np.nan,
                "std_accuracy": np.nan,
                "mean_train_distance": np.nan,
                "std_train_distance": np.nan,
                "n_folds": 0,
            }

        accuracies = [r["test_accuracy"] for r in fold_results]
        train_distances = [
            r["train_distance"]
            for r in fold_results
            if not np.isnan(r["train_distance"])
        ]

        results = {
            "fold_results": fold_results,
            "mean_accuracy": float(np.mean(accuracies)),
            "std_accuracy": float(np.std(accuracies)),
            "median_accuracy": float(np.median(accuracies)),
            "min_accuracy": float(np.min(accuracies)),
            "max_accuracy": float(np.max(accuracies)),
            "n_folds": len(fold_results),
        }

        if len(train_distances) > 0:
            results["mean_train_distance"] = float(np.mean(train_distances))
            results["std_train_distance"] = float(np.std(train_distances))
        else:
            results["mean_train_distance"] = np.nan
            results["std_train_distance"] = np.nan

        return results
