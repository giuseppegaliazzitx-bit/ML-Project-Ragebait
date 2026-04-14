from __future__ import annotations

from pathlib import Path
from typing import Any

from ragebait_detector.evaluation import (
    compute_classification_metrics,
    plot_confusion_matrix,
    save_metrics_report,
)
from ragebait_detector.utils.io import ensure_parent


def run_baseline_suite(
    splits,
    output_dir: str | Path,
    max_features: int,
    ngram_range: tuple[int, int],
    seed: int,
) -> dict[str, Any]:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    destination = ensure_parent(Path(output_dir) / "baseline_results.json").parent
    text_views = {
        "raw": (
            [row["raw_text"] for row in splits.train],
            [row["raw_text"] for row in splits.validation],
            [row["raw_text"] for row in splits.test],
        ),
        "clean": (
            [row["clean_text"] for row in splits.train],
            [row["clean_text"] for row in splits.validation],
            [row["clean_text"] for row in splits.test],
        ),
    }
    y_train = [int(row["label"]) for row in splits.train]
    y_test = [int(row["label"]) for row in splits.test]

    results: dict[str, Any] = {}
    for view_name, (train_texts, _, test_texts) in text_views.items():
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=False,
        )
        X_train = vectorizer.fit_transform(train_texts)
        X_test = vectorizer.transform(test_texts)

        dense_limit = min(X_train.shape[0] - 1, X_train.shape[1] - 1, 300)
        if dense_limit >= 2:
            svd = TruncatedSVD(n_components=dense_limit, random_state=seed)
            X_train_dense = svd.fit_transform(X_train)
            X_test_dense = svd.transform(X_test)
        else:
            X_train_dense = X_train.toarray()
            X_test_dense = X_test.toarray()

        estimators = {
            "logistic_regression": (
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=seed,
                ),
                X_train,
                X_test,
            ),
            "svc": (
                SVC(
                    kernel="linear",
                    probability=True,
                    class_weight="balanced",
                    random_state=seed,
                ),
                X_train,
                X_test,
            ),
            "gaussian_nb": (
                GaussianNB(),
                X_train_dense,
                X_test_dense,
            ),
            "decision_tree": (
                DecisionTreeClassifier(
                    max_depth=20,
                    class_weight="balanced",
                    random_state=seed,
                ),
                X_train_dense,
                X_test_dense,
            ),
        }

        results[view_name] = {}
        for estimator_name, (estimator, train_matrix, test_matrix) in estimators.items():
            estimator.fit(train_matrix, y_train)
            predictions = estimator.predict(test_matrix)
            metrics = compute_classification_metrics(y_test, predictions.tolist())
            results[view_name][estimator_name] = metrics

            metrics_path = destination / f"{view_name}_{estimator_name}_metrics.json"
            matrix_path = destination / f"{view_name}_{estimator_name}_confusion_matrix.png"
            save_metrics_report(metrics, metrics_path)
            plot_confusion_matrix(
                y_test,
                predictions.tolist(),
                matrix_path,
                title=f"{view_name.title()} {estimator_name.replace('_', ' ').title()}",
            )

    save_metrics_report(results, destination / "baseline_results.json")
    return results
