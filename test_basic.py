import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Import the library under test
from algorithmeclassifier import AlgorithmeClassifier


def _make_data(n_classes=2):
    X, y = make_classification(
        n_samples=240,
        n_features=10,
        n_informative=6,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=42,
    )
    return train_test_split(X, y, test_size=0.3, random_state=0)


def test_fit_predict_shapes_binary():
    X_train, X_test, y_train, y_test = _make_data(n_classes=2)
    clf = AlgorithmeClassifier()
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)
    assert proba.shape == (X_test.shape[0], 2)

    y_pred = clf.predict(X_test)
    assert y_pred.shape == (X_test.shape[0],)


def test_score_runs_all_metrics():
    X_train, X_test, y_train, y_test = _make_data(n_classes=2)
    clf = AlgorithmeClassifier()
    clf.fit(X_train, y_train)

    for metric in ["accuracy", "log_loss", "auc"]:
        score = clf.score(X_test, y_test, metric=metric)
        assert np.isfinite(score), f"{metric} returned non-finite value"
