"""Classical ML demos: classification + clustering (Iris)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ClassificationReport:
    accuracy: float
    feature_names: list[str]
    coefficients: list[float]


def iris_classification_demo(random_state: int = 42) -> ClassificationReport:
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.25, random_state=random_state, stratify=iris.target
    )
    clf = Pipeline(
        [
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(max_iter=200)),
        ]
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, pred))
    lr: LogisticRegression = clf.named_steps["lr"]
    coef = lr.coef_[0].tolist()
    return ClassificationReport(
        accuracy=acc,
        feature_names=list(iris.feature_names),
        coefficients=coef,
    )


@dataclass(frozen=True)
class ClusteringReport:
    silhouette: float
    cluster_sizes: list[int]


def iris_clustering_demo(n_clusters: int = 3, random_state: int = 42) -> ClusteringReport:
    iris = load_iris()
    X = StandardScaler().fit_transform(iris.data)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = km.fit_predict(X)
    sil = float(silhouette_score(X, labels))
    _, counts = np.unique(labels, return_counts=True)
    return ClusteringReport(silhouette=sil, cluster_sizes=counts.tolist())


def features_to_frame() -> pd.DataFrame:
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=list(iris.feature_names))
    df["target"] = iris.target
    return df
