from enterprise_ai.ml.traditional import iris_classification_demo, iris_clustering_demo


def test_classification_accuracy_reasonable():
    r = iris_classification_demo()
    assert r.accuracy >= 0.85


def test_clustering_silhouette():
    r = iris_clustering_demo()
    assert r.silhouette > 0.4
