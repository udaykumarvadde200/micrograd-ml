from micrograd.metrics import classification_metrics


def test_classification_metrics():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    metrics = classification_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 2 / 3
    assert metrics["recall"] == 1.0
    assert abs(metrics["f1"] - 0.8) < 1e-10

    assert metrics["confusion_matrix"].tolist() == [
        [1, 1],
        [0, 2]
    ]