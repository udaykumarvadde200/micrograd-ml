from sklearn.datasets import load_breast_cancer


def load_dataset():
    data = load_breast_cancer()

    X = data.data
    y = data.target

    return X, y