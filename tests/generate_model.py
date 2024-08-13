from skl2onnx import convert_sklearn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os
from matplotlib import pyplot as plt
from sklearn.tree import plot_tree
from skl2onnx import to_onnx
from sklearn.datasets import make_classification
from sklearn.tree import plot_tree


def plot_ensemble(model):
    # Also store the test inputs
    if hasattr(model, "estimators_"):
        for idx, t in enumerate(model.estimators_):
            plot_tree(t, proportion=True)
            plt.savefig(os.path.join(".", f"tree_{idx}.png"), dpi=600)
            plt.clf()
    else:
        plot_tree(model)
        plt.savefig(os.path.join(".", f"tree_0.png"))
        plt.clf()


def generate_classification_rf(n_estimators=2, max_depth=1, n_classes=3):
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_classes=n_classes,
        random_state=0,
        n_informative=4,
    )
    clr = RandomForestClassifier(n_estimators=2, max_depth=3, random_state=42)
    clr.fit(X, y)
    plot_ensemble(clr)
    onx = to_onnx(clr, X[:1])
    with open("rf.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    return X, clr


generate_classification_rf()
