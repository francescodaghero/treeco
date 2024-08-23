import os
from copy import deepcopy
import pytest
import joblib
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from treeco.utils import quantize
import onnx
from skl2onnx import to_onnx
import itertools

TMP_DIR = "./tmp"
TEST_MODELS = {
    "rf_classification": RandomForestClassifier,
    # "rf_regression": RandomForestRegressor,
    # "gbt_classification": GradientBoostingClassifier,
}
import json


def dump_to_onnx(model, X, y, model_name, metadata):
    model_proto = to_onnx(model, X[:1].astype(np.float32))
    # Serialize the data and store as metadata
    for k, v in metadata.items():
        arg = model_proto.metadata_props.add()
        arg.key = k
        arg.value = json.dumps(v)
        # metadata_props.append(onnx.helper.make_attribute(k, v))

    # model_proto.graph.initializer.extend([metadata])

    arg = model_proto.metadata_props.add()
    arg.key = "X"
    arg.value = json.dumps(X.tolist())

    # model_proto.graph.initializer.extend([metadata])

    arg = model_proto.metadata_props.add()
    arg.key = "y"
    arg.value = json.dumps(y.tolist())
    # model_proto.graph.initializer.extend([metadata_y])
    with open(os.path.join(TMP_DIR, model_name), "wb") as f:
        f.write(model_proto.SerializeToString())


def get_name(model_name, bits_input, depth, n_est, n_classes):
    return f"{model_name}_bitsInput{bits_input}_depth{depth}_nest{n_est}_classes{int(n_classes)}.onnx"


# Session-scoped fixture to generate and save models
# TODO : Can't make this work atm
# @pytest.fixture(scope="session", autouse=True)
def generate_and_save_models():
    if not os.path.exists(TMP_DIR):
        os.makedirs(TMP_DIR)

    tree_depths = [2]
    n_estimators_list = [2]
    n_classes_list = [3]
    bits_input = [None]

    classification_tests = itertools.product(
        bits_input, tree_depths, n_estimators_list, n_classes_list
    )
    regression_tests = itertools.product(
        bits_input, tree_depths, n_estimators_list, [4]
    )

    for model_name, model_class in TEST_MODELS.items():
        print(model_name)
        tests = (
            deepcopy(classification_tests)
            if "classification" in model_name
            else deepcopy(regression_tests)
        )
        for idx, (bits_input, depth, n_estimators, n_classes) in enumerate(tests):
            model_fname = get_name(
                model_name, bits_input, depth, n_estimators, n_classes
            )
            print(model_fname)
            if os.path.exists(os.path.join(TMP_DIR, model_fname)):
                continue

            X, y = make_classification(
                n_samples=100,
                n_features=20,
                n_informative=8,
                random_state=42,
                n_classes=n_classes,
            )
            # Test also negative values
            if "regression" in model_name:
                n_classes -= n_classes / 2

            model = model_class(
                max_depth=depth, n_estimators=n_estimators, random_state=42
            )
            if bits_input is not None:
                X, _, _ = quantize(
                    data=X, min_val=X.min(), max_val=X.max(), precision=bits_input
                )
            model.fit(X, y)

            dump_to_onnx(
                model,
                X,
                y,
                model_fname,
                metadata={
                    "model_name": model_name,
                    "bits_input": bits_input,
                    "depth": depth,
                    "n_estimators": n_estimators,
                    "n_classes": n_classes,
                },
            )

    # Optional: Clean up the directory at the end of the session
    # os.rmdir(TMP_DIR) # Remove the tmp directory if required


if __name__ == "__main__":
    generate_and_save_models()
