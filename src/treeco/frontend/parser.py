from pathlib import Path
from typing import Mapping
import numpy as np
import onnx
import logging


class ParseWarning(Warning):
    def __init__(self, message: str):
        message = f"Warning: {message}"
        super().__init__(message)


class ParseError(Exception):
    def __init__(self, message: str):
        message = f"Parse error: {message}"
        super().__init__(message)


# https://onnx.ai/onnx/operators/onnx_aionnxml_TreeEnsemble.html#treeensemble-5-ai-onnx-ml
class Parser:
    """
    A parser class to read an onnx model and extract the TreeEnsemble op
    """

    file: Path
    nodes: list

    def __init__(self, file: Path):
        """
        Generates the parser object, sets the file.

        Parameters
        ----------
        file : Path
            File path to the onnx model
        """
        self.file = file
        self.nodes = list()

    def parseModel(self) -> Mapping:
        """
        Actual model parsing on self.file

        Returns
        -------
        Mapping
            Dictionary of field names and values in the onnx file.

        Raises
        ------
        ParseError

        """

        # Load the model
        with open(self.file, "rb") as f:
            model = onnx.load(f)
        for i, node in enumerate(model.graph.node):
            if node.op_type == "TreeEnsembleClassifier":
                return self._parseTreeEnsembleClassifier(node, graph = model.graph)
            elif node.op_type == "TreeEnsembleRegressor":
                return self._parseTreeEnsembleRegressor(node, graph = model.graph)
            elif node.op_type == "TreeEnsemble":
                logging.info(f"Found TreeEnsemble {node}")
                return self._parseTreeEnsemble(node, graph = model.graph)
        raise ParseError("Model does not contain a TreeEnsemble op")

    def _parseTreeEnsembleClassifier(self, node: onnx.NodeProto, graph : onnx.GraphProto) -> Mapping:
        """
        Private method to parse the TreeEnsembleClassifier node

        Parameters
        ----------
        node : onnx.NodeProto
            TreeEnsembleClassifier node

        Returns
        -------
        Mapping
            The node fields
        """
        attributes_dict: Mapping[str, onnx.AttributeProto] = {}
        for attr in node.attribute:
            attributes_dict[attr.name] = onnx.helper.get_attribute_value(attr)
        n_classes = len(set(attributes_dict["class_ids"]))

        if "nodes_missing_value_tracks_true" in attributes_dict:
            all(v == 0 for v in attributes_dict["nodes_missing_value_tracks_true"])

        # Additional control variables
        if "base_values" not in attributes_dict:
            attributes_dict["base_values"] = [
                0.0 for _ in range(n_classes)
            ]

        attributes_dict["input_shape"] = graph.input[0].type.tensor_type.shape.dim[1].dim_value
        attributes_dict["output_shape"] = n_classes
        return attributes_dict

    def _parseTreeEnsembleRegressor(self, node: onnx.NodeProto, graph : onnx.GraphProto) -> Mapping:
        """
        Private method to parse the TreeEnsembleRegressor node

        Parameters
        ----------
        node : onnx.NodeProto
            TreeEnsembleRegressor node

        Returns
        -------
        Mapping
            The node fields
        """
        attributes_dict: Mapping[str, onnx.AttributeProto] = {}
        for attr in node.attribute:
            attributes_dict[attr.name] = onnx.helper.get_attribute_value(attr)

        if "nodes_missing_value_tracks_true" in attributes_dict:
            all(v == 0 for v in attributes_dict["nodes_missing_value_tracks_true"])

        # Additional control variables
        if "base_values" not in attributes_dict:
            attributes_dict["base_values"] = [
                0.0 for _ in np.asarray(attributes_dict["target_ids"]).unique()
            ]

        attributes_dict["input_shape"] = graph.input[0].type.tensor_type.shape.dim[1].dim_value
        attributes_dict["output_shape"] = graph.output[1].type.tensor_type.shape.dim[1].dim_value
        return attributes_dict
