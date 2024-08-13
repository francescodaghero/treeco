"""
Onnxml dialect, almost a 1:1 mirror of the ONNX-ML specification. 
Every field of the nodes is replicated in MLIR.
Used just a shared entry point for TreeCo.
"""

from treeco import utils as utils
import numpy as np
from xdsl.dialects.builtin import (
    ArrayAttr,
    FloatAttr,
    IntegerAttr,
    SSAValue,
    StringAttr,
    MemRefType,
)

from xdsl.irdl import (
    IRDLOperation,
    attr_def,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    result_def,
)
from xdsl.ir import Dialect


@irdl_op_definition
class TreeEnsembleClassifier(IRDLOperation):
    name = "onnxml.treeclassifier"

    buffer_in = operand_def(MemRefType)
    buffer_out = operand_def(MemRefType)

    # TreeEnsembleClassifier attributes
    # TODO : These fields should become tensors
    classlabels_int64s = opt_attr_def(ArrayAttr[IntegerAttr])
    classlabels_strings = opt_attr_def(ArrayAttr[StringAttr])
    class_ids = attr_def(ArrayAttr[IntegerAttr])
    class_treeids = attr_def(ArrayAttr[IntegerAttr])
    class_weights = attr_def(ArrayAttr[FloatAttr])
    class_nodeids = attr_def(ArrayAttr[IntegerAttr])

    # Tree attributes
    nodes_falsenodeids = attr_def(ArrayAttr[IntegerAttr])
    nodes_featureids = attr_def(ArrayAttr[IntegerAttr])
    nodes_hitrates = attr_def(ArrayAttr[FloatAttr])
    nodes_missing_value_tracks_true = attr_def(ArrayAttr[IntegerAttr])
    nodes_modes = attr_def(ArrayAttr[StringAttr])
    nodes_nodeids = attr_def(ArrayAttr[IntegerAttr])
    nodes_treeids = attr_def(ArrayAttr[IntegerAttr])
    nodes_truenodeids = attr_def(ArrayAttr[IntegerAttr])
    nodes_values = attr_def(ArrayAttr[FloatAttr])
    post_transform = attr_def(StringAttr)

    # TODO: Handle this, for now, we are ignoring it
    # The initial memref/tensor should be set to this value.
    base_values = opt_attr_def(ArrayAttr[FloatAttr])

    def __init__(
        self,
        buffer_in: SSAValue,
        buffer_out: SSAValue,
        class_ids: ArrayAttr[IntegerAttr],
        class_treeids: ArrayAttr[IntegerAttr],
        class_weights: ArrayAttr[FloatAttr],
        class_nodeids: ArrayAttr[IntegerAttr],
        nodes_falsenodeids: ArrayAttr[IntegerAttr],
        nodes_featureids: ArrayAttr[IntegerAttr],
        nodes_hitrates: ArrayAttr[FloatAttr],
        nodes_missing_value_tracks_true: ArrayAttr[IntegerAttr],
        nodes_modes: ArrayAttr[StringAttr],
        nodes_nodeids: ArrayAttr[IntegerAttr],
        nodes_treeids: ArrayAttr[IntegerAttr],
        nodes_truenodeids: ArrayAttr[IntegerAttr],
        nodes_values: ArrayAttr[FloatAttr],
        post_transform: StringAttr,
        base_values: ArrayAttr[FloatAttr],
        classlabels_int64s: ArrayAttr[IntegerAttr] = None,
        classlabels_strings: ArrayAttr[StringAttr] = None,
    ):
        super().__init__(
            attributes={
                "class_ids": class_ids,
                "class_treeids": class_treeids,
                "class_weights": class_weights,
                "class_nodeids": class_nodeids,
                "nodes_falsenodeids": nodes_falsenodeids,
                "nodes_featureids": nodes_featureids,
                "nodes_hitrates": nodes_hitrates,
                "nodes_missing_value_tracks_true": nodes_missing_value_tracks_true,
                "nodes_modes": nodes_modes,
                "nodes_nodeids": nodes_nodeids,
                "nodes_treeids": nodes_treeids,
                "nodes_truenodeids": nodes_truenodeids,
                "nodes_values": nodes_values,
                "post_transform": post_transform,
                "base_values": base_values,
                "classlabels_int64s": classlabels_int64s,
                "classlabels_strings": classlabels_strings,
            },
            operands=[buffer_in, buffer_out],
        )

    def verify_(self) -> None:
        # TODO: Add verification for the onnx file
        # Nothing for now, as the onnx file should be verified before
        return

    def get_n_trees(self) -> int:
        tree_ids = utils.convert_arrayattr_to_np(self.nodes_treeids)
        return int(np.unique(tree_ids).shape[0])


@irdl_op_definition
class TreeEnsembleRegressor(IRDLOperation):
    name = "onnxml.treeregressor"
    buffer_in = operand_def(MemRefType)
    buffer_out = result_def(MemRefType)

    # TreeEnsembleRegressor attributes
    base_values = opt_attr_def(ArrayAttr[FloatAttr])
    n_targets = attr_def(IntegerAttr)
    nodes_falsenodeids = attr_def(ArrayAttr[IntegerAttr])
    nodes_featureids = attr_def(ArrayAttr[IntegerAttr])
    nodes_hitrates = attr_def(ArrayAttr[FloatAttr])
    nodes_modes = attr_def(ArrayAttr[StringAttr])
    nodes_nodeids = attr_def(ArrayAttr[IntegerAttr])
    node_treeids = attr_def(ArrayAttr[IntegerAttr])
    nodes_truenodeids = attr_def(ArrayAttr[IntegerAttr])
    nodes_values = attr_def(ArrayAttr[FloatAttr])
    post_transform = attr_def(StringAttr)

    target_ids = attr_def(ArrayAttr[IntegerAttr])
    target_nodeids = attr_def(ArrayAttr[IntegerAttr])
    target_treeids = attr_def(ArrayAttr[IntegerAttr])
    target_weights = attr_def(ArrayAttr[FloatAttr])

    def __init__(
        self,
        buffer_in: SSAValue,
        buffer_out: SSAValue,
        class_ids: ArrayAttr[IntegerAttr],
        class_treeids: ArrayAttr[IntegerAttr],
        class_weights: ArrayAttr[FloatAttr],
        nodes_falsenodeids: ArrayAttr[IntegerAttr],
        nodes_featureids: ArrayAttr[IntegerAttr],
        nodes_hitrates: ArrayAttr[FloatAttr],
        nodes_missing_value_tracks_true: ArrayAttr[IntegerAttr],
        nodes_modes: ArrayAttr[StringAttr],
        nodes_nodeids: ArrayAttr[IntegerAttr],
        nodes_treeids: ArrayAttr[IntegerAttr],
        nodes_truenodeids: ArrayAttr[IntegerAttr],
        nodes_values: ArrayAttr[FloatAttr],
        post_transform: StringAttr,
        classlabels_int64s: ArrayAttr[IntegerAttr] = None,
        classlabels_strings: ArrayAttr[StringAttr] = None,
    ):
        super().__init__(
            attributes={
                "class_ids": class_ids,
                "class_treeids": class_treeids,
                "class_weights": class_weights,
                "nodes_falsenodeids": nodes_falsenodeids,
                "nodes_featureids": nodes_featureids,
                "nodes_hitrates": nodes_hitrates,
                "nodes_missing_value_tracks_true": nodes_missing_value_tracks_true,
                "nodes_modes": nodes_modes,
                "nodes_nodeids": nodes_nodeids,
                "nodes_treeids": nodes_treeids,
                "nodes_truenodeids": nodes_truenodeids,
                "nodes_values": nodes_values,
                "post_transform": post_transform,
                "classlabels_int64s": classlabels_int64s,
                "classlabels_strings": classlabels_strings,
            },
            operands=[buffer_in, buffer_out],
        )

    def verify(self) -> None:
        # TODO: Add verification for the onnx file
        # Nothing for now, as the onnx file should be verified before
        return

    def get_n_trees(self) -> int:
        tree_ids = utils.convert_arrayattr_to_np(self.nodes_treeids)
        return int(np.unique(tree_ids).shape[0])


Onnxml = Dialect(
    "onnxml",
    [
        TreeEnsembleClassifier,
        TreeEnsembleRegressor,
    ],
    [],
)
