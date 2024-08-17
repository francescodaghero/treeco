""" 
Dialect implementing all types/attributes used in Crown and Trunk, plus some Cast ops.
"""

import numpy as np
from xdsl.utils.exceptions import DiagnosticException
from treeco import utils as utils
from xdsl.dialects.builtin import (
    ArrayAttr,
    TensorType,
    StringAttr,
    DenseIntOrFPElementsAttr,
    AnyFloat,
)
from xdsl.irdl import (
    irdl_op_definition,
    result_def,
    IRDLOperation,
    operand_def,
    Attribute,
)
from xdsl.dialects.builtin import (
    AnyFloat,
    IntegerType,
    TensorType,
    DenseIntOrFPElementsAttr,
    SSAValue,
    IntegerAttr,
    FloatAttr,
    Signedness,
    IndexType,
)
from xdsl.ir import ParametrizedAttribute, TypeAttribute
from xdsl.irdl import irdl_attr_definition, ParameterDef
from xdsl.ir import Dialect
from xdsl.ir.affine import AffineMap
from typing import TypeAlias

from xdsl.dialects.builtin import (
    AffineMapAttr,
    AnyFloat,
    IntegerType,
    DenseIntOrFPElementsAttr,
    IntegerAttr,
    FloatAttr,
    Signedness,
    IndexType,
)
from typing import TypeAlias

AnyNumericType: TypeAlias = IntegerType | IndexType | AnyFloat
AnyNumericAttr: TypeAlias = FloatAttr | IntegerAttr
BoolType: IntegerType = IntegerType(1, Signedness.SIGNLESS)


@irdl_attr_definition
class LeafType(ParametrizedAttribute, TypeAttribute):
    name = "treeco.leaf"
    target_type: ParameterDef[TensorType]
    target_map: ParameterDef[AffineMapAttr]

    def __init__(self, target_type: TensorType, target_map: ArrayAttr):
        super().__init__([target_type, target_map])

    def get_leaf_shape(self):
        return self.target_type.get_shape()


@irdl_attr_definition
class NodeType(ParametrizedAttribute, TypeAttribute):
    name = "treeco.node"
    weight_type: ParameterDef[AnyNumericType]

    def __init__(self, weight_type: AnyNumericType):
        super().__init__([weight_type])


@irdl_attr_definition
class TreeType(ParametrizedAttribute, TypeAttribute):
    name = "treeco.tree"
    n_features: ParameterDef[IntegerAttr]
    node_type: ParameterDef[NodeType]
    leaf_type: ParameterDef[LeafType]

    def __init__(self, n_features, node_type: NodeType, leaf_type: LeafType):
        super().__init__([n_features, node_type, leaf_type])


@irdl_attr_definition
class TreeEnsembleType(ParametrizedAttribute, TypeAttribute):
    name = "treeco.ensembleType"
    tree_type: ParameterDef[TreeType]
    n_features: ParameterDef[IntegerAttr]
    aggregate_mode: ParameterDef[StringAttr]
    post_transform: ParameterDef[StringAttr]

    def __init__(
        self,
        tree_type: TreeType,
        n_features: IntegerAttr,
        aggregate_mode: StringAttr,
        post_transform: StringAttr,
    ):
        super().__init__([tree_type, n_features, aggregate_mode, post_transform])


@irdl_attr_definition
class TreeEnsembleAttr(ParametrizedAttribute):
    name = "treeco.ensembleAttr"

    # Attributes
    ## Tree stats
    n_features: ParameterDef[IntegerAttr]
    # A new attribute - SUM or VOTE, default is SUM. Used to avoid VOTE -> Quantization,
    # and to drive lowering to the right path.
    # TODO: Can this be avoided? Can we infer this from the model?
    aggregate_mode: ParameterDef[StringAttr]
    post_transform: ParameterDef[StringAttr]

    ## Leaf-level parameters
    targets_ids: ParameterDef[DenseIntOrFPElementsAttr]
    targets_treeids: ParameterDef[DenseIntOrFPElementsAttr]
    targets_nodeids: ParameterDef[DenseIntOrFPElementsAttr]
    targets_weights: ParameterDef[DenseIntOrFPElementsAttr]

    ## Tree-level
    nodes_falsenodeids: ParameterDef[DenseIntOrFPElementsAttr]
    nodes_featureids: ParameterDef[DenseIntOrFPElementsAttr]

    ## Ensemble level
    nodes_modes: ParameterDef[ArrayAttr[StringAttr]]
    nodes_nodeids: ParameterDef[DenseIntOrFPElementsAttr]
    nodes_treeids: ParameterDef[DenseIntOrFPElementsAttr]
    nodes_truenodeids: ParameterDef[DenseIntOrFPElementsAttr]
    nodes_values: ParameterDef[DenseIntOrFPElementsAttr]

    nodes_hitrates: ParameterDef[DenseIntOrFPElementsAttr]

    def __init__(
        self,
        n_features: IntegerAttr,
        aggregate_mode: StringAttr,
        post_transform: StringAttr,
        targets_ids: DenseIntOrFPElementsAttr,
        targets_treeids: DenseIntOrFPElementsAttr,
        targets_nodeids: DenseIntOrFPElementsAttr,
        targets_weights: DenseIntOrFPElementsAttr,
        nodes_falsenodeids: DenseIntOrFPElementsAttr,
        nodes_featureids: DenseIntOrFPElementsAttr,
        nodes_modes: ArrayAttr,
        nodes_nodeids: DenseIntOrFPElementsAttr,
        nodes_treeids: DenseIntOrFPElementsAttr,
        nodes_truenodeids: DenseIntOrFPElementsAttr,
        nodes_values: DenseIntOrFPElementsAttr,
        nodes_hitrates: DenseIntOrFPElementsAttr,
        **ignored,
    ):
        super().__init__(
            [
                n_features,
                aggregate_mode,
                post_transform,
                targets_ids,
                targets_treeids,
                targets_nodeids,
                targets_weights,
                nodes_falsenodeids,
                nodes_featureids,
                nodes_modes,
                nodes_nodeids,
                nodes_treeids,
                nodes_truenodeids,
                nodes_values,
                nodes_hitrates,
            ]
        )

    def verify(self):
        # TODO : Implement all verifications.
        pass

    # TODO Check for better ways to export all fields
    def to_dict(self):
        return {
            "n_features": self.n_features,
            "aggregate_mode": self.aggregate_mode,
            "post_transform": self.post_transform,
            "targets_ids": self.targets_ids,
            "targets_treeids": self.targets_treeids,
            "targets_nodeids": self.targets_nodeids,
            "targets_weights": self.targets_weights,
            "nodes_falsenodeids": self.nodes_falsenodeids,
            "nodes_featureids": self.nodes_featureids,
            "nodes_modes": self.nodes_modes,
            "nodes_nodeids": self.nodes_nodeids,
            "nodes_treeids": self.nodes_treeids,
            "nodes_truenodeids": self.nodes_truenodeids,
            "nodes_values": self.nodes_values,
            "nodes_hitrates": self.nodes_hitrates,
        }

    # Getters to return the data with no attributes
    def get_leaf_shape(self) -> int:
        # Extract the data of one tree
        targets_ids = utils.convert_arrayattr_to_np(self.targets_ids)
        tree_ids = utils.convert_arrayattr_to_np(self.targets_treeids)
        targets_nodeids = utils.convert_arrayattr_to_np(self.targets_nodeids)
        tgt_node_tree = targets_nodeids[tree_ids == 0]
        _, counts = np.unique(tgt_node_tree, return_counts=True)
        # The count should be identical for each target, so we take element 0
        return int(max(counts))

    def get_n_trees(self) -> int:
        _, counts = np.unique(
            utils.convert_arrayattr_to_np(self.nodes_treeids), return_counts=True
        )
        return len(counts)

    def get_n_targets(self) -> int:
        targets_ids = utils.convert_arrayattr_to_np(self.targets_ids)
        _, counts = np.unique(targets_ids, return_counts=True)
        return len(counts)

    # From the attribute, get the types of each element
    def node_type(self) -> NodeType:
        return NodeType(weight_type=self.nodes_values.type.element_type)

    def leaf_type(self) -> LeafType:
        leaf_shape = self.get_leaf_shape()
        n_targets = self.get_n_targets()
        # Only OvO or OvA are supported atm
        target_element_type = self.targets_weights.type.element_type
        if self.aggregate_mode.data == "VOTE":
            inner_map = lambda tree_idx: (-1,)
            target_element_type = IndexType()
        elif leaf_shape != n_targets:
            inner_map = lambda tree_idx: (tree_idx % n_targets)
        else:
            idxs = tuple([*range(n_targets)])
            inner_map = lambda tree_idx: idxs

        mappa = AffineMapAttr(AffineMap.from_callable(inner_map))
        leaf_shape_and_type = TensorType(
            element_type=target_element_type, shape=[1, leaf_shape]
        )

        return LeafType(target_type=leaf_shape_and_type, target_map=mappa)

    def tree_type(self) -> TreeType:
        return TreeType(
            n_features=self.n_features,
            node_type=self.node_type(),
            leaf_type=self.leaf_type(),
        )

    def get_stopping_constant(self) -> int:
        return self.n_features.value.data

    def ensemble_type(self) -> TreeEnsembleType:
        return TreeEnsembleType(
            tree_type=self.tree_type(),
            n_features=self.n_features,
            post_transform=self.post_transform,
            aggregate_mode=self.aggregate_mode,
        )

    # TODO: Implement a nicer printer/parser.
    # def parse_parameters(self):
    # def print_parameters(self, printer: Printer) -> None:
    #    printer.print_string("<")
    #    printer.print_attr_dict(self.to_dict())
    #    printer.print_string(">")


@irdl_op_definition
class CastSignOp(IRDLOperation):
    name = "treeco.cast_sign"
    operand1 = operand_def()
    res = result_def()

    def __init__(self, operand1: SSAValue, res: Attribute):
        super().__init__(operands=[operand1], result_types=[res])


@irdl_op_definition
class Cast(IRDLOperation):
    name = "treeco.cast"
    operand1 = operand_def(NodeType | IndexType)
    res = result_def(IndexType | NodeType)

    def __init__(self, operand1: SSAValue, output_type):
        super().__init__(operands=[operand1], result_types=[output_type])

    def verify(self, verify_nested_ops: bool = True) -> None:
        if self.operand1.type == self.res.type:
            raise DiagnosticException("Cast operation must have different types")


Treeco = Dialect(
    "treeco",
    [Cast, CastSignOp],
    [
        LeafType,
        NodeType,
        TreeType,
        TreeEnsembleType,
        TreeEnsembleAttr,
    ],
)
