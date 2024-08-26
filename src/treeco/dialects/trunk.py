"""
Lower-level IR to represent the ensemble.
Here the focus is on the tree visit algorithm (iterative, matmul...)
with tailored optimizations.
"""
from xdsl.traits import (
    HasParent,
    IsTerminator,
    Pure,
)


from xdsl.irdl import (
    irdl_op_definition,
    opt_operand_def,
    VarOperand,
    OpResult,
    attr_def,
    result_def,
    IRDLOperation,
    operand_def,
    var_operand_def,
    AnyAttr,
)
from xdsl.ir import Dialect, Operation, SSAValue
from xdsl.dialects.builtin import (
    IntegerType,
    TensorType,
    SSAValue,
    IntegerAttr,
    FloatAttr,
    Signedness,
    IndexType,
    MemRefType,
)
from xdsl.dialects.builtin import (
    StringAttr,
    FloatAttr,
    IntegerAttr,
    MemRefType,
    SSAValue,
    IndexType,
)
from xdsl.irdl import prop_def
from xdsl.ir import Dialect
from treeco import utils as utils

from xdsl.dialects import scf
from treeco.dialects.treeco import (
    LeafType,
    TreeType,
    NodeType,
    TreeEnsembleType,
    TreeEnsembleAttr,
)


BoolType: IntegerType = IntegerType(1, Signedness.SIGNLESS)
AnyNumericAttr = FloatAttr | IntegerAttr


@irdl_op_definition
class AggregateLeafOp(IRDLOperation):
    name = "trunk.aggregate_leaf"
    ensemble = operand_def(TreeEnsembleType)
    tree = operand_def(TreeType)
    # The two tensors to aggregate
    leaf = operand_def(TensorType)
    tensor_out = operand_def(TensorType)
    res = result_def(TensorType)

    def __init__(
        self,
        ensemble: SSAValue,
        tree: SSAValue,
        leaf: SSAValue,
        tensor_out: SSAValue,
    ):
        super().__init__(
            operands=[
                ensemble,
                tree,
                leaf,
                tensor_out,
            ],
            result_types=[tensor_out.type],
        )

@irdl_op_definition
class GetTreeDepthOp(IRDLOperation):
    """ 
    Get the depth of the tree. Useful for perfect iteration
    when the depths are all different
    #TODO This has still no lowering
    """
    name = "trunk.get_tree_depth"
    ensemble = operand_def(TreeEnsembleType)
    tree = operand_def(TreeType)

    def __init__(
        self,
        ensemble: SSAValue,
        tree: SSAValue,
    ):
        raise NotImplementedError("This operation has no lowering yet.")
        super().__init__(
            operands=[ensemble, tree],
            result_types=[IndexType()],
        )

@irdl_op_definition
class GetLeafValueOp(IRDLOperation):
    """
    Starting from a leaf struct or an index, get the actual leaf value.
    """

    name = "trunk.get_leaf_value"
    tree = operand_def(TreeType)
    leaf = operand_def(LeafType)
    res = result_def(TensorType)

    def __init__(self, tree: SSAValue, leaf: SSAValue):
        output_res = leaf.results[0].type.target_type
        super().__init__(operands=[tree, leaf], result_types=[output_res])


@irdl_op_definition
class GetLeafOp(IRDLOperation):
    """
    Starting from a terminal node, get the corresponding leaf.
    """

    name = "trunk.get_leaf"
    tree = operand_def(TreeType)
    node = operand_def(NodeType)
    res = result_def(LeafType)

    def __init__(self, tree: SSAValue, node: SSAValue):
        leaf_type = SSAValue.get(tree).owner.results[0].type.leaf_type
        super().__init__(operands=[tree, node], result_types=[leaf_type])


@irdl_op_definition
class VisitNextNodeOp(IRDLOperation):
    name = "trunk.visit_next_node"
    tree = operand_def(TreeType)
    node = operand_def(NodeType)
    data_in = operand_def(TensorType)
    result = result_def(NodeType)
    root_node = opt_operand_def(NodeType)
    # mode = prop_def(StringAttr)

    def __init__(
        self,
        tree: SSAValue,
        node: SSAValue,
        data_in: SSAValue,
        root_node: SSAValue | None = None,
        # mode: str = "right_child",
    ):
        # assert mode in ["breadthfirst", "children", "right_child"]
        if root_node is None:
            root_node = []

        super().__init__(
            operands=[tree, node, data_in, root_node],
            # properties={"mode": StringAttr(mode)},
            result_types=[node.type],
        )



@irdl_op_definition
class IsLeafOp(IRDLOperation):
    """
    Check if the current node is a leaf of the tree
    """

    name = "trunk.is_leaf"
    tree = operand_def(TreeType)
    node = operand_def(NodeType)
    result = result_def(BoolType)

    def __init__(
        self,
        tree: SSAValue,
        node: SSAValue,
    ):
        super().__init__(operands=[tree, node], result_types=[BoolType])


@irdl_op_definition
class IsLeafConditionOp(IRDLOperation):
    """
    Check if the current node is a leaf of the tree
    """

    name = "trunk.is_leaf"
    tree = operand_def(TreeType)
    node = operand_def(NodeType)
    # result = result_def(BoolType)
    arguments: VarOperand = var_operand_def(AnyAttr())
    traits = frozenset([HasParent(scf.While), IsTerminator(), Pure()])

    def __init__(
        self,
        tree: SSAValue,
        node: SSAValue,
        *output_ops: SSAValue | Operation,
    ):
        super().__init__(operands=[tree, node, [output for output in output_ops]])


@irdl_op_definition
class GetRootOp(IRDLOperation):
    """
    Gets directly the root node, skipping the tree type
    """

    name = "trunk.get_root"
    ensemble = operand_def(TreeEnsembleType)
    tree = operand_def(TreeType)
    result = result_def(NodeType)

    def __init__(self, ensemble: SSAValue, tree: SSAValue):
        result_of_tree = tree.results[0].type.node_type
        super().__init__(
            operands=[ensemble, tree],
            result_types=[result_of_tree],
        )


@irdl_op_definition
class TreeEnsembleConstantOp(IRDLOperation):
    """
    Constant data of the ensemble.
    """

    name = "trunk.tree_ensemble_constant"
    result: OpResult = result_def(TreeEnsembleType)

    # Attributes
    ensemble = attr_def(TreeEnsembleAttr)

    def __init__(
        self,
        ensemble_attr: TreeEnsembleAttr,
        return_type: TreeEnsembleType,
    ):
        super().__init__(
            operands=[],
            result_types=[return_type],
            attributes={
                "ensemble": ensemble_attr,
            },
        )


@irdl_op_definition
class PostTransform(IRDLOperation):
    name = "trunk.post_transform"
    buffer = operand_def(MemRefType)
    mode = attr_def(StringAttr)

    def __init__(self, buffer: SSAValue, mode: StringAttr):
        super().__init__(operands=[buffer], attributes={"mode": StringAttr(mode)})


@irdl_op_definition
class GetTreeOp(IRDLOperation):
    name = "trunk.get_tree"
    tree_ensemble = operand_def(TreeEnsembleType)
    tree_index = operand_def(IndexType)
    result = result_def(TreeType)

    def __init__(
        self,
        tree_ensemble: SSAValue,
        tree_index: SSAValue,
        result_type: TreeType,
    ):
        super().__init__(
            operands=[tree_ensemble, tree_index], result_types=[result_type]
        )


@irdl_op_definition
class TraverseTreeOp(IRDLOperation):
    name = "trunk.traverse_tree_op"
    tree = operand_def(TreeType)
    data_in = operand_def(MemRefType)
    result = result_def(LeafType)

    def __init__(
        self,
        tree: SSAValue,
        data_in: SSAValue,
        output_leaf_type: LeafType,
    ):
        super().__init__(operands=[tree, data_in], result_types=[output_leaf_type])


Trunk = Dialect(
    "trunk",
    [
        TreeEnsembleConstantOp,
        PostTransform,
        GetTreeOp,
        TraverseTreeOp,
        GetRootOp,
        IsLeafOp,
        VisitNextNodeOp,
        GetLeafOp,
        GetLeafValueOp,
        AggregateLeafOp,
    ],
    [],
)
