""" 
Crown Dialect : The top of the tree, a.k.a the closest to the parser/user.

This dialect implements a single operation, TreeEnsembleOp, which is a 
multi-input (batched), multi-tree inference operation.

At this level, quantization, pruning, padding etc require changing only few elements.

"""

from treeco import utils as utils
from xdsl.ir import Dialect
from treeco.dialects.treeco import TreeEnsembleAttr
from xdsl.irdl import (
    irdl_op_definition,
    attr_def,
    IRDLOperation,
    operand_def,
)
from xdsl.dialects.builtin import (
    AnyFloat,
    IntegerType,
    MemRefType,
    SSAValue,
    IntegerAttr,
    FloatAttr,
    Signedness,
)

AnyNumericType = AnyFloat | IntegerType
AnyNumericAttr = FloatAttr | IntegerAttr
BoolType: IntegerType = IntegerType(1, Signedness.SIGNLESS)


@irdl_op_definition
class TreeEnsembleOp(IRDLOperation):
    """

    A multi-input (Batched), multi-tree inference operation.
    Almost a one-to-one mapping of the ONNX-ML TreeEnsembles, but general
    for regressors and classifiers.
    At this level, quantization/pruning/logits->voting etc are ideal, as no loops
    have been represented yet.
    It works at the memref level, as the buffer allocation is done by the runtime.

    """

    name = "crown.tree_ensemble_predict"
    buffer_in = operand_def(MemRefType)
    buffer_out = operand_def(MemRefType)

    # Attributes
    ensemble = attr_def(TreeEnsembleAttr)

    def __init__(
        self,
        buffer_in: SSAValue,
        buffer_out: SSAValue,
        ensemble_attr: TreeEnsembleAttr,
    ):
        super().__init__(
            operands=[buffer_in, buffer_out],
            attributes={
                "ensemble": ensemble_attr,
            },
        )


Crown = Dialect(
    "crown",
    [
        TreeEnsembleOp,
    ],
    [],
)
