"""
This dialect is an extension of the one available in XDSL
"""

from xdsl.dialects.tensor import *


@irdl_op_definition
class ExtractOp(IRDLOperation):

    name = "tensor.extract"

    tensor = operand_def(TensorType)
    indices = var_operand_def()
    result = result_def()

    @classmethod
    def get(
        cls,
        tensor: SSAValue,
        indices: Sequence[SSAValue] | SSAValue,
    ):
        if isinstance(indices, SSAValue):
            indices = [indices]
        res_val_type = SSAValue.get(tensor).type.element_type
        return cls(operands=[tensor, indices], result_types=[res_val_type])
        # super().__init__(operands=[tensor, indices], result_types=[res_val_type])


@irdl_op_definition
class InsertOp(IRDLOperation):

    name = "tensor.insert"

    scalar = operand_def()
    destination = operand_def(TensorType)
    indices = var_operand_def()
    result = result_def(TensorType)

    @classmethod
    def get(
        cls,
        scalar: SSAValue,
        destination: SSAValue,
        indices: Sequence[SSAValue] | SSAValue,
    ):
        if isinstance(indices, SSAValue):
            indices = [indices]
        res_val_type = SSAValue.get(destination).type
        return cls(operands=[scalar, destination, indices], result_types=[res_val_type])
        # super().__init__(operands=[scalar, destination, indices], result_types=[res_val_type])


Tensor = Dialect(
    "tensor",
    [o for o in Tensor.operations] + [ExtractOp, InsertOp],
    [o for o in Tensor.attributes] + [],
)
