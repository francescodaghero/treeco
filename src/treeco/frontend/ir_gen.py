from __future__ import annotations

from xdsl.dialects.builtin import (
    ModuleOp,
    MemRefType,
    ArrayAttr,
    IntegerType,
    IndexType,
    StringAttr,
)
from xdsl.ir import Block, Region, SSAValue
from xdsl.dialects.func import FuncOp
from xdsl.dialects import func

from xdsl.dialects.builtin import DenseIntOrFPElementsAttr
from treeco.dialects import onnxml
from xdsl.dialects.builtin import f32


def convert_types_and_attr(attributes: dict) -> dict:
    """
    Base conversion function to parse the onnxml object

    Parameters
    ----------
    attributes : dict
        Mapping between the attribute name and its value

    Returns
    -------
    dict
        An xdsl compatible dictionary
    """
    converted_dict = {}
    for k, v in attributes.items():
        if isinstance(v, bytes):
            converted_dict[k] = StringAttr(v.decode("utf-8"))
        elif isinstance(v, list):
            shape = (len(v),)
            if all(isinstance(i, int) for i in v):
                basetype = IntegerType(32) if "id" not in k else IndexType()
                converted_dict[k] = DenseIntOrFPElementsAttr.tensor_from_list(
                    v, basetype, shape=shape
                )
            elif all(isinstance(i, float) or isinstance(i, int) for i in v):
                converted_dict[k] = DenseIntOrFPElementsAttr.tensor_from_list(
                    v, f32, shape=shape
                )
            elif all(isinstance(i, str) for i in v):
                converted_dict[k] = ArrayAttr([StringAttr(i) for i in v])
            elif all(isinstance(i, bytes) for i in v):
                converted_dict[k] = ArrayAttr(
                    [StringAttr(i.decode("utf-8")) for i in v]
                )
    return converted_dict


def ir_gen(parsed_onnx: dict, batch_size=1) -> ModuleOp:
    """
    Converts the parsed ONNX model to an XDSL module using the onnxml dialect

    Parameters
    ----------
    parsed_onnx : dict
        The onnx model parsed
    batch_size : int, optional
        Batch size of the inference function, by default 1

    Returns
    -------
    ModuleOp
        Xdsl module_op using the func and onnxml dialects.
    """
    input_shape = parsed_onnx.pop("input_shape")
    output_shape = parsed_onnx.pop("output_shape")

    # Declare the inputs and outputs of the function
    input_type = MemRefType(f32, [batch_size, input_shape])
    output_type = MemRefType(f32, [batch_size, output_shape])

    # Convert to xdsl compatible types
    parsed_onnx = convert_types_and_attr(parsed_onnx)

    # The input type is always a tensor of float32, with dimensions [batch_size, num_features]
    function = FuncOp.from_region(
        name="inference",
        input_types=[input_type, output_type],
        return_types=[],
        visibility="public",
        region=Region(
            [
                func_block := Block(
                    ops=[],
                    arg_types=[input_type, output_type],
                )
            ]
        ),
    )

    # Generate the TreeEnsembleOp based on the model type
    if "TreeEnsembleRegressor" in parsed_onnx:
        ensemble_op: onnxml.TreeEnsembleRegressor = ir_gen_regressor(
            parsed_onnx, input_data=func_block.args[0], output_data=func_block.args[1]
        )
    else:
        ensemble_op: onnxml.TreeEnsembleClassifier = ir_gen_classifier(
            parsed_onnx, input_data=func_block.args[0], output_data=func_block.args[1]
        )

    # Add the return statement to the function block
    func_block.add_ops([ensemble_op, func.Return()])

    # Generate the ModuleOp
    module = ModuleOp([function])
    return module


def ir_gen_regressor(
    onnx_model: dict, input_data: SSAValue, output_data: SSAValue
) -> onnxml.TreeEnsembleRegressor:
    op = onnxml.TreeEnsembleRegressor(
        buffer_in=input_data, buffer_out=output_data, **onnx_model
    )
    return op


def ir_gen_classifier(
    onnx_model: dict, input_data: SSAValue, output_data: SSAValue
) -> onnxml.TreeEnsembleClassifier:
    op = onnxml.TreeEnsembleClassifier(
        buffer_in=input_data, buffer_out=output_data, **onnx_model
    )
    return op
