import numpy as np
from xdsl.dialects.builtin import (
    ModuleOp,
    ModuleOp,
    UnitAttr,
)
from typing import Optional

from xdsl.dialects import memref, arith, func, printf, scf, builtin
from xdsl.ir import Block, Region
from xdsl.builder import ImplicitBuilder
from treeco.utils.numpy_to_xdsl import convert_np_to_tensor
from treeco.utils.xdsl_to_numpy import convert_xdsl_to_np_type


def find_func(
    module_op: ModuleOp, func_name: str = "inference"
) -> Optional[func.FuncOp]:
    for op in module_op.walk():
        if isinstance(op, func.FuncOp) and op.sym_name.data == func_name:
            return op


def generate_main_function(
    inference_module_op: ModuleOp, test_data: np.array
) -> builtin.ModuleOp:
    """
    Returns a general main function.
    """

    main_block = Block(arg_types=())
    main_func = func.FuncOp.from_region(
        name="main",
        input_types=[],
        return_types=[],
        region=Region([main_block]),
        visibility="public",
    )

    inference_func = find_func(module_op=inference_module_op, func_name="inference")
    # Allocate two empty buffers. Inputs are filled, outputs are set to 0
    # Note that the shape of the input/output should be divisible by the batch.
    batch_size = inference_func.args[0].type.get_shape()[0]
    n_features = inference_func.args[0].type.get_shape()[1]
    n_outputs = inference_func.args[1].type.get_shape()[1]

    assert (
        test_data.shape[0] % batch_size == 0
    ), "Batch size should be divisible by the input shape"
    assert (
        test_data.shape[1] == n_features
    ), "Number of features should match the input shape"

    # Convert the data to a global memref, care, the data should be ALREADY quantized
    # if the ensemble is quantized.
    data_in = convert_np_to_tensor(
        test_data, is_index=False, is_signless=True # else False
    )
    # Output buffer with same dtype as required by inference.
    np_data_out = np.zeros((test_data.shape[0], n_outputs)).astype(
        convert_xdsl_to_np_type(inference_func.args[1].type.get_element_type())
    )
    data_out = convert_np_to_tensor(
        np_data_out,
        is_index=False,
        is_signless=True,
    )
    data_in_memref_type = memref.MemRefType(
        element_type= data_in.type.element_type,
        shape = data_in.type.get_shape(),
    )
    data_out_memref_type = memref.MemRefType(
        element_type= data_out.type.element_type,
        shape = data_out.type.get_shape(),
    )

    # Input buffer becomes a global memref.
    input_global_op = memref.Global.get(
        sym_name=builtin.StringAttr("input_data"),
        sym_type=inference_func.args[0].type,
        initial_value=data_in,
        sym_visibility=builtin.StringAttr("public"),
        constant=UnitAttr(),
        alignment=None,
    )
    # Output buffer becomes a global memref.
    output_global_op = memref.Global.get(
        sym_name=builtin.StringAttr("output_data"),
        sym_type=inference_func.args[1].type,
        initial_value=data_out,
        sym_visibility=builtin.StringAttr("public"),
        constant=None,
        alignment=None,
    )

    # Get the globals
    input_global_get = memref.GetGlobal(
        name="input_data",
        return_type=data_in_memref_type,
    )
    output_global_get = memref.GetGlobal(
        name="output_data",
        return_type=data_out_memref_type,
    )

    # Add a for loop to feed batch-by-batch
    batch_constant = arith.Constant.from_int_and_width(batch_size, builtin.IndexType())
    start_const = arith.Constant.from_int_and_width(0, builtin.IndexType())
    step_const = arith.Constant.from_int_and_width(1, builtin.IndexType())
    stop_const = arith.Constant.from_int_and_width(
        test_data.shape[0], builtin.IndexType()
    )
    loop_block = Block(arg_types=(builtin.IndexType(),))
    for_op = scf.For(
        lb=start_const,
        step=batch_constant,
        ub=stop_const,
        iter_args=[],
        body=Region(loop_block),
    )
    with ImplicitBuilder(loop_block) as (idx,):
        input_slice = memref.Subview.get(
            source=input_global_get,
            offsets=[idx, 0],
            sizes =[batch_size, n_features],
            strides=[1, 1],
            result_type= builtin.MemRefType(
                element_type = inference_func.args[0].type.element_type,
                shape = inference_func.args[0].type.get_shape(),
                layout= builtin.StridedLayoutAttr(
                    strides = [n_features, 1],
                    offset=builtin.NoneAttr()
                )
            )
            #inference_func.args[0].type,
        )
        output_slice = memref.Subview.get(
            source=output_global_get,
            offsets=[idx, 0],
            sizes=[batch_size, n_outputs],
            strides=[1, 1],
            result_type= builtin.MemRefType(
                element_type = inference_func.args[1].type.element_type,
                shape = inference_func.args[1].type.get_shape(),
                layout= builtin.StridedLayoutAttr(
                    strides = [n_outputs, 1],
                    offset=builtin.NoneAttr()
                )
            )
        )
        input_slice_simple = memref.Cast.get(
            source=input_slice,
            type=inference_func.args[0].type,
        )
        output_slice_simple = memref.Cast.get(
            source = output_slice,
            type=inference_func.args[1].type,
        )
        call = func.Call(
            "inference",
            arguments=[input_slice_simple, output_slice_simple],
            return_types=[],
        )
        scf.Yield()

    out_const = arith.Constant.from_int_and_width(n_outputs, builtin.IndexType())
    print_block = Block(arg_types=(builtin.IndexType(),))
    print_block_batch = Block(arg_types=(builtin.IndexType(),))
    batch_printf = scf.For(
        lb = start_const,
        step = step_const,
        ub = stop_const,
        iter_args=[],
        body= Region(print_block_batch),
    )

    with ImplicitBuilder(print_block_batch) as (batch_idx,):
        cls_printf = scf.For(
            lb = start_const,
            step = step_const,
            ub = out_const,
            iter_args=[],
            body= Region(print_block),
        )
        with ImplicitBuilder(print_block) as (idx,):
            output_val = memref.Load.get(
                ref=output_global_get,
                indices=[batch_idx, idx],
            )
            if isinstance(inference_func.args[1].type.element_type, builtin.IntegerType):
                output_val = arith.ExtUIOp(op = output_val, target_type = builtin.IntegerType(64, builtin.Signedness.SIGNLESS))
            pr = printf.PrintFormatOp("OUTPUT[{}][{}]={}",batch_idx, idx, output_val)
            scf.Yield()
        scf.Yield()

    r = func.Return()
    main_block.add_ops(
        [
            batch_constant,
            input_global_get,
            output_global_get,
            start_const,
            step_const,
            stop_const,
            for_op,
            out_const, batch_printf,
            r,
        ]
    )
    # Add the emitc interface to the main function
    # main_func.attributes["llvm.emit_c_interface"] = UnitAttr()

    main_module_op = builtin.ModuleOp(
        [
            input_global_op,
            output_global_op,
            main_func,
        ]
    )
    return main_module_op
