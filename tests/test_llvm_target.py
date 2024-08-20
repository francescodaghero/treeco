from treeco.compiler import (
    parse_ensemble,
    generate_ir,
    crown_transform,
    trunk_transform,
    root_transform,
)
from treeco.model import Ensemble
from treeco.dialects import treeco, crown
from treeco.utils import find_operation_in_module
import numpy as np
from treeco.targets.llvm.library import compile_as_library, run_inference
from treeco.targets.llvm import target_transform_and_dump


def test_llvm_library(batch_size=2):
    np.random.seed(0)
    ensemble_ast = parse_ensemble("rf.onnx")
    ensemble_ir, ctx = generate_ir(ensemble_ast, batch_size=batch_size)
    module_op = crown_transform(
        ensemble_ir,
        ctx,
        quantize_output_to_n_bits=8,
    )

    # TODO: Use actual test data
    # Use the ensemble for prediction in python
    ensemble_attr: treeco.TreeEnsembleAttr = find_operation_in_module(
        module_op=module_op, target_op=crown.TreeEnsembleOp
    ).ensemble
    ensemble_data: Ensemble = Ensemble.parse_attr(ensemble_attr)
    test_data = np.zeros((batch_size, ensemble_data.n_features), dtype=np.float32)
    golden_out = ensemble_data.predict_raw(test_data).sum(axis=1)

    # Lower to Trunk
    module_op = trunk_transform(module_op, ctx)
    # Lower to arith/memref
    module_op = root_transform(module_op, ctx)
    # Lower to llvm ir
    target_transform_and_dump("output.mlir", module_op=module_op, ctx=ctx)

    lib = compile_as_library(".", "output.mlir")
    run_inference(
        lib,
        test_data,
        buffer_out := np.zeros((batch_size, ensemble_data.n_targets), dtype=np.uint8),
    )

    assert np.allclose(golden_out, buffer_out)
    print("Success")


test_llvm_library()
