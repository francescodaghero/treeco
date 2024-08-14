from treeco.compiler import (
    parse_ensemble,
    generate_ir,
    crown_transform,
    trunk_transform,
    target_transform,
    dump_to_llvm,
)
from treeco.model import Ensemble
from treeco.dialects import treeco, crown
from treeco.utils import find_operation_in_module
import numpy as np
from treeco.targets.llvm.library import compile_as_library, run_inference


def test_llvm_library(batch_size=2):
    np.random.seed(0)
    ensemble_ast = parse_ensemble("rf.onnx")
    ensemble_ir, ctx = generate_ir(ensemble_ast, batch_size=batch_size)
    module_op = crown_transform(ensemble_ir, ctx)

    # TODO: Use actual test data
    # Use the ensemble for prediction in python
    ensemble_attr: treeco.TreeEnsembleAttr = find_operation_in_module(
        module_op=module_op, target_op=crown.TreeEnsembleOp
    ).ensemble
    ensemble_data: Ensemble = Ensemble.parse_attr(ensemble_attr)
    test_data = np.zeros((batch_size, ensemble_data.n_features), dtype=np.float32)
    golden_out = ensemble_data.predict(test_data).sum(axis=1)

    # Lower to Trunk
    module_op = trunk_transform(module_op, ctx)
    # Lower to arith/memref
    module_op = target_transform(module_op, ctx, target="llvm")
    # Lower to llvm ir
    dump_to_llvm("output.mlir", module_op=module_op, ctx = ctx)

    lib = compile_as_library(".", "output.mlir")
    run_inference(
        lib,
        test_data,
        buffer_out := np.zeros((batch_size, ensemble_data.n_targets), dtype=np.float32),
    )

    assert np.allclose(golden_out, buffer_out)
    print("Success")


test_llvm_library()
