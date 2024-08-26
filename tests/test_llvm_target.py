import json
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
from treeco.targets.llvm.library import compile_as_library, run_ctype_inference
from treeco.targets.llvm.standalone import AddLLVMMainPass, compile_and_run
from treeco.targets.llvm import target_transform_and_dump
import onnx
import pytest
import os
from .conftest import TMP_DIR
from treeco.utils import quantize


# Fixture to list all model files
def model_files():
    return [
        os.path.join(TMP_DIR, f) for f in os.listdir(TMP_DIR) if f.endswith(".onnx")
    ]


def model_id(model_file):
    # Extract relevant parts of the filename for a descriptive ID
    return os.path.basename(model_file).replace(".onnx", "")


@pytest.mark.parametrize("model_file", model_files(), ids=model_id)
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("pad_to_perfect", [False])  # TODO : Not completed yet
@pytest.mark.parametrize(
    "convert_to_voting,quantize_output_to_n_bits",
    [
        [False, None],
        [True, None],
        [False, 8],
        [False, 16],
        [False, 32],
    ],
)
@pytest.mark.parametrize(  # Truncate is tested with the first arg, using bits_input
    "quantize_input_to_n_bits",
    [8, 16, 32, None],
)
@pytest.mark.parametrize(
    "tree_algorithm,pad_to_min_depth",
    [["iterative", False], ["iterative", 10], ["iterative", "auto"]],
)
@pytest.mark.parametrize("quantize_index_arrays", [True, False])
def un_test_llvm_library_mode(
    tmpdir,
    tmp_path,
    model_file,
    batch_size,
    pad_to_perfect,
    convert_to_voting,
    quantize_output_to_n_bits,
    quantize_input_to_n_bits,
    tree_algorithm,
    pad_to_min_depth,
    quantize_index_arrays,
):
    model_onnx = onnx.load(model_file)
    kv = {}
    for prop in model_onnx.metadata_props:
        kv[prop.key] = prop.value

    X = np.asarray(json.loads(kv["X"]))
    y = np.asarray(json.loads(kv["y"]))
    bits_input = (
        int(json.loads(kv["bits_input"]))
        if json.loads(kv["bits_input"]) is not None
        else None
    )
    is_classification = "classification" in model_file
    truncate_input_to_n_bits = bits_input if quantize_input_to_n_bits is None else None

    min_val_input, max_val_input = X.min(), X.max()
    # Recast the data
    if bits_input is not None:
        X = X.astype(dtype=np.dtype(f"uint{bits_input}").type)
    elif quantize_input_to_n_bits:
        min_val_input = X.min()
        max_val_input = X.max()
        X, _, _ = quantize(
            data=X, min_val=X.min(), max_val=X.max(), precision=quantize_input_to_n_bits
        )
    if is_classification:
        y = y.astype(dtype=np.int64)

    ensemble_ast = parse_ensemble(model_file)
    ensemble_ir, ctx = generate_ir(ensemble_ast, batch_size=batch_size)

    module_op = crown_transform(
        ensemble_ir,
        ctx,
        convert_to_voting=convert_to_voting,
        prune_to_n_trees=False,  # Disabled atm
        pad_to_perfect=pad_to_perfect,
        quantize_input_to_n_bits=quantize_input_to_n_bits,
        truncate_input_to_n_bits=truncate_input_to_n_bits,
        min_val_input=min_val_input,
        max_val_input=max_val_input,
        quantize_output_to_n_bits=quantize_output_to_n_bits,
    )

    # Note, this should be tested somewhere else, as it is used as golden
    ensemble_attr: treeco.TreeEnsembleAttr = find_operation_in_module(
        module_op=module_op, target_op=crown.TreeEnsembleOp
    ).ensemble
    ensemble_data: Ensemble = Ensemble.parse_attr(ensemble_attr)

    golden_out = ensemble_data.predict(X)

    # Lower to Trunk
    module_op = trunk_transform(
        module_op, ctx, tree_algorithm=tree_algorithm, pad_to_min_depth=pad_to_min_depth
    )
    # Lower to arith/memref
    module_op = root_transform(
        module_op, ctx, quantize_index_arrays=quantize_index_arrays
    )
    # Lower to llvm ir
    fname = model_file.replace(".onnx", ".mlir")
    # fname = os.path.join(tmpdir, fname.split("/")[-1])

    target_transform_and_dump(fname, module_op=module_op, ctx=ctx)

    lib = compile_as_library(".", fname)
    if quantize_output_to_n_bits:
        dtype_out = np.dtype(f"uint{quantize_output_to_n_bits}").type
    elif convert_to_voting:
        dtype_out = np.uint8
    else:
        dtype_out = np.float32

    buffer_out = np.zeros((batch_size, ensemble_data.n_targets), dtype=dtype_out)
    run_ctype_inference(lib, X[:batch_size], buffer_out)

    assert np.allclose(golden_out[:batch_size], buffer_out), buffer_out


@pytest.mark.parametrize("model_file", model_files(), ids=model_id)
@pytest.mark.parametrize("batch_size", [1, 2], ids = lambda x: f"bs={x}")
@pytest.mark.parametrize("pad_to_perfect", [False], ids = lambda x : f"pad={x}" )  # TODO : Not completed yet
@pytest.mark.parametrize(
    "convert_to_voting,quantize_output_to_n_bits",
    [
        [False, None],
        [True, None],
        [False, 8],
        [False, 16],
        [False, 32],
    ],
    #ids = lambda x : [f"voting={x[0]}", f"quantize={x[1]}"]
)
@pytest.mark.parametrize(  # Truncate is tested with the first arg, using bits_input
    "quantize_input_to_n_bits",
    [8, 16, 32, None],
    ids = lambda x : f"qin={x}"
)
@pytest.mark.parametrize(
    "tree_algorithm,pad_to_min_depth",
    [["iterative", False], ["iterative", 10], ["iterative", "auto"]],
    #ids = lambda x,y : (f"algo={x}", f"pad={y}")
)
@pytest.mark.parametrize("quantize_index_arrays", [True, False], ids = lambda x : f"qidx={x}")
def test_llvm_with_main(
    tmpdir,
    tmp_path,
    model_file,
    batch_size,
    pad_to_perfect,
    convert_to_voting,
    quantize_output_to_n_bits,
    quantize_input_to_n_bits,
    tree_algorithm,
    pad_to_min_depth,
    quantize_index_arrays,
):
    model_onnx = onnx.load(model_file)
    kv = {}
    for prop in model_onnx.metadata_props:
        kv[prop.key] = prop.value

    X = np.asarray(json.loads(kv["X"]))
    y = np.asarray(json.loads(kv["y"]))
    bits_input = (
        int(json.loads(kv["bits_input"]))
        if json.loads(kv["bits_input"]) is not None
        else None
    )
    is_classification = "classification" in model_file
    truncate_input_to_n_bits = bits_input if quantize_input_to_n_bits is None else None

    min_val_input, max_val_input = X.min(), X.max()
    # Recast the data
    if bits_input is not None:
        X = X.astype(dtype=np.dtype(f"uint{bits_input}").type)
    elif quantize_input_to_n_bits:
        min_val_input = X.min()
        max_val_input = X.max()
        X, _, _ = quantize(
            data=X, min_val=X.min(), max_val=X.max(), precision=quantize_input_to_n_bits
        )
    if is_classification:
        y = y.astype(dtype=np.int64)

    ensemble_ast = parse_ensemble(model_file)
    ensemble_ir, ctx = generate_ir(ensemble_ast, batch_size=batch_size)

    module_op = crown_transform(
        ensemble_ir,
        ctx,
        convert_to_voting=convert_to_voting,
        prune_to_n_trees=False,  # Disabled atm
        pad_to_perfect=pad_to_perfect,
        quantize_input_to_n_bits=quantize_input_to_n_bits,
        truncate_input_to_n_bits=truncate_input_to_n_bits,
        min_val_input=min_val_input,
        max_val_input=max_val_input,
        quantize_output_to_n_bits=quantize_output_to_n_bits,
    )

    # Note, this should be tested somewhere else, as it is used as golden
    ensemble_attr: treeco.TreeEnsembleAttr = find_operation_in_module(
        module_op=module_op, target_op=crown.TreeEnsembleOp
    ).ensemble
    ensemble_data: Ensemble = Ensemble.parse_attr(ensemble_attr)

    golden_out = ensemble_data.predict(X)

    # Lower to Trunk
    module_op = trunk_transform(
        module_op, ctx, tree_algorithm=tree_algorithm, pad_to_min_depth=pad_to_min_depth
    )
    # Lower to arith/memref
    module_op = root_transform(
        module_op, ctx, quantize_index_arrays=quantize_index_arrays
    )
    # Lower to llvm ir
    fname = model_file.replace(".onnx", ".mlir")
    fname = os.path.join(tmpdir, fname.split("/")[-1])
    AddLLVMMainPass().apply(ctx, module_op, test_data=X[:batch_size])
    target_transform_and_dump(fname, module_op=module_op, ctx=ctx)

    output = compile_and_run(build_dir=tmpdir, mlir_path=fname)
    output = np.asarray(output)
    assert np.allclose(golden_out[:batch_size], output)
