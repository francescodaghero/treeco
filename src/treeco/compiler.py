from pathlib import Path
from typing import Literal, Union, Optional, Mapping, Any, Tuple

from xdsl.context import MLContext
from xdsl.dialects import (
    arith,
    builtin,
    cf,
    func,
    llvm,
    memref,
    printf,
    scf,
    linalg,
    affine,
)
from xdsl.dialects.builtin import ModuleOp

from treeco.dialects import crown, onnxml, treeco, trunk
from treeco.dialects.extended import bufferization, ml_program, tensor
from treeco.frontend.ir_gen import ir_gen
from treeco.frontend.parser import Parser
from treeco.lowering import *
from treeco.lowering.convert_crown_to_trunk import ConvertCrownToTrunkIterativePass
from treeco.transforms import *


def context() -> MLContext:
    """
    Just loading a bunch of dialects, not sure this is the best way to handle
    the context
    #TODO : Investigate further, should this be split in a context per transform function?
    """
    ctx = MLContext()
    ctx.load_dialect(treeco.Treeco)
    ctx.load_dialect(bufferization.Bufferization)
    ctx.load_dialect(trunk.Trunk)
    ctx.load_dialect(onnxml.Onnxml)
    ctx.load_dialect(crown.Crown)
    ctx.load_dialect(llvm.LLVM)
    ctx.load_dialect(builtin.Builtin)
    ctx.load_dialect(scf.Scf)
    ctx.load_dialect(memref.MemRef)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(printf.Printf)
    ctx.load_dialect(cf.Cf)
    ctx.load_dialect(tensor.Tensor)
    ctx.load_dialect(ml_program.MLProgram)
    ctx.load_dialect(linalg.Linalg)
    ctx.load_dialect(affine.Affine)
    # Not working, emitc cannot be parsed back because opaque is both attribute and type
    # ctx.load_dialect(emitc.Emitc)
    return ctx


def parse_ensemble(onnx_path: str) -> Mapping[str, Any]:
    """
    Parse the onnx model and return the MLIR ModuleOp

    Parameters
    ----------
    onnx_path : str
        Path to the onnx file

    Returns
    -------
    Mapping[str, Any]
        A dictionary representation of the ensemble in the ONNX file. A key:val for each
        field in the ensemble
    """
    parsed_model = Parser(Path(onnx_path)).parseModel()
    return parsed_model


def generate_ir(
    parsed_model: Mapping[str, Any], batch_size: int = 1
) -> Tuple[ModuleOp, MLContext]:
    """
    Convert the parsed model to MLIR

    Parameters
    ----------
    parsed_model : Mapping[str,Any]
        A mapping of the parsed ONNX model
    batch_size : int, optional
        The batch size of the inference function, by default 1

    Returns
    -------
    ModuleOp
        The ensemble IR using the ONNXML dialect
    MLContext
        The context for lowering the program
    """
    ctx = context()
    module_op = ir_gen(parsed_onnx=parsed_model, batch_size=batch_size)
    return module_op, ctx


def crown_transform(
    module_op: ModuleOp,
    ctx: MLContext,
    # Convert to voting classifier
    convert_to_voting: bool = False,
    # Ensemble pruning parameters
    prune_to_n_trees: Union[Literal[False], int] = False,
    pad_to_perfect: bool = False,
    # Input and output quantization parameters
    quantize_input_to_n_bits: Optional[int] = None,
    truncate_input_to_n_bits: Optional[int] = None,
    min_val_input: Optional[float] = None,
    max_val_input: Optional[float] = None,
    quantize_output_to_n_bits: Optional[int] = None,
) -> ModuleOp:
    """
    Apply a series of passes to the crown dialect ops.
    It includes:
    1. logits to vote conversion (convert_to_voting)
    2. tree pruning (prune_to_n_trees)
    3. tree pad to perfect (pad_to_perfect)
    3. input quantization (quantize..., min_val..., max_val...) or (truncate...)
    4. output quantization (quantize_output_to_n_bits)

    Parameters
    ----------
    module_op : ModuleOp
        The program IR
    ctx : MLContext
        The context
    convert_to_voting : bool, optional
        Flag to enable logits -> vote conversion (and related optimizations), by default False
    prune_to_n_trees : Union[Literal[False], int], optional
        Flag to enable tree pruning, by default False
    pad_to_perfect : bool
        Flag to enable tree padding, by default False
    quantize_input_to_n_bits : Optional[int], optional
        Flag to quantize the thresholds, by default None
    truncate_input_to_n_bits : Optional[int], optional
        Flag to truncate the alphas to int, by default None
    min_val_input : Optional[float], optional
        The smallest value of alphas for quantization, by default None
    max_val_input : Optional[float], optional
        The largest value of alphas for quantizations, by default None
    quantize_output_to_n_bits : Optional[int], optional
        Flag to enable output quantization, by default None

    Returns
    -------
    ModuleOp
        The modified program IR
    """

    # TODO Avoid voting + output quantization
    # Convert to Crown IR
    ConvertOnnxmlToCrownPass().apply(ctx=ctx, op=module_op)
    if convert_to_voting:
        CrownConvertToVotingClassifierPass().apply(ctx, module_op)
    if prune_to_n_trees:
        CrownPruneTreesPass().apply(ctx, module_op, n_trees=prune_to_n_trees)
    if pad_to_perfect:
        CrownPadTreesPerfectPass().apply(ctx, module_op)

    if truncate_input_to_n_bits:
        CrownRoundInputPass().apply(
            ctx,
            module_op,
            precision=truncate_input_to_n_bits,
        )
    elif quantize_input_to_n_bits:
        CrownQuantizeInputPass().apply(
            ctx,
            module_op,
            precision=quantize_input_to_n_bits,
            min_val=min_val_input,
            max_val=max_val_input,
        )
    if quantize_output_to_n_bits:
        CrownQuantizeLeavesPass().apply(
            ctx,
            module_op,
            precision=quantize_output_to_n_bits,
        )
    mlir_opt_pass(module_op, ctx, [])
    return module_op


def trunk_transform(
    module_op: ModuleOp,
    ctx: MLContext,
    tree_algorithm: Literal["iterative"] = "iterative",
    pad_to_min_depth: Union[int, Literal["auto"], False] = False,
) -> ModuleOp:
    """
    Conversion passes to lower from Crown to Trunk dialect and then
    optimize Trunk.
    Here, the algorithm to traverse the tree is the main focus.

    Parameters
    ----------
    module_op : ModuleOp
        The program IR
    ctx : MLContext
        The context
    tree_algorithm : Literal[&quot;iterative&quot;], optional
        The algorithm to visit the tree, by default "iterative"
    pad_to_min_depth : Union[int, Literal[&quot;auto&quot;], False], optional
        whether to push all leaves to a minimum depth, by default False

    Returns
    -------
    ModuleOp
        The updated program IR
    """
    # From Crown to Trunk
    if tree_algorithm == "iterative":
        if pad_to_min_depth:
            TrunkPadToMinDepthPass().apply(ctx, module_op, min_depth=pad_to_min_depth)
        ConvertCrownToTrunkIterativePass().apply(ctx=ctx, op=module_op)

    mlir_opt_pass(module_op, ctx, [])
    # From Trunk/Treeco to Arith/Scf/Tensor/...
    return module_op


def root_transform(
    module_op: ModuleOp,
    ctx: MLContext,
    bufferize: bool = True,
    quantize_index_arrays=True,
):
    """
    Convert from trunks/treeco to arith/tensor.
    Optionally, bufferize with MLIR + a custom pass for ml_program
    """

    # Exit from Treeco custom type
    LowerTrunkPass().apply(ctx=ctx, op=module_op)
    LowerTreecoPass().apply(ctx=ctx, op=module_op)

    if bufferize:
        # Prepare the program for bufferization
        ConvertMlProgramToMemrefPass().apply(ctx=ctx, op=module_op)
        # Perform the actual bufferization
        bufferize_pass(module_op, ctx)

        # Optimize the program
        # Quantize the indices, from i64 to iN + add some casts.
        FoldMemRefSubViewChainPass().apply(ctx=ctx, op=module_op)
        if quantize_index_arrays:
            MemrefQuantizeGlobalIndexPass().apply(ctx=ctx, op=module_op)
        mlir_opt_pass(module_op=module_op, ctx=ctx)
        mlir_opt_pass(module_op, ctx, ["--convert-linalg-to-loops"])

    mlir_opt_pass(module_op=module_op, ctx=ctx)
    return module_op
