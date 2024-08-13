from pathlib import Path
from typing import Literal
import numpy as np

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
from treeco.lowering.emitc.convert_arith_to_emitc import ConvertArithToEmitcPass
from treeco.lowering.emitc.convert_memref_to_emitc import ConvertMemrefToEmitcPass
from treeco.lowering.emitc.convert_printf_to_emitc import ConvertPrintfToEmitcPass
from treeco.model.ensemble import Ensemble
from treeco.targets.cpp.add_entry_point import AddMainPass
from treeco.transforms import *
from treeco.utils.xdsl_utils import find_operation_in_module
from xdsl.transforms.printf_to_llvm import PrintfToLLVM
from treeco.targets.llvm.lower_to_llvm import dump_to_llvm


def context() -> MLContext:
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


def parse_ensemble(onnx_path: str) -> ModuleOp:
    parsed_model = Parser(Path(onnx_path)).parseModel()
    return parsed_model


def generate_ir(parsed_model, batch_size: int = 1) -> ModuleOp:
    ctx = context()
    module_op = ir_gen(parsed_onnx=parsed_model, batch_size=batch_size)
    return module_op, ctx


def crown_transform(
    module_op: ModuleOp,
    ctx: MLContext,
    # Convert to voting classifier
    convert_to_voting=False,
    # Ensemble pruning parameters
    prune_to_n_trees=False,
    pad_mode="None",
    # Input and output quantization parameters
    quantize_input_to_n_bits=None,
    truncate_input_to_n_bits=None,
    min_val_input=None,
    max_val_input=None,
    quantize_output_to_n_bits=None,
    min_val_output=None,
    max_val_output=None,
) -> ModuleOp:
    # TODO Avoid voting + output quantization
    # Convert to Crown IR
    ConvertOnnxmlToCrownPass().apply(ctx=ctx, op=module_op)
    if convert_to_voting:
        CrownConvertToVotingClassifierPass().apply(ctx, module_op)
    if prune_to_n_trees:
        CrownPruneTreesPass().apply(ctx, module_op, n_trees=prune_to_n_trees)
    if pad_mode != "None":
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
            min_val=min_val_output,
            max_val=max_val_output,
        )
    mlir_opt_pass(module_op, ctx, [])
    return module_op


def trunk_transform(
    module_op: ModuleOp,
    ctx: MLContext,
    tree_algorithm: Literal["iterative"] = "iterative",
    pad_ensemble_to_min_depth: Literal["auto"] | bool = False,
) -> ModuleOp:
    # From Crown to Trunk
    if tree_algorithm == "iterative":
        if pad_ensemble_to_min_depth:
            TrunkPadTreesPass().apply(
                ctx, module_op, min_depth_ensemble=pad_ensemble_to_min_depth
            )
        ConvertCrownToTrunkIterativePass().apply(ctx=ctx, op=module_op)

    mlir_opt_pass(module_op, ctx, [])
    # From Trunk/Treeco to Arith/Scf/Tensor/...
    return module_op


def target_transform(
    module_op: ModuleOp,
    ctx: MLContext,
    target: str,
):

    # Exit from Treeco custom type
    LowerTrunkPass().apply(ctx=ctx, op=module_op)
    LowerTreecoPass().apply(ctx=ctx, op=module_op)
    ConvertMlProgramToMemrefPass().apply(ctx=ctx, op=module_op)
    bufferize_pass(module_op, ctx)

    FoldMemRefSubViewChainPass().apply(ctx=ctx, op=module_op)
    mlir_opt_pass(module_op=module_op, ctx=ctx)
    MemrefQuantizeGlobalIndexPass().apply(ctx=ctx, op=module_op)
    mlir_opt_pass(module_op=module_op, ctx=ctx)
    # Linalg to scf loops
    mlir_opt_pass(module_op, ctx, ["--convert-linalg-to-loops"])

    if target == "cpp":
        ConvertMemrefToEmitcPass().apply(ctx=ctx, op=module_op)
        ConvertArithToEmitcPass().apply(ctx=ctx, op=module_op)
        ConvertPrintfToEmitcPass().apply(ctx=ctx, op=module_op)
    elif target == "llvm":
        # Disabled as I cannot make the mlir_c_inference function generated in the llvm ir work!
        # PrepareLLVMLoweringPass().apply(ctx=ctx, op=module_op)
        PrintfToLLVM().apply(ctx=ctx, op=module_op)
        convert_scf_to_cf_pass(module_op, ctx)

    return module_op
