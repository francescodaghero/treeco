import numpy as np
from xdsl.dialects.builtin import (
    ModuleOp,
    ModuleOp,
)
from typing import Optional

from xdsl.context import MLContext
from xdsl.passes import ModulePass
from xdsl.transforms.printf_to_llvm import PrintfToLLVM
from treeco.targets.codegen_main import generate_main_function, find_func
import subprocess
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint


class AddLLVMMain(RewritePattern):
    def __init__(self, test_data: Optional[np.array] = None):
        self.test_data = test_data

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: ModuleOp,
        rewriter: PatternRewriter,
    ):

        main_func = find_func(op, "main")
        # Avoids multiple calls, only one main should be present!
        if main_func is not None:
            return
        main = generate_main_function(inference_module_op=op, test_data=self.test_data)
        pos = InsertPoint.at_end(op.body.block)
        rewriter.inline_block(main.body.block, pos)


class AddLLVMMainPass(ModulePass):
    name = "generate-and-add-main"

    def apply(self, ctx: MLContext, op: ModuleOp, test_data: Optional[np.array] = None):
        # Get the base main function
        PatternRewriteWalker(AddLLVMMain(test_data)).rewrite_module(op)
        PrintfToLLVM().apply(ctx, op)


import pathlib


def compile_and_run(build_dir=".", mlir_path="model.mlir"):
    # MLIR to LL
    build_dir: pathlib.Path = pathlib.Path(build_dir)
    mlir_path: pathlib.Path = pathlib.Path(mlir_path)
    ll_path = build_dir / pathlib.Path(str(mlir_path).replace(".mlir", ".ll"))
    o_path = build_dir / pathlib.Path(str(mlir_path).replace(".mlir", ""))

    success = subprocess.check_call(
        ["mlir-translate", "-mlir-to-llvmir", mlir_path, "-o", ll_path]
    )
    if success != 0:
        raise RuntimeError("Failed to convert MLIR to LLVM IR")

    subprocess.check_call(["clang", "-O3", "-g", "-o", o_path, ll_path])
    # Run the command ./o_path, parse the output rows with the pattern:
    # OUTPUT[sample_id][class_id]=value
    # Return an output list of lists with dimensions [len(output), len(output[0])]
    output = subprocess.run([str(o_path)], capture_output=True)
    output = output.stdout.decode().strip().split('\n')
    output = [row for row in output if row.startswith('OUTPUT')]

    final_output = []
    for row in output:
        spl = row.split('=')
        value = float(spl[1])
        idx0 = int(spl[0].split('[')[1].split(']')[0])
        idx1 = int(spl[0].split('[')[2].split(']')[0])
        if len(final_output) <= idx0:
            final_output.append([])
        final_output[idx0].append(value)

    return final_output