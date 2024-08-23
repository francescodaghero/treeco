""" 
Mlir-opt with some default passes
"""

from xdsl.transforms.mlir_opt import MLIROptPass
from xdsl.dialects.builtin import ModuleOp
from xdsl.context import MLContext
import shutil
import os
from typing import List


def mlir_opt_pass(
    module_op: ModuleOp,
    ctx: MLContext,
    additional_args: List = [],
    allow_unregistered_dialects: bool = True,
    mlir_opt_path: str = "",
):

    mlir_path = os.path.join(mlir_opt_path, "mlir-opt")
    if not shutil.which(mlir_path):
        print("mlir-opt not found")
        return module_op
    # Conditional args
    arguments = ["-allow-unregistered-dialect"] if allow_unregistered_dialects else []
    arguments += [
        # A bad default since xDSL has some trouble parsing back hexs
        "-mlir-print-elementsattrs-with-hex-if-larger=9999",
        "-mlir-print-op-generic",
        "-scf-for-loop-canonicalization",
        "--loop-invariant-code-motion",
        "--loop-invariant-subset-hoisting",
        "--canonicalize",
        "-cse",
    ]
    arguments += additional_args
    arguments += [
        "-mlir-print-op-generic",
        "-scf-for-loop-canonicalization",
        "--loop-invariant-code-motion",
        "--loop-invariant-subset-hoisting",
        "--canonicalize",
        "-cse",
    ]
    MLIROptPass(
        executable=mlir_path,
        arguments=arguments,
    ).apply(ctx=ctx, op=module_op)
