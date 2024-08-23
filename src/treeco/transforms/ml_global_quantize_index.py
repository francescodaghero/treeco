""" 
Feels like an issue when bufferizing, so for now the compression is performed 
after bufferization.
"""

import numpy as np
from xdsl.context import MLContext
from xdsl.dialects import arith, builtin, memref
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from treeco.dialects.extended import ml_program, tensor
from treeco.utils.numpy_to_xdsl import convert_np_to_tensor
from treeco.utils.xdsl_to_numpy import convert_dense_to_np


def quantize_global_op(op: ml_program.Global):
    value = op.value
    np_arr: np.array = convert_dense_to_np(value)
    np_arr = np_arr.astype(np.min_scalar_type(np_arr.max()))
    new_value = convert_np_to_tensor(np_arr, is_signless=True)

    return ml_program.Global(
        sym_name=op.sym_name,
        type=new_value.type,
        is_mutable=op.is_mutable,
        value=new_value,
        sym_visibility=op.sym_visibility,
    )


def find_global_op_by_name(module_op, name):
    for op in module_op.walk():
        if isinstance(op, ml_program.Global) and op.sym_name == name:
            return op
    raise LookupError(f"Global with name {name} not found in module")


class MlGlobalQuantizeIndex(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: ml_program.GlobalLoadConstant,
        rewriter: PatternRewriter,
    ):
        if not isinstance(op.result.type.get_element_type(), builtin.IndexType):
            return

        # Quantize the global op
        global_op: ml_program.Global = find_global_op_by_name(
            op.get_toplevel_object(), op.global_attr.root_reference
        )
        # Only if constant, so not store along the way
        if global_op.is_mutable:
            return
        quant_global = quantize_global_op(global_op)
        rewriter.replace_op(global_op, quant_global, [])

        # Replace the memref
        new_op = ml_program.GlobalLoadConstant(
            global_attr=op.global_attr, result_type=quant_global.type
        )
        rewriter.replace_matched_op(
            [new_op],
            [new_op.results[0]],
        )


class MlGlobalQuantizeIndexPass(ModulePass):
    name = "ml-global-quantize-index-pass"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(MlGlobalQuantizeIndex()).rewrite_module(op)
        op.verify()
