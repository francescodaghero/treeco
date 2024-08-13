"""
Globals storing indices of the tree ensemble can be quantized.
However, this requires an additional cast when they are loaded from memory
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

from treeco.utils.numpy_to_xdsl import convert_np_to_tensor
from treeco.utils.xdsl_to_numpy import convert_dense_to_np


def quantize_global_op(op: memref.Global):
    value = op.initial_value
    np_arr: np.array = convert_dense_to_np(value)
    np_arr = np_arr.astype(np.min_scalar_type(np_arr.max()))
    new_value = convert_np_to_tensor(np_arr, is_signless=True)

    return memref.Global.get(
        sym_name=op.sym_name,
        sym_type=builtin.MemRefType(
            element_type=new_value.type.element_type,
            shape=op.type.shape,
            layout=op.type.layout,
            memory_space=op.type.memory_space,
        ),
        initial_value=new_value,
        sym_visibility=op.sym_visibility,
        constant=op.constant,
        alignment=op.alignment,
    )


def find_global_op_by_name(module_op, name):
    for op in module_op.walk():
        if isinstance(op, memref.Global) and op.sym_name == name:
            return op

    raise LookupError(f"Global with name {name} not found in module")


class MemrefQuantizeGlobalIndex(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: memref.GetGlobal,
        rewriter: PatternRewriter,
    ):
        if not isinstance(op.memref.type.get_element_type(), builtin.IndexType):
            return

        # Quantize the global op
        global_op: memref.Global = find_global_op_by_name(
            op.get_toplevel_object(), op.name_.root_reference
        )
        # Only if constant, so not store along the way
        if not global_op.constant:
            return
        quant_global = quantize_global_op(global_op)
        rewriter.replace_op(global_op, quant_global, [])

        # Replace the memref
        new_op = memref.GetGlobal(name=op.name_, return_type=quant_global.type)
        rewriter.replace_matched_op(
            new_op,
            [new_op.memref],
        )
        new_base_type = new_op.memref.type.get_element_type()

        uses = list(new_op.results[0].uses)
        for use in uses:
            op_use = use.operation
            new_results = list()
            for res_type in op_use.results:
                if isinstance(res_type.type, builtin.IndexType):
                    new_results.append(new_op.results[0].type.element_type)

                elif hasattr(res_type.type, "element_type") and isinstance(
                    res_type.type.element_type, builtin.IndexType
                ):
                    new_type = type(res_type.type)(
                        element_type=new_base_type, shape=res_type.type.get_shape()
                    )
                    new_results.append(new_type)

                else:
                    new_results.append(res_type)

            new_use = type(op_use)(
                operands=op_use.operands,
                result_types=new_results,
                properties=op_use.properties,
                attributes=op_use.attributes,
                regions=op_use.regions,
            )
            extend_sign = arith.ExtUIOp(
                new_use,
                builtin.IntegerType(64, builtin.Signedness.SIGNLESS),
            )
            cast_to_idx = arith.IndexCastOp(
                extend_sign, target_type=builtin.IndexType()
            )
            rewriter.replace_op(
                op_use,
                [new_use, extend_sign, cast_to_idx],
                [cast_to_idx.result],
            )


class MemrefQuantizeGlobalIndexPass(ModulePass):
    name = "memref-quantize-global-index-pass"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(MemrefQuantizeGlobalIndex()).rewrite_module(op)
        op.verify()
