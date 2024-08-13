from typing import Any, List
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl.context import MLContext

from treeco.model.ensemble import Ensemble
from xdsl.dialects.builtin import (
    IntegerAttr,
    ModuleOp,
    TensorType,
    MemRefType,
    NoneType,
    IndexType,
    DenseArrayBase,
    StridedLayoutAttr,
    IntAttr,
    NoneAttr,
    i64,
    ArrayAttr,
)
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
    TypeConversionPattern,
    attr_type_rewrite_pattern,
)
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.builder import Builder, ImplicitBuilder
from treeco.dialects import crown, trunk
from xdsl.dialects import func, scf, arith, memref, affine
from treeco.utils import tensor_to_memref, I64_MIN, convert_np_to_arrayattr
from treeco.dialects import emitc

import numpy as np
import time


def get_unique_name() -> str:
    # A function that returns a really unique name
    return str(time.time_ns())


class MemrefToArrayType(TypeConversionPattern):
    @attr_type_rewrite_pattern
    def convert_type(self, typ: MemRefType):
        et = typ.element_type
        sh = typ.shape
        return emitc.ArrayType(
            element_type=et,
            shape=sh,
        )


class AllocToGlobal(RewritePattern):
    def look_forward(self, op: memref.Alloc):
        values = np.zeros(op.results[0].type.shape)
        ops = list()
        # Look at the uses, find constants and return them
        for user in op.result.users:
            if isinstance(user, memref.Store) and isinstance(
                user.operands[0], arith.Constant
            ):
                if len(user.operands[0].uses) == 1:
                    ops.append(user)

            elif isinstance(user, arith.Store) and isinstance(
                user.operands[0], arith.Constant
            ):
                if len(user.operands[0].uses) == 1:
                    ops.append(user)
        return ops, values

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: memref.Alloc,
        rewriter: PatternRewriter,
    ):
        constant_ops, init_values = self.look_forward(op)
        for operation in constant_ops:
            rewriter.erase_op(operation)

        name = get_unique_name()

        globale = emitc.Global(
            name=name,
            type_=op.result.type,
            initial_value=convert_np_to_arrayattr(init_values),
        )
        modulo = op.get_toplevel_object()
        rewriter.insert_op([globale], InsertPoint.at_start(modulo.body.blocks[0]))
        getter = emitc.GetGlobal(name=name, return_type=op.result.type)
        rewriter.replace_matched_op(getter, [getter.results[0]])


class AllocToAllocOpaque(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: memref.Alloc,
        rewriter: PatternRewriter,
    ):
        name = get_unique_name()
        modulo = op.get_toplevel_object()
        malloc = emitc.CallOpaque.from_malloc(
            arguments=[IntAttr(size)], return_type=op.result.type.element_type
        )

        globale = emitc.Global(
            name=name,
            type_=op.result.type,
            initial_value=convert_np_to_arrayattr(np.zeros(op.result.type.shape)),
        )
        rewriter.insert_op([globale], InsertPoint.at_start(modulo.body.blocks[0]))
        getter = emitc.GetGlobal(name=name, return_type=op.result.type)
        rewriter.replace_matched_op(getter, [getter.results[0]])


class StoreToAssign(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: memref.Store,
        rewriter: PatternRewriter,
    ):
        memory_loc = emitc.Subscript(operand=op.operands[1], indices=op.indices)
        assigned = emitc.Assign(var=memory_loc, value=op.operands[0])
        rewriter.replace_matched_op([memory_loc, assigned], [])


class GlobalToGlobal(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: memref.Global,
        rewriter: PatternRewriter,
    ):
        globale = emitc.Global(
            name=op.sym_name,
            initial_value=op.initial_value,
            type_=op.type,
        )
        rewriter.replace_matched_op([globale], [])


class GetGlobalToGetGlobal(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.GetGlobal, rewriter: PatternRewriter):
        new_use = emitc.GetGlobal(name=op.name_, return_type=op.results[0].type)
        rewriter.replace_matched_op([new_use], [new_use.results[0]])


class MergeSubviewSlices(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.Subview, rewriter: PatternRewriter):
        # Check all ops using the subview,
        # Try to merge the idx of the Subview into the Loads and Stores
        # TODO: Implement this
        # NOTE: What happens if the size is different?
        # NOTE: How do i handle the offset?
        # NOTE: What if the parent is a Subview? Should not happen with folding
        # NOTE: How to handle the strides?

        # First, it should be simplified, if it was a constant, just remove it.
        # TODO : This should be optimized before
        if op.operands[0].type.shape == op.results[0].type.shape:
            rewriter.replace_matched_op([], [op.operands[0]])

        # IGNORE: Dynamic Sizes, ignored for now
        # SOLUTION: Dynamic Strides, not supported atm
        if len(op.strides) != 0:
            return
        # SOLUTION: Static strides, unsupported, return if != 1
        for e in op.static_strides.as_tuple():
            if e != 1:
                return

        # SOLUTION: Static sizes are ignored atm, return if not equal to input size
        # TREECO: We never reduce the rank
        for d1, d2 in zip(op.static_sizes.as_tuple(), op.operands[0].type.shape):
            if d1 != 1 and d1 != d2.data:
                return

        # Check that only supported OP will have the input modified
        for user in op.results[0].uses:
            if not (
                isinstance(user.operation, memref.Load)
                or isinstance(user.operation, memref.Store)
            ):
                return

        # Handle the static or dynamic offsets
        # Since this pass supports only shapes identical to the original,
        # every subview is just a slice op
        off_idx = 0
        per_dim_offsets = []
        for off in op.static_offsets.as_tuple():
            if off < 0:
                ssa_var = op.offsets[off_idx]
                per_dim_offsets.append(ssa_var)
            elif off == 0:
                per_dim_offsets.append(None)
            else:
                # Something is wrong, this should never happen, or it is not a slice
                return

        # Now modify the uses
        original_uses = [*op.results[0].uses]
        for idx, user in enumerate(original_uses):
            user = user.operation
            new_offsets = list()
            original_indices = user.indices
            for off_op, off_new in zip(original_indices, per_dim_offsets):
                if off_new is not None:
                    # If this was a constant
                    new_offset = arith.Addi(operand1=off_op, operand2=off_new)
                    rewriter.insert_op(new_offset, InsertPoint.before(user))
                    new_offsets.append(new_offset)
                else:
                    new_offsets.append(off_op)

            if isinstance(user, memref.Load):
                # Merge the idx lists
                # Add the new indices

                new_op = memref.Load.get(ref=op.operands[0], indices=new_offsets)
                rewriter.replace_op(user, new_op, [new_op.results[0]])
            elif isinstance(user, memref.Store):
                new_op = memref.Store.get(
                    ref=op.operands[0], indices=new_offsets, value=user.operands[0]
                )
                rewriter.replace_op(user, new_op, [])
        rewriter.erase_op(op)


class LoadToSubscript(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: memref.Load,
        rewriter: PatternRewriter,
    ):
        memory_loc = emitc.Subscript(operand=op.operands[0], indices=op.operands[1:])
        rewriter.replace_matched_op([memory_loc], [memory_loc.results[0]])


class FixFuncBlocks(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter):
        if op.args[0].type == op.function_type.inputs.data[0]:
            return
        op.update_function_type()
        rewriter.handle_operation_modification(op)


class ConvertMemrefToEmitcPass(ModulePass):
    def apply(self, ctx: MLContext, op: ModuleOp):
        # PatternRewriteWalker(
        #    GreedyRewritePatternApplier(
        #        [
        #            MergeSubviewSlices(),  # Needed, as subviews are not supported in emitc
        #        ]
        #    )
        # ).rewrite_module(op)
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    # MergeSubviewSlices(),  # Needed, as subviews are not supported in emitc
                    AllocToGlobal(),
                    StoreToAssign(),
                    LoadToSubscript(),
                    GlobalToGlobal(),
                    GetGlobalToGetGlobal(),
                ]
            )
        ).rewrite_module(op)

        # Convert the types and fix the blocks
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    MemrefToArrayType(),
                    # FixFuncBlocks()
                ]
            )
        ).rewrite_module(op)
