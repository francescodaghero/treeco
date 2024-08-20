import time

import numpy as np
from xdsl.context import MLContext
from xdsl.dialects import arith, func, memref
from xdsl.dialects.builtin import IntAttr, MemRefType, ModuleOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    TypeConversionPattern,
    attr_type_rewrite_pattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint

from treeco.dialects import emitc
from treeco.utils import convert_np_to_arrayattr


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
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
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
                ]
            )
        ).rewrite_module(op)
