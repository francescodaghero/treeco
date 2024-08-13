""" 
Transforms handling the data types, last step of conversion from Treeco's IRs
to MLIR's in-tree IRs.
"""

from xdsl.passes import ModulePass
from typing import cast
from xdsl.context import MLContext
from treeco.dialects import treeco
from xdsl.dialects import builtin

from treeco.dialects.extended import ml_program
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
    TypeConversionPattern,
    attr_type_rewrite_pattern,
)


class SignToSignLess(TypeConversionPattern):
    @attr_type_rewrite_pattern
    def convert_type(self, typ: builtin.IntegerType):
        if typ.signedness == builtin.Signedness.SIGNLESS:
            return
        else:
            return builtin.IntegerType(typ.width, builtin.Signedness.SIGNLESS)


class LowerCast(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: treeco.Cast,
        rewriter: PatternRewriter,
    ):
        if (
            isinstance(op.operands[0].type, builtin.IndexType)
            or op.operands[0].type == op.results[0].type
        ):
            rewriter.replace_matched_op([], [op.operands[0]])


class LowerCastSign(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: treeco.CastSignOp,
        rewriter: PatternRewriter,
    ):
        if op.operands[0].type == op.results[0].type:
            rewriter.replace_matched_op([], [op.operands[0]])


class RemoveLeftoverNodeTypes(TypeConversionPattern):
    @attr_type_rewrite_pattern
    def convert_type(self, typ: treeco.NodeType | treeco.LeafType):
        return builtin.IndexType()


class RemoveUnusedGlobals(RewritePattern):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.history = set()

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: ml_program.Global,
        rewriter: PatternRewriter,
    ):
        if op.sym_name.data in self.history:
            return

        for operation in op.get_toplevel_object().walk():
            if isinstance(operation, ml_program.GlobalLoadConstant):
                operation = cast(ml_program.GlobalLoadConstant, operation)
                self.history.add(operation.global_attr.string_value())

        if op.sym_name.data not in self.history:
            rewriter.replace_matched_op([], [])


class LowerTreecoPass(ModulePass):
    name = "lower-treeco"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(LowerCast(), apply_recursively=True).rewrite_module(op)
        # This fixes the while loop not changing type
        # TODO : Understand why this is necessary
        PatternRewriteWalker(RemoveLeftoverNodeTypes()).rewrite_module(op)
        PatternRewriteWalker(LowerCast(), apply_recursively=True).rewrite_module(op)
        PatternRewriteWalker(SignToSignLess(recursive=True)).rewrite_module(op)
        PatternRewriteWalker(LowerCastSign()).rewrite_module(op)
        PatternRewriteWalker(RemoveUnusedGlobals()).rewrite_module(op)
        op.verify()
