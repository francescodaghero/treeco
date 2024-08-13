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
from treeco.targets.cpp.add_entry_point import AddMainPass

import numpy as np


class CmpiToCmp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Cmpi, rewriter: PatternRewriter):
        mapped = {
            0: "eq",
            1: "ne",
            2: "lt",
            3: "le",
            4: "gt",
            5: "ge",
            6: "lt",
            7: "le",
            8: "gt",
            9: "ge",
        }

        new_op = emitc.Cmp(
            lhs=op.lhs,
            rhs=op.rhs,
            predicate=mapped[op.predicate.value.data],
        )
        rewriter.replace_matched_op(new_op, [new_op.results[0]])


class CmpfToCmp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Cmpf, rewriter: PatternRewriter):
        predicate = op.predicate.value.data
        cmpf_comparison_operations = {
            1: "eq",
            2: "gt",
            3: "ge",
            4: "lt",
            5: "le",
            6: "ne",
            8: "eq",
            9: "gt",
            10: "ge",
            11: "lt",
            12: "le",
            13: "ne",
        }
        if predicate not in cmpf_comparison_operations:
            return
        new_op = emitc.Cmp(
            lhs=op.lhs,
            rhs=op.rhs,
            predicate=cmpf_comparison_operations[op.predicate.value.data],
        )

        rewriter.replace_matched_op(new_op, [new_op.results[0]])


class AddToAdd(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Addi | arith.Addf, rewriter: PatternRewriter):
        new_op = emitc.Add(
            lhs=op.operands[0],
            rhs=op.operands[1],
        )
        rewriter.replace_matched_op(new_op, [new_op.results[0]])


class MulToMul(RewritePattern):
    # TODO : Overflows?
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Mulf | arith.Muli, rewriter: PatternRewriter):
        new_op = emitc.Mul(
            lhs=op.operands[0],
            rhs=op.operands[1],
        )
        rewriter.replace_matched_op(new_op, [new_op.results[0]])


class SubToSub(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Subi | arith.Subf, rewriter: PatternRewriter):
        new_op = emitc.Sub(
            lhs=op.operands[0],
            rhs=op.operands[1],
        )
        rewriter.replace_matched_op(new_op, [new_op.results[0]])


class ExtUIToCast(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.ExtUIOp, rewriter: PatternRewriter):
        new_op = emitc.Cast(
            operand=op.operands[0],
            result=IndexType(),
        )
        rewriter.replace_matched_op(new_op, [new_op.results[0]])


class IndexCastToCast(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.IndexCastOp, rewriter: PatternRewriter):
        new_op = emitc.Cast(
            operand=op.operands[0],
            result=IndexType(),
        )
        rewriter.replace_matched_op(new_op, [new_op.results[0]])


class ConstantToConstant(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Constant, rewriter: PatternRewriter):
        new_op = emitc.Constant(
            value=op.value,
        )
        rewriter.replace_matched_op(new_op, [new_op.results[0]])


class ConvertArithToEmitcPass(ModulePass):
    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    CmpiToCmp(),
                    CmpfToCmp(),
                    SubToSub(),
                    MulToMul(),
                    AddToAdd(),
                    ExtUIToCast(),
                    IndexCastToCast(),
                    ConstantToConstant(),
                ]
            )
        ).rewrite_module(op)
