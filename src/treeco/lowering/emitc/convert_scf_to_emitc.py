"""
Unusable as emitc.for allows only a single block inside, which is not the case when
using scf.while or any cf.cond_br. We do scf - > cf -> emitc (although the code is full of goto)
#TODO : Keep checking if emitc.while is implemented and use it
An alternative could be an additional function:
for ....:
    for ...:
        tree_visit() <-- This includes the while()
"""

from typing import Any, List
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl.ir import Region, Block
from xdsl.context import MLContext
from treeco.model.ensemble import Ensemble
from xdsl.dialects.builtin import (
    IntegerType,
    StringAttr,
    IntegerAttr,
    ModuleOp,
    FloatAttr,
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
from xdsl.dialects import func, scf, arith, memref, affine, printf
from treeco.utils import tensor_to_memref, I64_MIN, convert_np_to_arrayattr
from treeco.dialects import emitc


class ForToFor(RewritePattern):
    """
    Converts only easy scf.for loop, the rest is handled by converting to CF
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.For, rewriter: PatternRewriter):
        if len(op.iter_args) != 0:
            return
        if len(op.results) != 0:
            return
        noop = emitc.For(
            lowerBound=op.lb,
            upperBound=op.ub,
            step=op.step,
            region=op.detach_region(op.regions[0]),
        )
        new_yield = emitc.Yield()
        rewriter.replace_op(noop.body.blocks[0].last_op, [new_yield], [])

        rewriter.replace_matched_op(noop, [])
        # noop.last_op = op


# TODO: Wait for a better solution, 1) use cf.branch 2) wait for a emitc.While to be implemented
class WhileToFor(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.While, rewriter: PatternRewriter):
        # Working only with while - > do
        if not isinstance(op.regions[0].ops.last, scf.Condition):
            return
        if len(op.results) > 1:
            return
        if len(op.arguments) != 1:
            return

        output_arg = op.arguments
        # This may be unnecessary
        iteration_variable = emitc.Variable(
            value=IntegerAttr(0, IndexType()),
        )
        # Assign the initial value to this one

        lb = emitc.Literal(StringAttr(" "), emitc.OpaqueType(" "))
        ub = emitc.Literal(StringAttr("1"), emitc.OpaqueType(" "))
        step = emitc.Literal(StringAttr(" "), emitc.OpaqueType(" "))
        assigner = emitc.Assign(var=iteration_variable, value=op.arguments[0])
        rewriter.insert_op(
            [iteration_variable, lb, ub, step, assigner], InsertPoint.before(op)
        )

        # Region modification
        before_region = op.detach_region(op.regions[0])
        before_region.blocks[0].args[0].replace_by(assigner.operands[0])
        rewriter.modify_block_argument_type(
            before_region.blocks[0].args[0], lb.results[0].type
        )
        after_region = op.detach_region(op.regions[0])
        after_region.blocks[0].args[0].replace_by(assigner.operands[0])
        rewriter.modify_block_argument_type(
            after_region.blocks[0].args[0], lb.results[0].type
        )
        new_for = emitc.For(
            lowerBound=lb, upperBound=ub, step=step, region=after_region
        )
        # The first operation of the infinite loop is the condition evaluation
        # Get the last two operations of the before region

        op_condition = before_region.ops.last
        op_comparison = before_region.ops.last.operands[0].owner

        if not isinstance(op_comparison, arith.Cmpi) and not isinstance(
            op_condition, emitc.Cmp
        ):
            return
        if op_comparison.properties["predicate"].value.data != 1:
            return
        new_cmp = arith.Cmpi(
            operand1=op_comparison.operands[0],
            operand2=op_comparison.operands[1],
            arg="eq",
        )
        rewriter.replace_op(op_comparison, [new_cmp], [new_cmp.results[0]])
        # The cmpi has been modified, now we modify the regions
        if_condition = emitc.If(
            condition=new_cmp,
            true_region=Region([Block([emitc.Verbatim("break"), emitc.Yield()])]),
        )
        rewriter.replace_op(op_condition, [if_condition], [])

        rewriter.inline_block(
            before_region.blocks[0],
            InsertPoint.at_start(after_region.blocks[0]),
        )

        new_yield = emitc.Yield()
        previous_yield = new_for.body.blocks[0].last_op
        # This is assigned to the initial value
        out_assign = emitc.Assign(
            var=iteration_variable, value=previous_yield.operands[0]
        )

        rewriter.replace_op(previous_yield, [out_assign, new_yield], [])
        rewriter.replace_matched_op(new_for, [iteration_variable.results[0]])
        # noop.last_op = op


class ConvertScfToEmitcPass(ModulePass):
    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(ForToFor()).rewrite_module(op)
