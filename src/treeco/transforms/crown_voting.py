import numpy as np
from xdsl.context import MLContext
from xdsl.dialects.builtin import IntegerType, MemRefType, ModuleOp, Signedness
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from treeco.dialects import crown, treeco
from treeco.model.ensemble import Ensemble
from .func_legalize import UpdateSignatureFuncOp


class ConvertToVoting(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: crown.TreeEnsembleOp, rewriter: PatternRewriter):
        if op.attributes["ensemble"].aggregate_mode.data == Ensemble.AGGREGATE_MODE_VOTE:
            return

        ensemble: Ensemble = Ensemble.parse_attr(op.ensemble)
        ensemble.logits_to_vote()

        attr = ensemble.to_attr()
        new_nptype = np.min_scalar_type(ensemble.n_targets)
        new_dtype = IntegerType(new_nptype.itemsize * 8, signedness=Signedness.UNSIGNED)
        new_shape = op.operands[1].type.get_shape()
        new_shape = (
            new_shape[0],
            ensemble.n_targets,
        )  # Changes for binary classification
        new_buffer_out = MemRefType(element_type=new_dtype, shape=new_shape)
        # This changes the block type
        rewriter.modify_block_argument_type(op.operands[1], new_buffer_out)

        # TODO: CHeck if this is still needed
        new_op = crown.TreeEnsembleOp(
            buffer_in=op.operands[0],
            buffer_out=op.operands[1],
            ensemble_attr=treeco.TreeEnsembleAttr(**attr),
        )
        rewriter.replace_matched_op(
            [new_op],
            [],
        )


class CrownConvertToVotingClassifierPass(ModulePass):
    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(
            ConvertToVoting(),
        ).rewrite_module(op)
        PatternRewriteWalker(
            UpdateSignatureFuncOp(),
        ).rewrite_module(op)
        op.verify()
