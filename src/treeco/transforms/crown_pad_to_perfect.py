""" 
Pad the trees to ensure all leaves are at a minimum depth or directly make the trees
perfect.
"""

from xdsl.context import MLContext
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from treeco.dialects import crown, treeco
from treeco.model.ensemble import Ensemble


class CrownPadTreesPerfect(RewritePattern):
    """
    Pad each tree in the Crown Ensemble to perfect.
    Note: It uses the maximum depth of the ensemble.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: crown.TreeEnsembleOp, rewriter: PatternRewriter):
        ensemble = Ensemble.parse_attr(op.ensemble)
        if ensemble.is_perfect():
            return

        # TODO: Add an option for auto or int
        ensemble.pad_to_perfect("auto")
        ensemble_attr = treeco.TreeEnsembleAttr(**ensemble.to_attr())
        pop = crown.TreeEnsembleOp(
            buffer_in=op.operands[0],
            ensemble_attr=ensemble_attr,
            buffer_out=op.operands[1],
        )
        rewriter.replace_matched_op(pop, [])


class CrownPadTreesPerfectPass(ModulePass):
    def apply(
        self,
        ctx: MLContext,
        op: ModuleOp,
    ) -> None:
        PatternRewriteWalker(
            CrownPadTreesPerfect(),
        ).rewrite_module(op)
