""" 
Removes trees from the ensemble. Useful for iterative +  parallelization
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


class CrownPadTrees(RewritePattern):
    def __init__(self, min_depth_ensemble: int, **kwargs):
        """
        min_depth_ensemble: int
            The minimum depth of all leaves in the ensemble. -1 means pad to perfect
        """
        super().__init__(**kwargs)
        assert min_depth_ensemble >= -1, "Invalid minimum depth selected"
        self.min_depth_ensemble = min_depth_ensemble

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: crown.TreeEnsembleOp, rewriter: PatternRewriter):
        ensemble = Ensemble.parse_attr(op.ensemble)
        if ensemble.min_depth_leaves >= self.min_depth_ensemble:
            return

        ensemble.pad_to_depth(min_depth=self.min_depth_ensemble)
        ensemble_attr = treeco.TreeEnsembleAttr(**ensemble.to_attr())
        pop = crown.TreeEnsembleOp(
            buffer_in=op.operands[0],
            ensemble_attr=ensemble_attr,
            buffer_out=op.operands[1],
        )
        rewriter.replace_matched_op(pop, [])


class CrownPadTreesPerfect(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: crown.TreeEnsembleOp, rewriter: PatternRewriter):
        ensemble = Ensemble.parse_attr(op.ensemble)
        if ensemble.min_depth_leaves >= self.min_depth_ensemble:
            return

        ensemble.pad_to_perfect()
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
