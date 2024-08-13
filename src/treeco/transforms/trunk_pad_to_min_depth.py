from xdsl.context import MLContext
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from treeco.dialects import treeco, trunk
from treeco.model.ensemble import Ensemble


class TrunkPadTrees(RewritePattern):
    def __init__(self, min_depth_ensemble: int, **kwargs):
        """
        min_depth_ensemble: int
            The minimum depth of all leaves in the ensemble.
        """
        super().__init__(**kwargs)
        assert min_depth_ensemble >= 1, "Invalid minimum depth selected"
        self.min_depth_ensemble = min_depth_ensemble

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: trunk.TreeEnsembleConstantOp, rewriter: PatternRewriter
    ):
        ensemble = Ensemble.parse_attr(op.ensemble)
        if ensemble.min_depth_leaves >= self.min_depth_ensemble:
            return

        ensemble.pad_to_depth(min_depth=self.min_depth_ensemble)
        ensemble_attr = treeco.TreeEnsembleAttr(**ensemble.to_attr())
        pop = trunk.TreeEnsembleConstantOp(
            buffer_in=op.operands[0],
            ensemble_attr=ensemble_attr,
            buffer_out=op.operands[1],
        )
        rewriter.replace_matched_op(pop, [])


class TrunkPadTreesPass(ModulePass):
    def apply(
        self,
        ctx: MLContext,
        op: ModuleOp,
        min_depth_ensemble: int,
    ) -> None:
        PatternRewriteWalker(
            TrunkPadTrees(min_depth_ensemble=min_depth_ensemble),
        ).rewrite_module(op)
