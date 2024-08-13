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


class CrownPruneTrees(RewritePattern):
    def __init__(self, multiple_of_n_trees: int, **kwargs):
        super().__init__(**kwargs)
        self.multiple_of_n_trees = multiple_of_n_trees

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: crown.TreeEnsembleOp, rewriter: PatternRewriter):
        ensemble = Ensemble.parse_attr(op.ensemble)
        if ensemble.n_trees % self.multiple_of_n_trees == 0:
            return

        n_trees = (
            ensemble.n_trees // self.multiple_of_n_trees
        ) * self.multiple_of_n_trees
        ensemble.prune_trees(n_trees=n_trees)
        ensemble_attr = treeco.TreeEnsembleAttr(**ensemble.to_attr())
        pop = crown.TreeEnsembleOp(
            buffer_in=op.operands[0],
            ensemble_attr=ensemble_attr,
            buffer_out=op.operands[1],
        )
        rewriter.replace_matched_op(pop, [])


class CrownPruneTreesPass(ModulePass):
    def apply(
        self,
        ctx: MLContext,
        op: ModuleOp,
        multiple_of_n_trees: int,
    ) -> None:
        PatternRewriteWalker(
            CrownPruneTrees(multiple_of_n_trees=multiple_of_n_trees),
        ).rewrite_module(op)
