from xdsl.context import MLContext
from xdsl.dialects.builtin import IntegerType, MemRefType, ModuleOp, Signedness
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from treeco.dialects import crown, treeco
from treeco.model.ensemble import Ensemble
from treeco.transforms.func_legalize import UpdateSignatureFuncOp


class QuantizeInput(RewritePattern):
    def __init__(self, precision: int, min_val: float, max_val: float, **kwargs):
        super().__init__(**kwargs)
        self.precision = precision
        self.min_val = min_val
        self.max_val = max_val

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: crown.TreeEnsembleOp, rewriter: PatternRewriter):
        if op.operands[0].type.element_type == IntegerType(
            self.precision, signedness=Signedness.UNSIGNED
        ):
            return
        idata = op.operands[0]
        new_etype = IntegerType(self.precision, signedness=Signedness.UNSIGNED)
        ensemble = Ensemble.parse_attr(op.ensemble)
        ensemble.quantize_thresholds(
            method="quantize",
            precision=self.precision,
            min_val=self.min_val,
            max_val=self.max_val,
        )
        ensemble_attr = treeco.TreeEnsembleAttr(**ensemble.to_attr())
        # TODO : Check if operand is actually a block argument
        rewriter.modify_block_argument_type(
            idata,
            MemRefType(
                element_type=new_etype,
                shape=idata.type.get_shape(),
            ),
        )

        qop = crown.TreeEnsembleOp(
            buffer_in=op.operands[0],
            ensemble_attr=ensemble_attr,
            buffer_out=op.operands[1],
        )
        rewriter.replace_matched_op(qop, [])


class RoundInput(RewritePattern):
    def __init__(self, precision: int, **kwargs):
        super().__init__(**kwargs)
        self.precision = precision

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: crown.TreeEnsembleOp, rewriter: PatternRewriter):
        if op.operands[0].type.element_type == IntegerType(
            self.precision, signedness=Signedness.UNSIGNED
        ):
            return
        idata = op.operands[0]
        new_etype = IntegerType(self.precision, signedness=Signedness.UNSIGNED)
        ensemble = Ensemble.parse_attr(op.ensemble)
        ensemble.quantize_thresholds(
            "round",
            precision=self.precision,
        )
        ensemble_attr = treeco.TreeEnsembleAttr(**ensemble.to_attr())
        # TODO : Check if operand is actually a block argument
        rewriter.modify_block_argument_type(
            idata,
            MemRefType(
                element_type=new_etype,
                shape=idata.type.get_shape(),
            ),
        )

        qop = crown.TreeEnsembleOp(
            buffer_in=op.operands[0],
            ensemble_attr=ensemble_attr,
            buffer_out=op.operands[1],
        )
        rewriter.replace_matched_op(qop, [])


class QuantizeLeaves(RewritePattern):
    def __init__(self, precision: int, **kwargs):
        super().__init__(**kwargs)
        self.precision = precision

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: crown.TreeEnsembleOp, rewriter: PatternRewriter):
        if isinstance(op.operands[1].type.element_type, IntegerType):
            return
        new_etype = IntegerType(self.precision, signedness=Signedness.UNSIGNED)
        ensemble = Ensemble.parse_attr(op.ensemble)
        # Exclude non-SUM ensembles (i.e. vote)
        if ensemble.aggregate_mode != Ensemble.AGGREGATE_MODE_SUM:
            return

        ensemble_output_min, ensemble_output_max = ensemble.output_range
        ensemble.quantize_leaves(
            precision=self.precision,
            min_val=ensemble_output_min,
            max_val=ensemble_output_max,
        )

        # Regenerate the block argument
        rewriter.modify_block_argument_type(
            op.operands[1],
            MemRefType(
                element_type=new_etype,
                shape=op.operands[1].type.get_shape(),
            ),
        )
        odata = op.operands[1]
        ensemble_attr = treeco.TreeEnsembleAttr(**ensemble.to_attr())
        # TODO : Check if operand is actually a block argument
        qop = crown.TreeEnsembleOp(
            buffer_in=op.operands[0],
            ensemble_attr=ensemble_attr,
            buffer_out=op.operands[1],
        )
        rewriter.replace_matched_op(qop, [])


class CrownQuantizeInputPass(ModulePass):
    def apply(
        self,
        ctx: MLContext,
        op: ModuleOp,
        precision: int,
        min_val: float,
        max_val: float,
    ) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    QuantizeInput(
                        precision=precision, min_val=min_val, max_val=max_val
                    ),
                ]
            )
        ).rewrite_module(op)


class CrownRoundInputPass(ModulePass):
    def apply(
        self,
        ctx: MLContext,
        op: ModuleOp,
        precision: int,
    ) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    RoundInput(
                        precision=precision,
                    ),
                ]
            )
        ).rewrite_module(op)
        PatternRewriteWalker(
            UpdateSignatureFuncOp(),
        ).rewrite_module(op)


class CrownQuantizeLeavesPass(ModulePass):
    def apply(
        self,
        ctx: MLContext,
        op: ModuleOp,
        precision: int,
    ) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    QuantizeLeaves(precision=precision),
                ]
            )
        ).rewrite_module(op)

        PatternRewriteWalker(
            UpdateSignatureFuncOp(),
        ).rewrite_module(op)
