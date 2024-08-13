from xdsl.dialects.builtin import (
    ModuleOp,
    IntegerAttr,
    i64,
    StringAttr,
)
from xdsl.context import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from treeco.dialects import onnxml, crown, treeco
from xdsl.dialects import func
from treeco import utils as utils


def convert_attributes(
    op: onnxml.TreeEnsembleClassifier | onnxml.TreeEnsembleRegressor,
):
    ensemble_attr = op.attributes
    new_attr = {}

    # Unify naming
    for k, v in ensemble_attr.items():
        if "class" in k:
            new_attr[k.replace("class", "targets")] = v
        else:
            new_attr[k] = v

    new_attr["n_features"] = IntegerAttr(op.buffer_in.type.shape.data[-1].data, i64)
    new_attr["aggregate_mode"] = StringAttr("SUM")
    # new_attr["post_transform"] = StringAttr("NONE")
    return new_attr


class ConvertEnsemble(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: onnxml.TreeEnsembleClassifier | onnxml.TreeEnsembleRegressor,
        rewriter: PatternRewriter,
    ):

        # Skip if this is not inside a function (i.e. main/inference)
        if not isinstance(op.parent_op(), func.FuncOp):
            return
        # Convert the attributes
        ensemble_data = convert_attributes(op)
        # and generate ours...
        attr = treeco.TreeEnsembleAttr(**ensemble_data)

        new_op = crown.TreeEnsembleOp(
            buffer_in=op.operands[0],
            buffer_out=op.operands[1],
            ensemble_attr=attr,
        )
        rewriter.replace_matched_op([new_op], [])


class ConvertOnnxmlToCrownPass(ModulePass):
    name = "convert-onnxml-to-crown"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(ConvertEnsemble()).rewrite_module(op)
