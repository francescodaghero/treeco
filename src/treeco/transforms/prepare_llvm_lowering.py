from xdsl.dialects.builtin import (
    ModuleOp,
    UnitAttr,
)
from xdsl.dialects import func
from xdsl.context import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class CCompatibleFunc(RewritePattern):
    """
    Adds the emic_c_interface attribute for C-like function definitions.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter):
        if op.attributes.get("llvm.emit_c_interface"):
            return
        op.attributes["llvm.emit_c_interface"] = UnitAttr()


class PrepareLLVMLoweringPass(ModulePass):
    name = "prepare-llvm-lowering"

    def apply(self, ctx: MLContext, op: ModuleOp):
        PatternRewriteWalker(
            CCompatibleFunc(),
        ).rewrite_module(op)
