""" 
Utility transforms, used to legalize transforms to the block in the funcOp that are 
not transfered to the funcOp signature
"""

from xdsl.dialects import func
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)


class UpdateSignatureFuncOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter):
        update_signature = False
        for operand_block, operand_signature in zip(
            op.args, op.function_type.inputs.data
        ):
            if operand_block.type != operand_signature:
                update_signature = True
        for result_block, result_signature in zip(
            op.results, op.function_type.outputs.data
        ):
            if result_block.type != result_signature:
                update_signature = True
        if not update_signature:
            return
        op.update_function_type()
        rewriter.handle_operation_modification(op)
