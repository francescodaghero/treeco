import re
from typing import Any

from xdsl.builder import Builder
from xdsl.context import MLContext
from xdsl.dialects import printf
from xdsl.dialects.builtin import (
    AnyFloat,
    ArrayAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    ModuleOp,
)
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from treeco.dialects import emitc


class PrintfToEmitc(RewritePattern):
    def convert_string_python_to_c(self, op: printf.PrintFormatOp):
        pattern = re.compile(r"\{([^{}]*)\}")
        stringa = op.format_str.data
        vals = op.format_vals

        base_str = '"'
        prev_start = 0
        for idx, match in enumerate(pattern.finditer(string=stringa)):
            base_str += stringa[prev_start : match.start()]
            # TODO : Implement here the formatting rules, e.g. %.3f
            if (
                isinstance(vals[idx].type, IntegerType)
                or isinstance(vals[idx].type, IndexType)
                or (
                    isinstance(vals[idx].type, emitc.ArrayType)
                    and isinstance(vals[idx].type.element_type, IntegerType)
                )
            ):
                base_str += "%d"
            elif isinstance(vals[idx].type, AnyFloat) or (
                isinstance(vals[idx].type, emitc.ArrayType)
                and isinstance(vals[idx].type.element_type, AnyFloat)
            ):
                base_str += "%f"
            else:
                return None
            prev_start = match.end() + 1
        base_str += stringa[prev_start:]
        base_str += '"\n'

        return base_str

    def generate_nested_printf(self, op, rewriter):
        # Support limited to 2D array
        output_len = op.format_vals[0].type.get_shape()[-1]
        batch_size = op.format_vals[0].type.get_shape()[-2]

        lb_const = emitc.Constant(value=IntegerAttr(0, IndexType()))
        step_const = emitc.Constant(value=IntegerAttr(1, IndexType()))
        ub_const_targets = emitc.Constant(value=IntegerAttr(output_len, IndexType()))
        ub_const_batch = emitc.Constant(value=IntegerAttr(batch_size, IndexType()))

        @Builder.implicit_region((IndexType(),))
        def outer_loop_body(args_outer: tuple[Any, ...]) -> None:
            (batch_idx,) = args_outer

            @Builder.implicit_region((IndexType(),))
            def inner_loop_body(args_inner: tuple[Any, ...]) -> None:
                (target_idx,) = args_inner
                sbs = emitc.Subscript(
                    operand=op.format_vals[0], indices=[batch_idx, target_idx]
                )
                pr = printf.PrintFormatOp("{}\n", sbs)
                emitc.Yield()

            inner_for = emitc.For(
                lowerBound=lb_const,
                upperBound=ub_const_targets,
                step=step_const,
                region=inner_loop_body,
            )
            emitc.Yield()

        outer_for = emitc.For(
            lowerBound=lb_const,
            upperBound=ub_const_batch,
            region=outer_loop_body,
            step=step_const,
        )

        rewriter.replace_matched_op(
            [lb_const, step_const, ub_const_batch, ub_const_targets, outer_for], []
        )

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: printf.PrintFormatOp, rewriter: PatternRewriter):

        formatted_str = self.convert_string_python_to_c(op)
        if formatted_str is None:
            return
        if (
            any(isinstance(o.type, emitc.ArrayType) for o in op.format_vals)
            and len(op.format_vals) > 1
        ):
            # Support limited to printing an array in a printf
            return
        args = [
            emitc.OpaqueAttr(value=formatted_str),
        ]
        args += [IntegerAttr(i, IndexType()) for i in range(len(op.format_vals))]

        if any(isinstance(o.type, emitc.ArrayType) for o in op.format_vals):
            for op_val in op.format_vals:
                if isinstance(op_val.type, emitc.ArrayType):
                    self.generate_nested_printf(op, rewriter)

        else:
            new_op = emitc.CallOpaque(
                callee="printf",
                args=ArrayAttr(args),
                operands_=list(op.format_vals),
                results_=[],
            )

            rewriter.replace_matched_op(new_op, [])


class ConvertPrintfToEmitcPass(ModulePass):
    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(PrintfToEmitc()).rewrite_module(op)
