from xdsl.rewriter import InsertPoint
from treeco.lowering.emitc.convert_printf_to_emitc import ConvertPrintfToEmitcPass

import numpy as np
from xdsl.dialects.builtin import (
    ModuleOp,
    StringAttr,
    ModuleOp,
    UnitAttr,
)

from xdsl.dialects import func, printf
from xdsl.context import MLContext
from xdsl.ir import Block, Region
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from treeco.dialects import emitc
from treeco.utils import convert_np_to_tensor


class AddEmitC(RewritePattern):
    has_main = False

    def get_functions(self, op: ModuleOp):
        functs = {}
        for o in op.walk():
            if isinstance(o, func.FuncOp):
                functs[o.sym_name.data] = o
        return functs

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ModuleOp, rewriter: PatternRewriter):
        functs = self.get_functions(op)
        if "main" in functs:
            return
        # input_data
        value_shape = functs["inference"].args[0].type.get_shape()
        element_type = functs["inference"].args[0].type.get_element_type()

        input_data_global = emitc.Global(
            name=StringAttr("input_data"),
            type_=emitc.ArrayType(
                element_type=functs["inference"].args[0].type.element_type,
                shape=functs["inference"].args[0].type.get_shape(),
            ),
            initial_value=convert_np_to_tensor(np.zeros(value_shape)),
        )
        output_data_global = emitc.Global(
            name=StringAttr("output_data"),
            type_=emitc.ArrayType(
                element_type=functs["inference"].args[1].type.element_type,
                shape=functs["inference"].args[1].type.get_shape(),
            ),
            initial_value=convert_np_to_tensor(
                np.zeros(functs["inference"].args[1].type.get_shape())
            ),
        )
        rewriter.insert_op(
            [input_data_global, output_data_global],
            InsertPoint.at_start(op.body.block),
        )

        blocco = Block(arg_types=())
        main = func.FuncOp.from_region(
            name="main",
            input_types=[],
            return_types=[],
            region=Region([blocco]),
            visibility="public",
        )

        input_data = emitc.GetGlobal(
            name="input_data",
            return_type=input_data_global.type_,
        )
        output_data = emitc.GetGlobal(
            name="output_data",
            return_type=output_data_global.type_,
        )
        fc = func.Call(
            callee="inference",
            arguments=[input_data, output_data],
            return_types=[],
        )

        pr = printf.PrintFormatOp("{}\n", output_data)

        r = func.Return()

        main.attributes["llvm.emit_c_interface"] = UnitAttr()
        rewriter.insert_op(
            [input_data, output_data, fc, pr, r],
            InsertPoint.at_start(blocco),
        )
        rewriter.insert_op(main, InsertPoint.after(functs["inference"]))


class AddMainPass(ModulePass):
    name = "add-main-emitc"

    def apply(self, ctx: MLContext, op: ModuleOp):
        PatternRewriteWalker(AddEmitC()).rewrite_module(op)
        ConvertPrintfToEmitcPass().apply(ctx, op)
