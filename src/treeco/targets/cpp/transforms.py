from treeco.lowering.emitc import (
    ConvertMemrefToEmitcPass,
    ConvertArithToEmitcPass,
    ConvertPrintfToEmitcPass,
)
from xdsl.dialects.builtin import ModuleOp
from xdsl.context import MLContext
from treeco.utils import dump_ir


def target_transform_and_dump(
    output_path: str,
    module_op: ModuleOp,
    ctx: MLContext,
) -> ModuleOp:
    ConvertMemrefToEmitcPass().apply(ctx=ctx, op=module_op)
    ConvertArithToEmitcPass().apply(ctx=ctx, op=module_op)
    ConvertPrintfToEmitcPass().apply(ctx=ctx, op=module_op)
    dump_ir(
        target_path=output_path,
        module_op=module_op,
        run_mlir_opt_passes=[
            "--canonicalize",
            "--cse",
        ],
    )
    return module_op
