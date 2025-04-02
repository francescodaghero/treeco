from .mlir_opt import mlir_opt_pass
from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp


def bufferize_pass(module_op: ModuleOp, ctx: Context):
    mlir_opt_pass(
        module_op,
        ctx,
        additional_args=[
            "--one-shot-bufferize=allow-unknown-ops",
            "--buffer-hoisting",
            "--buffer-loop-hoisting",
            "--buffer-results-to-out-params",
            "--drop-equivalent-buffer-results",
            "--promote-buffers-to-stack",
            "--buffer-deallocation-pipeline",
            "--cse",
            "--canonicalize",
        ],
        # Generally for the PrintOp
        allow_unregistered_dialects=True,
    )
