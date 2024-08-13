""" 
This pass is completely left to mlir-opt!
"""

from treeco.lowering.mlir_opt import mlir_opt_pass


def convert_scf_to_cf_pass(module_op, ctx, allow_unregistered_dialects=True):
    mlir_opt_pass(
        module_op=module_op,
        ctx=ctx,
        additional_args=[
            "--convert-scf-to-cf",
            # "--expand-strided-metadata", # Reinterpret_cast
            "--canonicalize",
            "-cse",
        ],
        allow_unregistered_dialects=allow_unregistered_dialects,
    )
