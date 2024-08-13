from treeco.utils import dump_ir


def dump_to_llvm(target_path, module_op, ctx):
    """
    Writes the module_op to the target_path in LLVM dialect.
    Note: The module_op cannot not re-parsed back by XDSL at this step.
    """
    # TODO : There is definitely a better way to do this
    dump_ir(
        target_path=target_path,
        module_op=module_op,
        run_mlir_opt_passes=[
            "--expand-strided-metadata",  # Removes the leftover memref.subview
            "--convert-func-to-llvm",  # Converts func AND blocks to i64! This must be run before the cf conversion!!
            "--convert-arith-to-llvm",  # Converts arith to llvm
            "--convert-cf-to-llvm",  # Converts control flow to llvm
            "--finalize-memref-to-llvm",  # Finalizes memref to llvm
            "--lower-affine",  # Somewhere popped up an affine.apply....
            "--convert-arith-to-llvm",  # Converts the affine->arith to llvm
            "--reconcile-unrealized-casts",  # Reconciles unrealized casts
            "--canonicalize",  # Cause why not?
            "--cse",  # Same as above
        ],
    )
