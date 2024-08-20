from xdsl.context import MLContext
from xdsl.dialects.builtin import ModuleOp
from treeco.lowering import convert_scf_to_cf_pass
from xdsl.transforms.printf_to_llvm import PrintfToLLVM
from treeco.utils import dump_ir


def dump_to_llvm(
    target_path: str,
    module_op: ModuleOp,
    ctx: MLContext,
):
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


def target_transform_and_dump(
    output_path: str,
    module_op: ModuleOp,
    ctx: MLContext,
) -> ModuleOp:
    """
    Writes to output_path the IR lowered to LLVM IR.
    Returns the last valid step in xdsl (i.e. before LLVM IR lowering).

    Parameters
    ----------
    output_path : str
        The output file name where the LLVM IR is written.
    module_op : ModuleOp
        The program IR
    ctx : MLContext
        The program context

    Returns
    -------
    ModuleOp
        IR before LLVM IR conversion
    """

    # Currently not used, we call directly the inference function
    # PrepareLLVMLoweringPass().apply(ctx=ctx, op=module_op)

    # Lower printf if still present
    PrintfToLLVM().apply(ctx=ctx, op=module_op)
    # Lower scf
    convert_scf_to_cf_pass(module_op, ctx)
    # Convert to LLVM IR
    dump_to_llvm(output_path, module_op=module_op, ctx=ctx)

    return module_op
