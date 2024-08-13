import shutil
from .utils import quantize
from .xdsl_utils import (
    find_op_in_operands_chain,
    find_op_in_results_chain,
    find_operation_in_module,
)
from xdsl.utils.exceptions import DiagnosticException
import subprocess
from io import StringIO
from xdsl.printer import Printer

from .xdsl_to_numpy import (
    convert_arrayattr_to_np,
    convert_attrdict_to_lists,
    convert_dense_to_np,
    convert_xdsl_to_np_type,
)
from .numpy_to_xdsl import (
    convert_np_to_arrayattr,
    convert_np_to_emitcarray,
    convert_np_to_tensor,
)
from xdsl.dialects.builtin import (
    TensorType,
    MemRefType,
    IndexType,
    IntegerType,
    FloatAttr,
)
from xdsl.ir import Operation, Block
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects import func, scf
import numpy as np

I64_MIN = -(2**63)


def copy_attribute(attr):
    new_attr = {}
    for k, v in attr.items():
        new_attr[k] = v
    return new_attr


def tensor_to_memref(value: TensorType) -> MemRefType:
    return MemRefType(element_type=value.element_type, shape=value.shape)


def dump_ir(module_op, target_path, run_mlir_opt_passes):
    executable = "mlir-opt"

    # Add some automatic optimizations
    run_mlir_opt_passes += ["--canonicalize", "-cse"]

    if not shutil.which(executable):
        raise ValueError(f"{executable} is not available")

    stream = StringIO()
    printer = Printer(print_generic_format=False, stream=stream)
    printer.print(module_op)

    my_string = stream.getvalue()

    completed_process = subprocess.run(
        [executable, *run_mlir_opt_passes],
        input=my_string,
        capture_output=True,
        text=True,
    )
    try:
        completed_process.check_returncode()
        stdout_output = completed_process.stdout
        with open(target_path, "w") as f:
            f.write(stdout_output)
    except Exception as e:
        raise DiagnosticException(
            "Error executing ir dump:", completed_process.stderr
        ) from e
