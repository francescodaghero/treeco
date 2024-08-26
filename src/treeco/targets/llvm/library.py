import ctypes
import pathlib
import subprocess
import os
import numpy as np


def compile_as_library(build_dir=".", mlir_path="output.mlir"):
    if not os.path.exists(mlir_path):
        raise RuntimeError(f"File {mlir_path} does not exist")

    build_dir: pathlib.Path = pathlib.Path(build_dir)
    mlir_path: pathlib.Path = pathlib.Path(mlir_path)
    # Change the extension of the file to .ll
    ll_path = build_dir / pathlib.Path(str(mlir_path).replace(".mlir", ".ll"))
    o_path = build_dir / pathlib.Path(str(mlir_path).replace(".mlir", ".o"))
    so_path = build_dir / pathlib.Path(str(mlir_path).replace(".mlir", ".so"))

    if os.path.exists(build_dir) == False:
        os.mkdir(build_dir)

    # MLIR to LL
    success = subprocess.check_call(
        ["mlir-translate", "-mlir-to-llvmir", mlir_path, "-o", ll_path]
    )
    if success != 0:
        raise RuntimeError("Failed to convert MLIR to LLVM IR")

    success = subprocess.check_output(
        ["clang", "-g", "-O3", "-fPIC", "-shared", ll_path, "-o", so_path]
    )
    lib = ctypes.CDLL(so_path, use_last_error=True)
    return lib


def run_ctype_inference(
    lib: ctypes.CDLL,
    buffer_in: np.ndarray,
    buffer_out: np.ndarray,
    function_name: str = "inference",
):
    func = getattr(lib, function_name)
    # Ensure memory is contiguous
    buffer_in = np.ascontiguousarray(buffer_in)
    buffer_out = np.ascontiguousarray(buffer_out)
    # The two buffer pointers depend on the dtype of the buffers
    func.argtypes = (
        # Buffer1 - Base ptr + aligned ptr
        ctypes.POINTER(
            ctypes.c_void_p,
        ),
        ctypes.POINTER(
            ctypes.c_void_p,
        ),
        # offset
        ctypes.c_int,
        # size 1
        ctypes.c_int,
        # size 2
        ctypes.c_int,
        # stride 1
        ctypes.c_int,
        # stride 2,
        ctypes.c_int,
        # Buffer2 - Base ptr + aligned ptr
        ctypes.POINTER(
            ctypes.c_void_p,
        ),
        ctypes.POINTER(
            ctypes.c_void_p,
        ),
        # offset
        ctypes.c_int,
        # size 1
        ctypes.c_int,
        # size 2
        ctypes.c_int,
        # stride 1
        ctypes.c_int,
        # stride 2,
        ctypes.c_int,
    )

    buffer_in_ptr = buffer_in.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))
    buffer_in_ptr_aligned = buffer_in.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))

    buffer_out_ptr = buffer_out.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))
    buffer_out_ptr_aligned = buffer_out.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))

    func(
        buffer_in_ptr,
        buffer_in_ptr_aligned,
        0,
        buffer_in.shape[0],
        buffer_in.shape[1],
        buffer_in.shape[1],
        1,
        buffer_out_ptr,
        buffer_out_ptr_aligned,
        0,
        buffer_out.shape[0],
        buffer_out.shape[1],
        buffer_out.shape[1],
        1,
    )
