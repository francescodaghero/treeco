import numpy as np
from xdsl.dialects.builtin import (
    ModuleOp,
    ModuleOp,
)
from typing import Optional

from xdsl.context import MLContext
from xdsl.passes import ModulePass
from xdsl.transforms.printf_to_llvm import PrintfToLLVM
from treeco.targets.codegen_main import generate_main_function


class AddLLVMMainPass(ModulePass):
    name = "generate-and-add-main"

    def apply(self, ctx: MLContext, op: ModuleOp, test_data: Optional[np.array] = None):
        # Get the base main function
        main_module: ModuleOp = generate_main_function(
            inference_module_op=op, test_data=test_data
        )
        PrintfToLLVM().apply(ctx, main_module)
