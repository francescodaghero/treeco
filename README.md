# **TreeCo**: Tree COmpiler for efficient inferences
TreeCo is an [xDSL](https://github.com/xdslproject/xdsl)-based AI compiler for tree-based ensembles focusing on the inference.
It parses an ONNX TreeEnsemble(Classifier/Regressor), and generates a valid MLIR-based representation that can be further optimized according to the user preferences.
Optimizations such as leaf/node quantization, logits-> vote conversion for Random Forest classifiers, loop unrolling are already implemented, and more are on the way!

<b>Disclaimer</b> This package is still under construction and is more a personal project to explore MLIR/xDSL than a full-fledged deployment solution.

For an optimized version of this package targeting RISC-V, please refer to the original framework: [Eden](https://github.com/eml-eda/eden).

## An example 
### IR
Crown is the highest level dialect after the ONNXML IR. Here most optmization passes that change the structure of the tree are performed.
```mlir
builtin.module {
  func.func public @inference(%arg0 : memref<2x10xf32>, %arg1 : memref<2x3xf32>) {
    "crown.tree_ensemble_predict"(%arg0, %arg1) {"ensemble" = #treeco.ensembleAttr<...>} : (memref<2x10xf32>, memref<2x3xf32>) -> ()
    func.return
  }
}
```
Then we lower to Trunk (here a shortened version), a dialect that models more precisely the tree visit algorithm.
```mlir
  func.func public @inference(%arg0: memref<2x10xf32>, %arg1: memref<2x3xf32>) {
    %0 = bufferization.to_tensor %arg0 restrict : memref<2x10xf32>
    %1 = bufferization.to_tensor %arg1 restrict writable : memref<2x3xf32>
    %2 = "trunk.tree_ensemble_constant"() ...
    // Loop on the inputs
    %3 = scf.for %arg2 = %c0 to %c2 step %c1 iter_args(%arg3 = %1) -> (tensor<2x3xf32>) {
      %extracted_slice = tensor.extract_slice %0[%arg2, 0] [1, 10] [1, 1] : tensor<2x10xf32> to tensor<1x10xf32>
      %extracted_slice_0 = tensor.extract_slice %arg3[%arg2, 0] [1, 3] [1, 1] : tensor<2x3xf32> to tensor<1x3xf32>
      // Loop on the trees
      %4 = scf.for %arg4 = %c0 to %c2 step %c1 iter_args(%arg5 = %extracted_slice_0) -> (tensor<1x3xf32>) {
        // The actual tree visit - an iterative one in this case
        %5 = "trunk.get_tree"(%2, %arg4) 
        %6 = "trunk.get_root"(%2, %5) 
        %7 = scf.while (%arg6 = %6) : (!treeco.node<f32>) -> !treeco.node<f32> {
          %11 = "trunk.is_leaf"(%5, %arg6) 
          scf.condition(%11) %arg6 : !treeco.node<f32>
        } do {
        ^bb0(%arg6: !treeco.node<f32>):
          %11 = "trunk.visit_next_node"(%5, %arg6, %extracted_slice) <{mode = "right_child"}> 
          scf.yield %11 : !treeco.node<f32>
        }
  ...
```
Finally, this can be lowered to LLVM IR.

### The flow
Given an onnx file storing only the model.
```python
from treeco.compiler import *
from treeco.model import Ensemble
from treeco.dialects import treeco, crown
from treeco.utils import find_operation_in_module
import numpy as np
from treeco.targets.llvm.library import compile_as_library, run_inference

ensemble_ast = parse_ensemble("rf.onnx")
ensemble_ir, ctx = generate_ir(ensemble_ast, batch_size=2)
module_op = crown_transform(ensemble_ir, ctx)
# Lower to Trunk 
module_op = trunk_transform(module_op, ctx)
# Exit from TreeCo.
module_op = target_transform(module_op, ctx, target="llvm")

# Compiles the inference function as a shared library and imports it via ctypes
lib = compile_as_library(".", "output.mlir")
run_inference(
    lib,
    test_data,
    buffer_out := np.zeros((batch_size, ensemble_data.n_targets), dtype=np.float32),
)
```

## Why not using X/Y/Z?
Most available frameworks (e.g. TVM/IREE) focus on Deep Learning (DLs) operators, while tree ensemble need scalar ones. Implementing them in other frameworks can rapidly become tricky.
xDSL, being Python-based and compatible with MLIR seemed an ideal alternative to implement a custom solution. New high-level passes can be rapidly developed in Python, while reusing most lowering passes available in TreeCo/xDSL/MLIR.

## Requirements
Aside from the package requirements, an MLIR installation compatible with XDSL is required to lower to LLVM IR.
To translate to C, mlir-translate of version >= 20 is required (Note, only mlir-translate, a compatible mlir-opt should be used for all passes before).


## Citing 
If you use this code in your work, please consider citing the original framework:
```
@article{daghero2023dynamic,
  title={Dynamic Decision Tree Ensembles for Energy-Efficient Inference on IoT Edge Nodes},
  author={Daghero, Francesco and Burrello, Alessio and Macii, Enrico and Montuschi, Paolo and Poncino, Massimo and Pagliari, Daniele Jahier},
  journal={IEEE Internet of Things Journal},
  year={2023},
  publisher={IEEE}
}
```
