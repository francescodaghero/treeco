# **TreeCo**: Tree COmpiler for efficient inferences
TreeCo is an [xDSL](https://github.com/xdslproject/xdsl)-based AI compiler for tree-based ensembles focusing on the inference.
It parses an ONNX TreeEnsemble(Classifier/Regressor), and generates a valid MLIR-based representation that can be further optimized according to the user preferences.
Optimizations such as leaf/node quantization, logits-> vote conversion for Random Forest classifiers, loop unrolling are already implemented, and more are on the way!

<b>Disclaimer</b> This package is still under construction and is more of a personal project to explore MLIR/xDSL than a full-fledged deployment solution.

For an optimized version of this package targeting RISC-V, please refer to the original framework: [Eden](https://github.com/eml-eda/eden).

### TreeCo Dialects
TreeCo implements three dialects:
- Crown: for high-level modeling of the inference of the ensemble. The dialect is compact and useful to perform optimization that change the structure of the ensemble (e.g. tree pruning, tree padding).
- Trunk: designed for mid-level modeling of the inference, focusing specifically on the tree visit algorithm (e.g. fully unrolled, iterative, vectorized...) and related optimizations (e.g. tree peeling).
- Treeco: dialect implementing custom data types for the ensembles (node, tree, ensemble..) and few operations for casting operators.

Noteworthy, since we stick to Python for the whole stack, all modification to the ensemble (e.g. padding) are done on a easy-to-navigate python class (found under model/ensemble.py).
This class also implements a (slow) predict function, fully independent from the ir.
See an example of standalone usage in the test cases.

Following here some examples of the IRs.

#### Crown
```mlir
builtin.module {
  func.func public @inference(%arg0 : memref<2x10xf32>, %arg1 : memref<2x3xf32>) {
    "crown.tree_ensemble_predict"(%arg0, %arg1) {"ensemble" = #treeco.ensembleAttr<...>} : (memref<2x10xf32>, memref<2x3xf32>) -> ()
    func.return
  }
}
```
#### Trunk
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

## How to use: 
### Setup
First, generate an onnx file that contains your tree ensemble. 
You can obtain one from a scikit-learn model using [skl2onnx](https://onnx.ai/sklearn-onnx/) (see the test cases for an example).
Currently TreeCo supports both [TreeEnsembleClassifier](https://onnx.ai/onnx/operators/onnx_aionnxml_TreeEnsembleClassifier.html) or [TreeEnsembleRegressor](https://onnx.ai/onnx/operators/onnx_aionnxml_TreeEnsembleRegressor.html).

### Importing in Treeco
Importing the onnx file in TreeCo should be straighforward, and requires only calling the Parser.
```python
ensemble_ast = parse_ensemble("model.onnx")
ensemble_ir, ctx = generate_ir(ensemble_ast, batch_size=1)
```

At this step, you decide the batch size that the inference function will use. While no parallelization is currently implemented, I wanted higher flexibility from the beginning.
The variable ensemble_ir stores the IR of the model, using a TreeCo dialect named ONNX-ML, that is, a close mirror of the onnx-ml operations.

N.B. A batch size of 1 will still generate a for loop of with upper bound 1, optimized away while lowering.

### Lowering : from ONNX-ML IR to LLVM IR
The lowering can be fully performed in python and does not depend on MLIR until the bufferization pass (i.e. after lowering Trunk to Arith/Ml_Program).

Continuing from the code snippet above:
```python
from treeco.compiler import crown_transform, trunk_transform, root_transform
# Lowers from onnxml to crown and performs crown-related optimization passes
# (Listed as argument of the function)
ensemble_ir = crown_transform(ensemble_ir, ctx)
# From crown to trunk
ensemble_ir = trunk_transform(ensemble_ir, ctx)
# From Trunk to Arith/ML_Program/Tensor. Then performs bufferization 
ensemble_ir = root_transform(ensemble_ir, ctx)
# Lowers to LLVM IR, requires MLIR and cannot currently be parsed back to python.
target_transform_and_dump("output.mlir", module_op=ensemble_ir, ctx=ctx)
```
The ensemble file can then be compiled as shared library and run or a main function can be added (see treeco/targets/llvm/standalone.py/AddLLVMMainPass()) for a standalone file.
An example:
```python3
# To be added before calling target_transform_and_dump
from treeco.targets.llvm import AddLLVMMainPass
AddLLVMMainPass().apply(ctx, ensemble_ir, test_data=data_test)
```
The output mlir file can be then translated to llvm and compiled with clang.

## Why not using X/Y/Z?
Many available frameworks focus on tensor-level operations while tree ensemble generally need scalar ones. Implementing them in other frameworks can rapidly become tricky.
xDSL, being Python-based and compatible with MLIR seemed an ideal alternative to implement a custom solution. New high-level passes can be rapidly developed in Python, while reusing most lowering passes available in TreeCo/xDSL/MLIR.

## Requirements
Aside from the package requirements, an MLIR installation compatible with XDSL is required to lower to LLVM IR.
To translate to C, mlir-translate of version >= 20 is required (Note, only mlir-translate, an xDSL-compatible mlir-opt should be used for all passes before).


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
