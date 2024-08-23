"""
Lowers the crown dialects to either trunk or in-tree dialects
Instead of performing a single pass on the ensemble, 
we perform multiple to lower piece-by-piece the operation.
"""

from xdsl.passes import ModulePass
from xdsl.context import MLContext
from xdsl.builder import Builder
from typing import Any
from treeco.dialects import crown, trunk
from treeco.model.ensemble import Ensemble
from xdsl.dialects import builtin, arith, scf
from treeco.dialects.extended import tensor, bufferization
from treeco.utils import I64_MIN
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from treeco.lowering._utils_trunk_leaf_aggregate import _aggregate_leaf_tensors


class LowerEnsembleToMatmulTraverse(RewritePattern):
    def __init__(self, mode="hummingbird_gemm"):
        super().__init__()

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: crown.TreeEnsembleOp,
        rewriter: PatternRewriter,
    ):
        raise NotImplementedError("Not implemented yet")


class LowerPostTransform(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: crown.TreeEnsembleOp,
        rewriter: PatternRewriter,
    ):
        ensemble_data = Ensemble.parse_attr(op.ensemble)
        if ensemble_data.post_transform == "NONE":
            return

        # Change it to NONE and rebuild the ensemble
        # TODO : Add the supported post-transforms + the option to skip them
        raise NotImplementedError("Not implemented yet")
        ensemble_data.post_transform = "NONE"
        crown.TreeEnsembleOp(
            buffer_in=op.operands[0],
            ensemble_attr=Ensemble.to_attr(ensemble_data),
            buffer_out=op.operands[1],
        )


class LowerEnsembleToIterativeTraverse(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: crown.TreeEnsembleOp,
        rewriter: PatternRewriter,
    ):

        ensemble_data = Ensemble.parse_attr(op.ensemble)

        # Get data from the ensemble
        input_buffer = op.operands[0]
        output_buffer = op.operands[1]

        # ----------- Beginning of the operations -----------
        # Tensorize, not strictly needed here, it's just to see the MLIR
        # bufferization in action. Plus, it should make the whole flow easier to export
        input_tensor = bufferization.ToTensorOp(
            memref=input_buffer, restrict=True, writable=False
        )
        output_tensor = bufferization.ToTensorOp(
            memref=output_buffer, restrict=True, writable=True
        )

        # ------------- Operations end -------------

        # A python "pointer" for shorter code.
        input_data = input_tensor.results[0]
        output_data = output_tensor.results[0]

        # Some metadata of the tensors/buffers
        output_type: tensor.TensorType = output_data.type
        output_slice_type = tensor.TensorType(
            element_type=output_data.type.element_type,
            shape=[1] + list(output_type.get_shape())[1:],
        )

        # Build the Treeco types
        node_type = op.ensemble.node_type()
        leaf_type = op.ensemble.leaf_type()

        # Some constants used for loops
        n_features = ensemble_data.n_features
        n_trees = ensemble_data.n_trees
        batch_size = input_data.type.shape.data[0].data
        n_targets = output_data.type.shape.data[-1].data

        # ----------- Start of the operations -----------
        # The ensemble constant
        ensemble_op = trunk.TreeEnsembleConstantOp(
            ensemble_attr=op.ensemble, return_type=op.ensemble.ensemble_type()
        )

        # Constants for loops
        zero_const = arith.Constant.from_int_and_width(0, builtin.IndexType())
        one_const = arith.Constant.from_int_and_width(1, builtin.IndexType())
        batch_size_const = arith.Constant.from_int_and_width(
            batch_size, builtin.IndexType()
        )
        n_trees_const = arith.Constant.from_int_and_width(n_trees, builtin.IndexType())

        # An all zero tensor to store the output
        @Builder.implicit_region((builtin.IndexType(), output_type))
        def input_loop_body(args_input: tuple[Any, ...]) -> None:
            (input_idx, output_tensor_batch) = args_input
            input_slice = tensor.ExtractSliceOp.build(
                operands=[input_data, [input_idx], [], []],
                result_types=[
                    builtin.TensorType[Any](
                        input_data.type.get_element_type(), [1, n_features]
                    )
                ],
                properties={
                    "static_offsets": builtin.DenseArrayBase.from_list(
                        builtin.i64, [I64_MIN, 0]
                    ),
                    "static_sizes": builtin.DenseArrayBase.from_list(
                        builtin.i64, [1, n_features]
                    ),
                    "static_strides": builtin.DenseArrayBase.from_list(
                        builtin.i64, [1, 1]
                    ),
                },
            )
            output_slice = tensor.ExtractSliceOp.build(
                operands=[output_tensor_batch, [input_idx], [], []],
                result_types=[
                    builtin.TensorType[Any](
                        output_tensor_batch.type.get_element_type(), [1, n_targets]
                    )
                ],
                properties={
                    "static_offsets": builtin.DenseArrayBase.from_list(
                        builtin.i64, [I64_MIN, 0]
                    ),
                    "static_sizes": builtin.DenseArrayBase.from_list(
                        builtin.i64, [1, n_targets]
                    ),
                    "static_strides": builtin.DenseArrayBase.from_list(
                        builtin.i64, [1, 1]
                    ),
                },
            )

            @Builder.implicit_region((builtin.IndexType(), output_slice_type))
            def trees_loop_body(args_trees: tuple[Any, ...]):
                (tree_idx, output_tensor_tree) = args_trees
                tree_element = trunk.GetTreeOp(
                    ensemble_op, tree_idx, result_type=op.ensemble.tree_type()
                )
                root_node = trunk.GetRootOp(ensemble_op, tree_element)

                @Builder.implicit_region((node_type,))
                def before_region(args_before: tuple[Any, ...]):
                    (node,) = args_before
                    is_leaf = trunk.IsLeafOp(tree_element, node)
                    scf.Condition(is_leaf, node)

                @Builder.implicit_region((node_type,))
                def after_region(args_after: tuple[Any, ...]):
                    (node,) = args_after
                    next_node = trunk.VisitNextNodeOp(
                        tree=tree_element, node=node, data_in=input_slice
                    )
                    scf.Yield(next_node)

                while_loop = scf.While(
                    [root_node], [node_type], before_region, after_region
                )
                leaf_op = trunk.GetLeafOp(tree=tree_element, node=while_loop)
                leaf_value = trunk.GetLeafValueOp(tree=tree_element, leaf=leaf_op)
                new_output_tensor_tree = trunk.AggregateLeafOp(
                    ensemble = ensemble_op,
                    tree = tree_element,
                    leaf = leaf_value,
                    tensor_out = output_tensor_tree,
                )

                scf.Yield(new_output_tensor_tree)


            tree_for = scf.For(
                lb=zero_const,
                ub=n_trees_const,
                step=one_const,
                iter_args=(output_slice,),
                body=trees_loop_body,
            )
            output_tensor_iter_inputs = tensor.InsertSliceOp.get(
                source=tree_for,
                dest=output_tensor_batch,
                static_offsets=[I64_MIN, 0],
                static_sizes=[1, n_targets],
                static_strides=[1, 1],
                offsets=[input_idx],
                sizes=[],
                strides=[],
                result_type=output_type,
            )
            scf.Yield(output_tensor_iter_inputs)

        input_for = scf.For(
            lb=zero_const,
            ub=batch_size_const,
            step=one_const,
            iter_args=(output_tensor,),
            body=input_loop_body,
        )
        materialize = bufferization.MaterializeInDestinationOp(
            source=input_for, dest=output_buffer, writable=builtin.UnitAttr()
        )
        rewriter.replace_matched_op(
            [
                # Bufferization
                input_tensor,
                output_tensor,
                # External variables
                ensemble_op,
                zero_const,
                one_const,
                batch_size_const,
                n_trees_const,
                # Nested loops
                input_for,
                # Materialization
                materialize,
            ],
            [],
        )


class LowerEnsembleToVectorTraverse(RewritePattern):
    pass


class LowerEnsemblePostTransform(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: crown.TreeEnsembleOp,
        rewriter: PatternRewriter,
    ):
        ensemble_data = Ensemble.parse_attr(op.ensemble)
        # Possible values are taken from :
        # https://onnx.ai/onnx/operators/onnx_aionnxml_TreeEnsembleClassifier.html
        # ‘NONE,’ ‘SOFTMAX,’ ‘LOGISTIC,’ ‘SOFTMAX_ZERO,’ or ‘PROBIT.’
        if op.ensemble.post_transform == "NONE":
            return
        else:
            raise NotImplementedError(
                "SOFTMAX, LOGISTIC, SOFTMAX_ZERO, PROBIT not implemented yet"
            )


class LowerEnsembleAggregateMode(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: crown.TreeEnsembleOp,
        rewriter: PatternRewriter,
    ):
        ensemble_data = Ensemble.parse_attr(op.ensemble)
        # Possible values are taken from :
        # https://onnx.ai/onnx/operators/onnx_aionnxml_TreeEnsembleClassifier.html
        # "AVG", "SUM", "MIN", "MAX", "VOTE"
        # "VOTE" is added by us
        if op.ensemble.aggregate_mode == "NONE":
            return
        else:
            raise NotImplementedError("SUM, AVG, MIN, MAX not implemented yet")


class ConvertCrownToTrunkIterativePass(ModulePass):
    name = "lower-crown-to-trunk"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(LowerPostTransform()).rewrite_module(op)
        PatternRewriteWalker(LowerEnsembleToIterativeTraverse()).rewrite_module(op)
        op.verify()
