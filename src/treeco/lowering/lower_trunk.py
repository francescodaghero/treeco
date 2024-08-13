"""
Lowers the trunk dialect
"""

from xdsl.passes import ModulePass
from xdsl.dialects import printf
from treeco.utils import convert_np_to_tensor
from xdsl.context import MLContext
from treeco.dialects import trunk, treeco
from treeco.model.ensemble import Ensemble
from typing import Optional
from xdsl.dialects import builtin, arith, scf
from xdsl.dialects.builtin import StringAttr
from xdsl.rewriter import InsertPoint

from treeco.dialects.extended import tensor, ml_program
from treeco.utils import I64_MIN
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

ROOTSID_DATA = "roots_ids"
FEATUREIDS_DATA = "nodes_featureids"
FALSENODEIDS_DATA = "nodes_falsenodeids"
TRUENODEIDS_DATA = "nodes_truenodeids"
THRESHOLD_DATA = "nodes_values"
LEAVES_ID_DATA = "targets_ids"
LEAVES_VALUE_DATA = "targets_weights"


def find_global_mlprogram_by_name(module_op: builtin.ModuleOp, name: str):
    for op in module_op.walk():
        if isinstance(op, ml_program.Global) and op.sym_name.data == name:
            return op
    return None


class PartialLowerEnsemble(RewritePattern):
    """
    Dump the ensemble data to a set of globals, depending on the mode argument.
    The ensemble op is NOT deleted, it stays as metadata

    """

    mode: str
    nodes_mode: str
    leaf_array: str

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: trunk.TreeEnsembleConstantOp,
        rewriter: PatternRewriter,
    ):
        self.mode = "iterative"
        ensemble_data = Ensemble.parse_attr(op.ensemble)

        # Export the data according to mode #TODO
        ensemble_mapping, nodes_mode, leaf_array = ensemble_data.to_numpy_arrays(
            node_indices="auto", leaf_values="external"
        )
        # Store the output metadata to avoid re-computations
        self.nodes_mode = nodes_mode
        self.leaf_array = leaf_array

        # Convert the elements to xdsl types
        for k, v in ensemble_mapping.items():
            ensemble_mapping[k] = convert_np_to_tensor(v, is_index="ids" in k)

        # Add the globals
        for k, v in ensemble_mapping.items():
            ml_global_val = ml_program.Global(
                sym_name=StringAttr(k),
                type=ensemble_mapping[k].type,
                is_mutable=None,
                value=ensemble_mapping[k],
                sym_visibility=StringAttr("public"),
            )
            rewriter.insert_op(
                ml_global_val,
                insertion_point=InsertPoint.at_start(
                    op.get_toplevel_object().body.block
                ),
            )

        # Delete the ensemble op. Ensure that the module_op is invalid if
        # all ensembles references are not removed
        # TODO: Is this necessary? Isn't this just a dead constant at the end?
        rewriter.erase_matched_op(safe_erase=False)


class LowerGetRoot(RewritePattern):
    mode: str
    nodes_mode: str
    leaf_array: str

    def __init__(self, mode, nodes_mode, leaf_array, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.nodes_mode = nodes_mode
        self.leaf_array = leaf_array

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: trunk.GetRootOp,
        rewriter: PatternRewriter,
    ):

        tree_element = op.operands[1].owner

        root_cast_in = treeco.Cast(
            operand1=tree_element, output_type=builtin.IndexType()
        )
        root_global_op: Optional[ml_program.Global] = find_global_mlprogram_by_name(
            module_op=op.get_toplevel_object(), name=ROOTSID_DATA
        )
        if not root_global_op:
            return

        roots_tensor = ml_program.GlobalLoadConstant(
            global_attr=builtin.SymbolRefAttr(ROOTSID_DATA),
            result_type=root_global_op.value.type,
        )
        # Load the correct index
        root_idx = tensor.ExtractOp.get(
            tensor=roots_tensor,
            indices=root_cast_in,
        )
        root_cast_out = treeco.Cast(operand1=root_idx, output_type=op.results[0].type)
        rewriter.replace_matched_op(
            [root_cast_in, roots_tensor, root_idx, root_cast_out],
            [root_cast_out.results[0]],
        )


def add_and_get_ensemble_global_leaf(rewriter, ensemble_op) -> str:
    """
    Returns the name of the leaf array. Obvious if it is external, changes if internal.
    """
    ensemble_data = Ensemble.parse_attr(ensemble_op.ensemble)
    ensemble_mapping, _, leaf_position_name = ensemble_data.to_numpy_arrays(
        node_indices="auto", leaf_values="external"
    )
    global_op: Optional[ml_program.Global] = find_global_mlprogram_by_name(
        module_op=ensemble_op.get_toplevel_object(), name=leaf_position_name
    )

    if not global_op:
        rewriter.insert_op(
            global_op,
            insertion_point=InsertPoint.at_start(
                ensemble_op.get_toplevel_object().body.block
            ),
        )

    return global_op


def is_external(leaf_array: str) -> bool:
    return "target" in leaf_array.lower()


class LowerIsLeaf(RewritePattern):
    mode: str
    nodes_mode: str
    leaf_array: str

    def __init__(self, mode, nodes_mode, leaf_array, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.nodes_mode = nodes_mode
        self.leaf_array = leaf_array

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: trunk.IsLeafOp,
        rewriter: PatternRewriter,
    ):
        n_features: int = op.operands[0].type.n_features.value.data

        # Node to idx
        cast_in = treeco.Cast(operand1=op.operands[1], output_type=builtin.IndexType())

        # Load the feature id
        nodes_featureids_global = find_global_mlprogram_by_name(
            module_op=op.get_toplevel_object(), name=FEATUREIDS_DATA
        )
        if nodes_featureids_global is None:
            return

        nodes_featureids_data = ml_program.GlobalLoadConstant(
            global_attr=builtin.SymbolRefAttr(FEATUREIDS_DATA),
            result_type=nodes_featureids_global.value.type,
        )

        # Load the correct index
        feature_idx = tensor.ExtractOp.get(
            tensor=nodes_featureids_data,
            indices=cast_in,
        )
        n_features_const = arith.Constant.from_int_and_width(
            n_features, builtin.IndexType()
        )

        out_bool = arith.Cmpi(operand1=feature_idx, operand2=n_features_const, arg="ne")
        cast_out = treeco.Cast(operand1=cast_in, output_type=op.operands[1].type)

        rewriter.replace_matched_op(
            [
                cast_in,
                nodes_featureids_data,
                feature_idx,
                n_features_const,
                out_bool,
                cast_out,
            ],
            [cast_out.results[0]],
        )
        cond_op = scf.Condition(out_bool, cast_out)
        rewriter.replace_op(cast_out.next_op, [cond_op], [])


class LowerVisitNextNode(RewritePattern):
    mode: str
    nodes_mode: str
    leaf_array: str

    def __init__(self, mode, nodes_mode, leaf_array, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.nodes_mode = nodes_mode
        self.leaf_array = leaf_array

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: trunk.VisitNextNodeOp,
        rewriter: PatternRewriter,
    ):
        if self.nodes_mode != "rchild":
            return
        node_idx = treeco.Cast(operand1=op.operands[1], output_type=builtin.IndexType())

        zero_constant = arith.Constant.from_int_and_width(0, builtin.IndexType())
        one_constant = arith.Constant.from_int_and_width(1, builtin.IndexType())
        nodes_values_global = find_global_mlprogram_by_name(
            module_op=op.get_toplevel_object(), name=THRESHOLD_DATA
        )
        if nodes_values_global is None:
            return

        nodes_falsenodeids_global = find_global_mlprogram_by_name(
            module_op=op.get_toplevel_object(), name=FALSENODEIDS_DATA
        )
        if nodes_falsenodeids_global is None:
            return

        nodes_featureids_global = find_global_mlprogram_by_name(
            module_op=op.get_toplevel_object(), name=FEATUREIDS_DATA
        )
        if nodes_featureids_global is None:
            return

        # Load the constants
        nodes_values = ml_program.GlobalLoadConstant(
            global_attr=builtin.SymbolRefAttr(THRESHOLD_DATA),
            result_type=nodes_values_global.value.type,
        )
        nodes_falsenodeids = ml_program.GlobalLoadConstant(
            global_attr=builtin.SymbolRefAttr(FALSENODEIDS_DATA),
            result_type=nodes_falsenodeids_global.value.type,
        )
        nodes_featureids = ml_program.GlobalLoadConstant(
            global_attr=builtin.SymbolRefAttr(FEATUREIDS_DATA),
            result_type=nodes_featureids_global.value.type,
        )

        # Load the value
        threshold_val = tensor.ExtractOp.get(
            tensor=nodes_values,
            indices=node_idx,
        )
        right_shift = tensor.ExtractOp.get(
            tensor=nodes_falsenodeids,
            indices=node_idx,
        )
        feature_idx = tensor.ExtractOp.get(
            tensor=nodes_featureids,
            indices=node_idx,
        )

        input_val = tensor.ExtractOp.get(
            op.operands[2], indices=[zero_constant, feature_idx]
        )
        if isinstance(input_val.results[0].type, builtin.IntegerType):
            is_signed = (
                input_val.results[0].type.signedness == builtin.Signedness.SIGNED
            )
            cmp = "sgt" if is_signed else "ugt"
            cmp_out = arith.Cmpi(input_val, threshold_val, cmp)
        else:
            # Ordered -> Neither can be NaN
            cmp_out = arith.Cmpf(input_val, threshold_val, "ogt")

        node_plus = arith.Addi(node_idx, one_constant, result_type=builtin.IndexType())
        # Included as constant
        # right_minus = arith.Subi(
        #    right_shift, one_constant, result_type=builtin.IndexType()
        # )
        cmp_out_int = arith.ExtUIOp(cmp_out, builtin.IntegerType(64))
        cmp_out_idx = arith.IndexCastOp(cmp_out_int, builtin.IndexType())
        mul_out = arith.Muli(cmp_out_idx, right_shift, result_type=builtin.IndexType())
        new_node_idx = arith.Addi(node_plus, mul_out, result_type=builtin.IndexType())

        pr = printf.PrintFormatOp(
            "NEW_NODE: {} - NODEPL {} - RS {} - CMP {} - MOUT: {}, THRESHOLD: {}",
            new_node_idx,
            node_plus,
            right_shift,
            cmp_out_idx,
            mul_out,
            threshold_val,
        )
        cast_out = treeco.Cast(operand1=new_node_idx, output_type=op.results[0].type)
        rewriter.replace_matched_op(
            [
                node_idx,
                zero_constant,
                one_constant,
                nodes_values,
                nodes_falsenodeids,
                nodes_featureids,
                threshold_val,
                right_shift,
                feature_idx,
                input_val,
                cmp_out,
                node_plus,
                # right_minus,
                cmp_out_int,
                cmp_out_idx,
                mul_out,
                new_node_idx,
                # pr,
                cast_out,
            ],
            [cast_out.results[0]],
        )


class LowerGetTreeOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: trunk.GetTreeOp,
        rewriter: PatternRewriter,
    ):
        cast_out = treeco.Cast(operand1=op.operands[1], output_type=op.results[0].type)
        rewriter.replace_matched_op([cast_out], [cast_out.results[0]])


class LowerGetLeafOp(RewritePattern):
    """
    When lowered it will become:
    From the idx of the current node, get the idx of the leaf element.
    If leaves are internal, this is a noop
    """

    mode: str
    nodes_mode: str
    leaf_array: str

    def __init__(self, mode, nodes_mode, leaf_array, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.nodes_mode = nodes_mode
        self.leaf_array = leaf_array

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: trunk.GetLeafOp,
        rewriter: PatternRewriter,
    ):
        cast_in = treeco.Cast(operand1=op.operands[1], output_type=builtin.IndexType())

        # If the the elements in the leaves are 1, then they are internal.
        # This op then is erased.

        # External OP
        to_be_recasted = cast_in
        additional_ops = []
        if is_external(self.leaf_array):
            name = FALSENODEIDS_DATA
            leaf_global = find_global_mlprogram_by_name(
                module_op=op.get_toplevel_object(), name=name
            )
            # In this case the tree was perfect...,
            if leaf_global is None:
                name = THRESHOLD_DATA
                # TODO : Can i store it in the feature_id? Or is it better in the thresholds?
                leaf_global = find_global_mlprogram_by_name(
                    module_op=op.get_toplevel_object(), name=name
                )
            leaf_idx_store = ml_program.GlobalLoadConstant(
                global_attr=builtin.SymbolRefAttr(name),
                result_type=leaf_global.value.type,
            )
            leaf_idx = tensor.ExtractOp.get(tensor=leaf_idx_store, indices=cast_in)
            additional_ops.extend([leaf_idx_store, leaf_idx])
            to_be_recasted = leaf_idx
        pr = printf.PrintFormatOp(
            "LEAF: {}",
            to_be_recasted,
        )
        cast_out = treeco.Cast(operand1=to_be_recasted, output_type=op.results[0].type)
        rewriter.replace_matched_op(
            [cast_in]
            + additional_ops
            + [
                # pr,
                cast_out,
            ],
            [cast_out.results[0]],
        )


class LowerGetLeafValueOp(RewritePattern):
    mode: str
    nodes_mode: str
    leaf_array: str

    def __init__(self, mode, nodes_mode, leaf_array, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.nodes_mode = nodes_mode
        self.leaf_array = leaf_array

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: trunk.GetLeafValueOp, rewriter: PatternRewriter):
        # Input: the index of the leaf in the right array.
        cast_in = treeco.Cast(operand1=op.operands[1], output_type=builtin.IndexType())

        # Load the leaf array
        leavesvalues_global = find_global_mlprogram_by_name(
            module_op=op.get_toplevel_object(), name=self.leaf_array
        )
        leaves = ml_program.GlobalLoadConstant(
            global_attr=builtin.SymbolRefAttr(self.leaf_array),
            result_type=leavesvalues_global.value.type,
        )

        # Slice the tensor to get only the leaf value.
        # This depends on the number of dimensions.
        output_shape = op.results[0].type.get_shape()
        output_slice = tensor.ExtractSliceOp.build(
            operands=[leaves, [cast_in], [], []],
            result_types=[op.results[0].type],
            properties={
                "static_offsets": builtin.DenseArrayBase.from_list(
                    builtin.i64, [I64_MIN] + [0] * (len(output_shape) - 1)
                ),
                "static_sizes": builtin.DenseArrayBase.from_list(
                    builtin.i64, op.results[0].type.get_shape()
                ),
                "static_strides": builtin.DenseArrayBase.from_list(
                    builtin.i64, [1 for _ in range(len(output_shape))]
                ),
            },
        )
        rewriter.replace_matched_op(
            [cast_in, leaves, output_slice], [output_slice.results[0]]
        )


class LowerTrunkPass(ModulePass):
    name = "lower-trunk"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        lowering_pass = PartialLowerEnsemble()
        PatternRewriteWalker(lowering_pass).rewrite_module(op)
        # Retrieve the data
        mode, nodes_mode, leaf_array = (
            lowering_pass.mode,
            lowering_pass.nodes_mode,
            lowering_pass.leaf_array,
        )
        PatternRewriteWalker(LowerGetTreeOp()).rewrite_module(op)
        PatternRewriteWalker(
            LowerGetRoot(mode=mode, nodes_mode=nodes_mode, leaf_array=leaf_array)
        ).rewrite_module(op)
        PatternRewriteWalker(
            LowerIsLeaf(mode=mode, nodes_mode=nodes_mode, leaf_array=leaf_array)
        ).rewrite_module(op)
        PatternRewriteWalker(
            LowerVisitNextNode(mode=mode, nodes_mode=nodes_mode, leaf_array=leaf_array)
        ).rewrite_module(op)

        PatternRewriteWalker(
            LowerGetLeafOp(mode=mode, nodes_mode=nodes_mode, leaf_array=leaf_array)
        ).rewrite_module(op)

        PatternRewriteWalker(
            LowerGetLeafValueOp(mode=mode, nodes_mode=nodes_mode, leaf_array=leaf_array)
        ).rewrite_module(op)

        op.verify()
