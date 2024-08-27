from typing import List, Tuple, List
from treeco.dialects import treeco
from xdsl.ir import Operation
from xdsl.dialects import builtin, arith, linalg
from treeco.dialects.extended import tensor
from xdsl.ir.affine import AffineConstantExpr

AGGREGATE_MODE_VOTE = "VOTE"
AGGREGATE_MODE_SUM = "SUM"


def _aggregate_leaf_tensors_vote(
    #leaf_type: treeco.LeafType,
    tree_idx: Operation,
    leaf_tensor: Operation,
    output_tensor: Operation,
) -> Tuple[List[Operation], Operation]:
    """
    Voting: load the correct value from the output given by the idx in the leaf, add one
    """

    zc = arith.Constant.from_int_and_width(0, builtin.IndexType())
    leaf_idx= tensor.ExtractOp.get(
        tensor=leaf_tensor,
        indices=[zc, zc],
    )
    sub_output = tensor.ExtractOp.get(
        tensor=output_tensor,
        indices=[zc, leaf_idx],
    )
    cast_in = treeco.CastSignOp(
        operand1=sub_output,
        res=builtin.IntegerType(
            sub_output.results[0].type.width,
            signedness=builtin.Signedness.SIGNLESS,
        ),
    )
    one_const_tens = arith.Constant.from_int_and_width(
        1, sub_output.results[0].type.width.data
    )
    added_tensor = arith.Addi(
        operand1=cast_in,
        operand2=one_const_tens,
    )
    cast_out = treeco.CastSignOp(
        operand1=added_tensor,
        res=sub_output.results[0].type,
    )

    new_output = tensor.InsertOp.get(
        scalar=cast_out,
        destination=output_tensor,
        indices=[zc, leaf_idx],
    )

    return (
        zc,
        leaf_idx,
        sub_output,
        one_const_tens,
        cast_in,
        added_tensor,
        cast_out,
        new_output,
    )


def _aggregate_leaf_tensors_sum_single(
    tree_idx: Operation,
    leaf_tensor: Operation,
    output_tensor: Operation,
    load_on_tree_idx: bool = False,
) -> Tuple[List[Operation], Operation]:
    """
    Sum: load the correct value from the output given by the idx in the leaf, add the leaf tensor
    """

    zc = arith.Constant.from_int_and_width(0, builtin.IndexType())
    leaf_value = tensor.ExtractOp.get(
        tensor=leaf_tensor,
        indices=[zc, tree_idx if load_on_tree_idx else zc],
    )
    sub_output = tensor.ExtractOp.get(
        tensor=output_tensor,
        indices=[zc, tree_idx if load_on_tree_idx else zc],
    )

    add_ops = list()
    if isinstance(sub_output.results[0].type, builtin.IntegerType):
        cast_in_output = treeco.CastSignOp(
            operand1=sub_output,
            res=builtin.IntegerType(
                sub_output.results[0].type.width,
                signedness=builtin.Signedness.SIGNLESS,
            ),
        )
        cast_in_leaf = treeco.CastSignOp(
            operand1=leaf_value,
            res=builtin.IntegerType(
                leaf_value.results[0].type.width,
                signedness=builtin.Signedness.SIGNLESS,
            ),
        )
        added_tensor = arith.Addi(
            operand1=cast_in_output,
            operand2=cast_in_leaf,
        )
        cast_out = treeco.CastSignOp(
            operand1=added_tensor,
            res=sub_output.results[0].type,
        )
        add_ops += [cast_in_output, cast_in_leaf, added_tensor, cast_out]
    else:
        added_tensor = arith.Addf(
            operand1=sub_output,
            operand2=leaf_value,
        )
        add_ops += [added_tensor]

    new_output = tensor.InsertOp.get(
        scalar=add_ops[-1],
        destination=output_tensor,
        indices=[zc, tree_idx if load_on_tree_idx else zc],
    )

    return tuple(
        [
            zc,
            leaf_value,
            sub_output,
        ]
        + add_ops
        + [new_output]
    )


def _aggregate_leaf_tensors_sum_multi(
    tree_idx: Operation,
    leaf_tensor: Operation,
    output_tensor: Operation,
) -> Tuple[List[Operation], Operation]:
    """
    Sum: load the correct value from the output given by the idx in the leaf, add the leaf tensor
    """
    add_ops = list()
    if isinstance(output_tensor.type.get_element_type(), builtin.IntegerType):
        cast_in_leaf = treeco.CastSignOp(
            operand1=leaf_tensor,
            res=builtin.TensorType(
                element_type=builtin.IntegerType(
                    leaf_tensor.type.get_element_type().width,
                    signedness=builtin.Signedness.SIGNLESS,
                ),
                shape=leaf_tensor.type.get_shape(),
            ),
        )
        cast_in_output = treeco.CastSignOp(
            operand1=output_tensor,
            res=builtin.TensorType(
                element_type=builtin.IntegerType(
                    output_tensor.type.get_element_type().width,
                    signedness=builtin.Signedness.SIGNLESS,
                ),
                shape=output_tensor.type.get_shape(),
            ),
        )

        added_tensor = linalg.AddOp(
            inputs=[cast_in_output.res, cast_in_leaf.res],
            outputs=[cast_in_output.res],
            res=[cast_in_output.res.type],
        )

        cast_out = treeco.CastSignOp(
            operand1=added_tensor,
            res=output_tensor.type,
        )
        add_ops += [cast_in_output, cast_in_leaf, added_tensor, cast_out]
    else:
        added_tensor = linalg.AddOp(
            inputs=[output_tensor, leaf_tensor],
            outputs=[output_tensor],
            res=[output_tensor.type],
        )

        add_ops += [added_tensor]

    return tuple(add_ops)


def _aggregate_leaf_tensors(
    aggregate_mode: str,
    leaf_type: treeco.LeafType,
    tree_idx: Operation,
    leaf_tensor: Operation,
    output_tensor: Operation,
) -> Tuple[List[Operation], Operation]:
    leaf_shape = leaf_type.get_leaf_shape()
    leaf_map = leaf_type.target_map
    is_constant_map = all(
        [isinstance(x, AffineConstantExpr) for x in leaf_map.data.results]
    )

    if aggregate_mode == AGGREGATE_MODE_VOTE:
        return _aggregate_leaf_tensors_vote(
            tree_idx=tree_idx,
            leaf_tensor=leaf_tensor,
            output_tensor=output_tensor,
        )

    elif aggregate_mode == AGGREGATE_MODE_SUM:
        # OvO
        if not is_constant_map and leaf_shape[-1] < 2:
            return _aggregate_leaf_tensors_sum_single(
                load_on_tree_idx=True,
                leaf_type=leaf_type,
                tree_idx=tree_idx,
                leaf_tensor=leaf_tensor,
                output_tensor=output_tensor,
            )
        elif leaf_shape[-1] < 2:
            return _aggregate_leaf_tensors_sum_single(
                tree_idx=tree_idx,
                leaf_tensor=leaf_tensor,
                output_tensor=output_tensor,
            )
        else:
            return _aggregate_leaf_tensors_sum_multi(
                tree_idx=tree_idx,
                leaf_tensor=leaf_tensor,
                output_tensor=output_tensor,
            )
    else:
        raise ValueError(f"Unsupported aggregate mode {aggregate_mode}")
