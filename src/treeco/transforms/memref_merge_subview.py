""" 
Merge chains of chains of subviews of chains of subviews that end in load/stores ops.
Supports only subviews handling offsets, not sizes or strides.
Simplifies the readability, possibly simplifies codegen/bufferization/etc
"""

from xdsl.dialects import builtin, arith, memref
from xdsl.rewriter import InsertPoint

from treeco.utils import I64_MIN
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.passes import ModulePass
from xdsl.context import MLContext
from xdsl.dialects import builtin, arith
from treeco.utils import I64_MIN
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class FoldMemRefSubViewChain(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: memref.Subview,
        # Tensor part currently disabled
        # | tensor.ExtractSliceOp,
        rewriter: PatternRewriter,
    ):
        # Check if the only dynamic dimension is the offset, others are not supported
        # atm
        if len(op.offsets) > 0 and (len(op.sizes) != 0 or len(op.strides) != 0):
            return
        # Also force the output dimension to be identical to the input one for subviews.
        # No changes in rank.
        if len(op.result.type.get_shape()) != len(op.operands[0].type.get_shape()):
            return
        # If this is the case, understand on which dimensions we have a dyn offset
        # dimension_map : stores for each dimension (Key) if there is a dyn or stat
        # offset
        dimension_map = {}
        # The number of dyn dimensions
        n_dynamic_dims = 0
        # Iterate over the static offsets property

        for i, offset in enumerate(op.static_offsets.data.data):
            # A dynamic dimension
            if offset.data == I64_MIN:
                dimension_map[i] = op.offsets[n_dynamic_dims]
                n_dynamic_dims += 1
            # A static dimension
            elif offset.data != 0:
                dimension_map[i] = arith.Constant.from_int_and_width(
                    offset.data, builtin.IndexType()
                )

        # Nothing to do, no static offsets or dynamic ones
        if len(dimension_map) == 0:
            return

        # Iterate on the OP using this subview
        # Remove one-by-one the dependency on the subview if possible
        uses = list(op.result.uses)  # Copy the list, we will modify it
        for use in uses:
            op_use = use.operation
            # This is not ok, as the iter_arg should be equal to the return type.
            # if len(op_use.regions)!= 0:
            #    # Move the slicing into the block, i.e. repeat the slicing op the highest number of times.
            #    # The opposite of a smart move.
            #    duplicate_op_in_regions()

            if not hasattr(op_use, "indices"):  # Support only ops with indices as field
                continue

            # Create the new operands
            new_results = op_use.results
            new_attributes = op_use.attributes
            new_properties = op_use.properties
            regions = op_use.regions
            new_operands = []

            new_dims = list()
            dim_idx = 0
            for idx, operand in enumerate(op_use.operands):
                # Now we start with the dimensions
                if operand in op_use.indices:
                    # If the subview changes that dimension

                    if dim_idx in dimension_map:
                        new_operand = dimension_map[dim_idx]
                        # Update the operand, we need to add the subview offset.
                        # Case 0: Constant with value = 0, we swap it, to avoid an additional sum
                        # N.B. No constant op is removed.
                        if (
                            isinstance(operand.owner, arith.Constant)
                            and operand.op.properties["value"].value.data == 0
                        ):
                            new_dims.append(new_operand)
                        # Case 1: Constant with value != 0, we sum it
                        elif isinstance(operand.owner, arith.Constant) and isinstance(
                            new_operand, arith.Constant
                        ):
                            # Sum it, both are indices
                            new_idx = arith.Addi(
                                operand, new_operand, result_type=builtin.IndexType()
                            )
                            # Insert the new constant
                            rewriter.insert_op([new_idx], InsertPoint.insert_before(op))
                            # Add the new dimension
                            new_dims.append(new_idx)
                        # Case 2: Dynamic index, we sum it
                        else:
                            # It was a dynamic idx
                            new_idx = arith.Addi(
                                operand, new_operand, result_type=builtin.IndexType()
                            )
                            rewriter.insert_op([new_idx], InsertPoint.before(op))
                            new_dims.append(new_idx)
                    else:
                        # Leave the dimension unchanged
                        new_dims.append(operand)

                    dim_idx += 1
                elif operand.owner == op:
                    new_operands.append(op.operands[0])
                else:
                    new_operands.append(operand)
                if dim_idx == len(op.operands[0].type.get_shape()):
                    new_operands.append(new_dims)

            # Recreate the op with the new fields
            new_results = [r.type for r in op_use.results]
            new_regions = op.regions
            new_op = type(op_use)(
                operands=new_operands,
                result_types=new_results,
                properties=new_properties,
                attributes=new_attributes,
                regions=new_regions,
            )
            # Replace it in the module
            rewriter.replace_op(
                op_use,
                [new_op],
                new_op.results,
            )

        # Add the constants before the op
        if rewriter.has_done_action:
            rewriter.insert_op(
                [o for o in dimension_map.values() if isinstance(o, arith.Constant)],
                InsertPoint.before(op),
            )
            rewriter.has_done_action
        # Only if all usages are removed, we can remove the subview
        if len(op.result.uses) == 0:
            rewriter.erase_matched_op(op)


class MergeSubviewSlices(RewritePattern):
    """
    This is a simpler version of the above, working only with Load and Store.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.Subview, rewriter: PatternRewriter):
        # Check all ops using the subview,
        # Try to merge the idx of the Subview into the Loads and Stores
        # TODO: Implement this
        # NOTE: What happens if the size is different?
        # NOTE: How do i handle the offset?
        # NOTE: What if the parent is a Subview? Should not happen with folding
        # NOTE: How to handle the strides?

        # First, it should be simplified, if it was a constant, just remove it.
        # TODO : This should be optimized before
        if op.operands[0].type.shape == op.results[0].type.shape:
            rewriter.replace_matched_op([], [op.operands[0]])

        # IGNORE: Dynamic Sizes, ignored for now
        # SOLUTION: Dynamic Strides, not supported atm
        if len(op.strides) != 0:
            return
        # SOLUTION: Static strides, unsupported, return if != 1
        for e in op.static_strides.as_tuple():
            if e != 1:
                return

        # SOLUTION: Static sizes are ignored atm, return if not equal to input size
        # TREECO: We never reduce the rank
        for d1, d2 in zip(op.static_sizes.as_tuple(), op.operands[0].type.shape):
            if d1 != 1 and d1 != d2.data:
                return

        # Check that only supported OP will have the input modified
        for user in op.results[0].uses:
            if not (
                isinstance(user.operation, memref.Load)
                or isinstance(user.operation, memref.Store)
            ):
                return

        # Handle the static or dynamic offsets
        # Since this pass supports only shapes identical to the original,
        # every subview is just a slice op
        off_idx = 0
        per_dim_offsets = []
        for off in op.static_offsets.as_tuple():
            if off < 0:
                ssa_var = op.offsets[off_idx]
                per_dim_offsets.append(ssa_var)
            elif off == 0:
                per_dim_offsets.append(None)
            else:
                # Something is wrong, this should never happen, or it is not a slice
                return

        # Now modify the uses
        original_uses = [*op.results[0].uses]
        for idx, user in enumerate(original_uses):
            user = user.operation
            new_offsets = list()
            original_indices = user.indices
            for off_op, off_new in zip(original_indices, per_dim_offsets):
                if off_new is not None:
                    # If this was a constant
                    new_offset = arith.Addi(operand1=off_op, operand2=off_new)
                    rewriter.insert_op(new_offset, InsertPoint.before(user))
                    new_offsets.append(new_offset)
                else:
                    new_offsets.append(off_op)

            if isinstance(user, memref.Load):
                # Merge the idx lists
                # Add the new indices

                new_op = memref.Load.get(ref=op.operands[0], indices=new_offsets)
                rewriter.replace_op(user, new_op, [new_op.results[0]])
            elif isinstance(user, memref.Store):
                new_op = memref.Store.get(
                    ref=op.operands[0], indices=new_offsets, value=user.operands[0]
                )
                rewriter.replace_op(user, new_op, [])
        rewriter.erase_op(op)


class FoldMemRefSubViewChainPass(ModulePass):
    name = "fold-subview-chain-pass"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(FoldMemRefSubViewChain()).rewrite_module(op)
        op.verify()
