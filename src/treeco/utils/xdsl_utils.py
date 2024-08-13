""" 
Xdsl specific utils functions.
"""

from xdsl.ir import Operation
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation, Block
from xdsl.dialects import func, scf


def find_operation_in_module(module_op: ModuleOp, target_op: Operation) -> Operation:
    """
    Finds an operation in a module, stopping at the first instance found.

    Parameters
    ----------
    module_op : ModuleOp
        The module to search in
    target_op : Operation
        The operation to search for

    Returns
    -------
    Operation
        The first operation found of type target_op

    Raises
    -------
    LookupError
        If the operation is not found
    """
    for op in module_op.walk():
        if isinstance(op, target_op):
            return op

    raise LookupError(f"Operation {target_op} not found in module")


def find_op_in_operands_chain(op: Operation, op_type: Operation) -> Operation:
    """
    Searches in the chain of operands if an operation of type op_type is present.

    Parameters
    ----------
    op : Operation
        The starting operation for the search
    op_type : Operation
        An operation type to be searched for.

    Returns
    -------
    Operation
        The operation found

    Raises
    ------
    LookupError
        No operation of type op_type found in the chain of operands
    """
    visited = set()
    stack = [op]
    while stack:
        current_op = stack.pop()
        if isinstance(current_op, op_type):
            return current_op
        visited.add(current_op)
        for operand in current_op.operands:
            if hasattr(operand.owner, "body"):
                owner = operand.owner.body.block.last_op
            else:
                owner = operand.owner
            if isinstance(owner, Block):
                parent_op = current_op.parent_op()
                if isinstance(parent_op, ModuleOp) or isinstance(
                    parent_op, func.FuncOp
                ):
                    continue
                elif isinstance(parent_op, scf.For) or isinstance(parent_op, scf.While):
                    owner = parent_op.operands[operand.index - 1].owner

            if owner not in visited:
                stack.append(owner)

    raise LookupError(f"Operation {op_type} not found in operands chain")


def find_op_in_results_chain(op: Operation, op_type: type[Operation]) -> Operation:
    """
    Searches in the chain of results if an operation of type op_type is present.

    Parameters
    ----------
    op : Operation
        The starting operation for the search
    op_type : Operation
        An operation type to be searched for.

    Returns
    -------
    Operation
        The operation found

    Raises
    ------
    LookupError
        No operation of type op_type found in the chain of results
    """
    visited = set()

    stack = [op]
    while stack:
        current_op = stack.pop()
        if isinstance(current_op, op_type):
            return current_op
        visited.add(current_op)
        if len(current_op.results) == 0:
            continue
        for use in current_op.results[0].uses:
            owner = use.operation
            if owner not in visited:
                stack.append(owner)
    raise LookupError(f"Operation {op_type} not found in the results chain")
