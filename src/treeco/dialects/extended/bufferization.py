"""
This dialect is an extension of the one available in XDSL
"""

from xdsl.dialects.bufferization import *
from xdsl.printer import Printer
from xdsl.parser import Parser
from typing import Optional, Sequence, Self
from xdsl.irdl import opt_prop_def, var_result_def, VarOpResult, AnyAttr


@irdl_op_definition
class MaterializeInDestinationOp(IRDLOperation):
    name = "bufferization.materialize_in_destination"

    source = operand_def(AnyOf([TensorType, MemRefType]))
    dest = operand_def(AnyOf([TensorType, MemRefType]))
    restrict = opt_prop_def(UnitAttr)
    writable = opt_prop_def(UnitAttr)
    res: VarOpResult = var_result_def(AnyAttr())

    def __init__(
        self,
        source: SSAValue,
        dest: SSAValue,
        restrict: Optional[UnitAttr] = None,
        writable: Optional[UnitAttr] = None,
        result: Sequence[Attribute] = [],
    ):
        properties = {}
        if restrict:
            properties["restrict"] = restrict
        if writable:
            properties["writable"] = writable

        super().__init__(
            operands=[source, dest], properties=properties, result_types=[result]
        )

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print_ssa_value(self.source)
        printer.print_string(" in ")
        if self.restrict:
            printer.print_string("restrict ")
        if self.writable:
            printer.print_string("writable ")
        printer.print_ssa_value(self.dest)
        printer.print_string(" : ")
        printer.print_operation_type(self)

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        parser.parse_operand()
        source = parser.parse_characters(" in ")
        restrict = parser.parse_optional_unit_attr()
        writable = parser.parse_optional_unit_attr()
        dest = parser.parse_operand()

        cls(source, dest, restrict, writable)


Bufferization = Dialect(
    "bufferization",
    [o for o in Bufferization.operations] + [MaterializeInDestinationOp],
    [o for o in Bufferization.attributes] + [],
)
