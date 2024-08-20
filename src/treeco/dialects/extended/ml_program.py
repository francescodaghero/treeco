"""
Almost a copy of the XDSL dialect, with some changes to attributes/properties
"""

from __future__ import annotations

from typing_extensions import Self

from xdsl.dialects.builtin import (
    NoneType,
    StringAttr,
    SymbolRefAttr,
    UnitAttr,
)
from xdsl.ir import (
    Attribute,
    Dialect,
    TypeAttribute,
)
from xdsl.irdl import (
    IRDLOperation,
    opt_prop_def,
    prop_def,
    irdl_op_definition,
    result_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import SymbolOpInterface
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class Global(IRDLOperation):
    """
    Module level declaration of a global variable

    See https://mlir.llvm.org/docs/Dialects/MLProgramOps/#ml_programglobal-ml_programglobalop
    """

    name = "ml_program.global"

    sym_name = prop_def(StringAttr)
    type = prop_def(TypeAttribute)
    is_mutable = opt_prop_def(UnitAttr)
    value = opt_prop_def(Attribute)
    sym_visibility = prop_def(StringAttr)

    traits = frozenset([SymbolOpInterface()])

    def __init__(
        self,
        sym_name: Attribute,
        type: Attribute,
        is_mutable: Attribute | None,
        value: Attribute | None,
        sym_visibility: Attribute,
    ):
        super().__init__(
            properties={
                "sym_name": sym_name,
                "type": type,
                "is_mutable": is_mutable,
                "value": value,
                "sym_visibility": sym_visibility,
            },
        )

    def _verify(self) -> None:
        if isinstance(self.is_mutable, NoneType) and isinstance(self.value, NoneType):
            raise VerifyException("Immutable global must have an initial value")

    def print(self, printer: Printer):
        printer.print_string(" ")
        if self.sym_visibility:
            printer.print_string(self.sym_visibility.data)
            printer.print_string(" ")
        if self.is_mutable:
            printer.print_string("mutable ")
        printer.print_string("@")
        printer.print_string(self.sym_name.data)
        if self.value:
            printer.print_string("(")
            printer.print_attribute(self.value)
            printer.print_string(")")
        printer.print_string(" : ")
        printer.print_attribute(self.type)

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        attrs = parser.parse_optional_attr_dict()
        sym_visibility = parser.parse_visibility_keyword()

        if parser.parse_optional_keyword("mutable"):
            is_mutable = UnitAttr()
        else:
            is_mutable = None
        sym_name = parser.parse_symbol_name().data
        if parser.parse_optional_punctuation("("):
            value = parser.parse_attribute()
            parser.parse_punctuation(")")
        else:
            value = None
        parser.parse_punctuation(":")
        type = parser.parse_type()
        global_op = cls(
            StringAttr(sym_name),
            type,
            is_mutable,
            value,
            sym_visibility,
        )
        global_op.attributes |= attrs
        return global_op


@irdl_op_definition
class GlobalLoadConstant(IRDLOperation):
    """
    Direct load a constant value from a global

    See https://mlir.llvm.org/docs/Dialects/MLProgramOps/#ml_programglobal_load_const-ml_programgloballoadconstop
    """

    name = "ml_program.global_load_const"

    global_attr = prop_def(SymbolRefAttr, prop_name="global")
    result = result_def()

    def __init__(
        self,
        global_attr: Attribute,
        result_type: Attribute | None,
    ):
        super().__init__(
            properties={
                "global": global_attr,
            },
            result_types=[result_type],
        )

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print_attribute(self.global_attr)
        printer.print_string(" : ")
        printer.print_attribute(self.result.type)

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        attrs = parser.parse_optional_attr_dict()
        global_attr = parser.parse_attribute()
        parser.parse_punctuation(":")
        result_type = parser.parse_attribute()

        global_const = cls(
            global_attr,
            result_type,
        )
        global_const.attributes |= attrs
        return global_const


MLProgram = Dialect(
    "ml_program",
    [
        Global,
        GlobalLoadConstant,
    ],
)
