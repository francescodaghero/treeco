"""
Emitc dialect for XDSL.
Not all operations and traits have been implemented yet.

Note:
Since EmitC has been extended with new operators only in recent MLIR versions (>=19),
it is not compatible with the pinned version for XDSL and it requires an additional
installation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Generic, Literal, Iterable
from xdsl.ir import Dialect

from xdsl.dialects.func import FuncOpCallableInterface
from xdsl.dialects import builtin
from xdsl.dialects.builtin import (
    ShapedType,
    SymbolRefAttr,
    DictionaryAttr,
    IntAttr,
    StringAttr,
    FunctionType,
    UnitAttr,
    IntegerAttr,
    IntegerType,
    AnyIntegerAttr,
    BoolAttr,
    Signedness,
)
from xdsl.traits import OpTrait
from xdsl.dialects.builtin import ModuleOp, ContainerType
from xdsl.ir import (
    ParametrizedAttribute,
    TypeAttribute,
    Attribute,
    Operation,
    AttributeCovT,
    SSAValue,
    Region,
)
from xdsl.irdl import (
    AnyAttr,
    traits_def,
    opt_prop_def,
    opt_region_def,
    region_def,
    irdl_attr_definition,
    irdl_op_definition,
    ParameterDef,
    operand_def,
    prop_def,
    IRDLOperation,
    result_def,
    var_operand_def,
    var_result_def,
    VarOperand,
    VarOpResult,
)
from xdsl.printer import Printer
from xdsl.traits import (
    HasParent,
    IsolatedFromAbove,
    IsTerminator,
    SymbolOpInterface,
)
from xdsl.traits import (
    HasParent,
    IsTerminator,
    Pure,
    RecursiveMemoryEffect,
    SingleBlockImplicitTerminator,
)
from xdsl.ir import Block, Region, Operation
from xdsl.dialects.utils import AbstractYieldOperation

BoolType: IntegerType = IntegerType(1, Signedness.SIGNLESS)


class CExpression(OpTrait):
    def verify(self, op: Operation) -> None:
        return


@irdl_attr_definition
class ArrayType(
    Generic[AttributeCovT],
    ParametrizedAttribute,
    TypeAttribute,
    ShapedType,
    ContainerType[AttributeCovT],
):
    name = "emitc.array"

    shape: ParameterDef[builtin.ArrayAttr[IntAttr]]
    element_type: ParameterDef[AttributeCovT]

    def __init__(
        self,
        element_type: AttributeCovT,
        shape: Iterable[int | IntAttr],
    ):
        shape = builtin.ArrayAttr(
            [IntAttr(dim) if isinstance(dim, int) else dim for dim in shape]
        )
        super().__init__([shape, element_type])

    def get_num_dims(self) -> int:
        return len(self.shape.data)

    def get_shape(self) -> tuple[int, ...]:
        return tuple(i.data for i in self.shape.data)

    def get_element_type(self) -> AttributeCovT:
        return self.element_type

    def print_parameters(self, printer: Printer) -> None:
        printer.print(f"<")
        printer.print_list(
            self.shape.data,
            lambda x: printer.print(x.data) if x.data != -1 else printer.print("?"),
            "x",
        )
        if len(self.shape.data) != 0:
            printer.print("x")
        printer.print(self.element_type)
        printer.print(">")


@irdl_attr_definition
class OpaqueType(ParametrizedAttribute, TypeAttribute):
    name = "emitc.opaque"
    value: ParameterDef[StringAttr]

    @classmethod
    def parse_parameter(cls, parser) -> str:
        with parser.in_angle_brackets():
            return cls(value=parser.parse_str_literal())

    def __init__(self, value: str):
        if isinstance(value, str):
            value = StringAttr(value)
        super().__init__([value])


@irdl_attr_definition
class PointerType(ParametrizedAttribute, TypeAttribute):
    name = "emitc.ptr"
    pointee: ParameterDef[Attribute]

    def __init__(self, pointee: Attribute):
        super().__init__([pointee])


@irdl_attr_definition
class OpaqueAttr(ParametrizedAttribute):
    name = "emitc.opaque"
    value: ParameterDef[StringAttr]

    def __init__(self, value: str | StringAttr):
        if isinstance(value, str):
            value = StringAttr(value)
        super().__init__([value])

    @classmethod
    def parse_parameter(cls, parser) -> str:
        with parser.in_angle_brackets():
            return cls(value=parser.parse_str_literal())

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string("<")
        printer.print_attribute(self.value)
        printer.print_string(">")


@irdl_op_definition
class Add(IRDLOperation):
    name = "emitc.add"
    lhs = operand_def(Attribute)
    rhs = operand_def(Attribute)
    res = result_def(Attribute)

    traits = frozenset([CExpression()])

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        res = SSAValue.get(lhs).type
        super().__init__(
            operands=[lhs, rhs],
            result_types=[res],
        )


@irdl_op_definition
class Apply(IRDLOperation):
    name = "emitc.apply"
    operand = operand_def(Attribute)
    applicableOperator: StringAttr = prop_def(StringAttr)
    result = result_def(Attribute)

    traits = frozenset([CExpression()])

    def __init__(
        self, operand: SSAValue, applicableOperator: str | StringAttr, result: Attribute
    ):
        if isinstance(applicableOperator, str):
            applicableOperator = StringAttr(applicableOperator)
        super().__init__(
            operands=[operand],
            attributes=[applicableOperator],
            result_types=[result],
        )


@irdl_op_definition
class Conditional(IRDLOperation):
    name = "emitc.conditional"
    condition = operand_def(BoolAttr)
    true_value = operand_def(Attribute)
    false_value = operand_def(Attribute)

    result = result_def(Attribute)

    def __init__(
        self, condition: SSAValue, true_value: SSAValue, false_value: SSAValue
    ):
        typ = SSAValue.get(true_value).type
        super().__init__(
            operands=[condition, true_value, false_value],
            result_types=[typ],
        )


@irdl_op_definition
class Constant(IRDLOperation):
    name = "emitc.constant"
    value_ = prop_def(Attribute, prop_name="value")
    result = result_def(Attribute)

    def __init__(self, value: Attribute):
        super().__init__(
            properties={"value": value},
            result_types=[value.type],
        )


@irdl_op_definition
class DeclareFunction(IRDLOperation):
    name = "emitc.declare_function"
    sym_name = prop_def(StringAttr)

    def __init__(self, sym_name: str | StringAttr):
        if isinstance(name, str):
            name = StringAttr(name)
        super().__init__(
            attributes={"sym_name": name},
        )


@irdl_op_definition
class Div(IRDLOperation):
    name = "emitc.div"
    divisor = operand_def(Attribute)
    dividend = operand_def(Attribute)

    result = result_def(Attribute)

    def __init__(self, divisor: SSAValue, dividend: SSAValue):
        super().__init__(
            operands=[divisor, dividend],
            result_types=[dividend.type],
        )


@irdl_op_definition
class Assign(IRDLOperation):
    name = "emitc.assign"
    var = operand_def(Attribute)
    value = operand_def(Attribute)

    def __init__(self, var: SSAValue, value: SSAValue):
        super().__init__(operands=[var, value])


@irdl_op_definition
class CallOpaque(IRDLOperation):
    name = "emitc.call_opaque"
    callee = prop_def(StringAttr)
    args = prop_def(builtin.ArrayAttr[Attribute])
    operands_: VarOperand = var_operand_def()
    results_: VarOpResult = var_result_def(AnyAttr())

    def __init__(
        self,
        callee: str | StringAttr,
        args: builtin.ArrayAttr[Attribute] | Iterable[Attribute] = [],
        operands_: Sequence[SSAValue] = [],
        results_: Sequence[Attribute] = [],
    ):
        if isinstance(callee, str):
            callee = StringAttr(callee)
        properties = {"callee": callee}

        if args is not None:
            if isinstance(args, Iterable) and not isinstance(args, builtin.ArrayAttr):
                args = builtin.ArrayAttr([i for i in args])
            properties["args"] = args
        super().__init__(
            properties=properties,
            operands=[operands_],
            result_types=[results_],
        )


@irdl_op_definition
class Cast(IRDLOperation):
    name = "emitc.cast"
    operand = operand_def(Attribute)
    result = result_def(Attribute)
    # traits = frozenset([CExpression(), SameOperandsAndResultShape(), CastOp])

    def __init__(self, operand: SSAValue, result: Attribute):
        super().__init__(operands=[operand], result_types=[result])


@irdl_op_definition
class Cmp(IRDLOperation):
    name = "emitc.cmp"
    predicate: AnyIntegerAttr = prop_def(AnyIntegerAttr)
    lhs = operand_def(Attribute)
    rhs = operand_def(Attribute)
    result = result_def(BoolType)

    traits = frozenset([CExpression()])

    def __init__(self, lhs: SSAValue, rhs: SSAValue, predicate: str | int):
        if isinstance(predicate, str):
            operators = {
                "eq": 0,
                "ne": 1,
                "lt": 2,
                "le": 3,
                "gt": 4,
                "ge": 5,
                "three_way": 6,
            }
            operator = operators[predicate]

        super().__init__(
            operands=[lhs, rhs],
            properties={"predicate": IntegerAttr.from_int_and_width(operator, 64)},
            result_types=[BoolType],
        )


@irdl_op_definition
class Variable(IRDLOperation):
    name = "emitc.variable"
    value = prop_def(Attribute)
    result = result_def(Attribute)

    def __init__(self, value: Attribute):
        result_type = value.type
        super().__init__(
            properties={"value": value},
            result_types=[result_type],
        )


@irdl_op_definition
class Func(IRDLOperation):
    name = "emitc.func"
    body: Region = region_def()
    sym_name: StringAttr = prop_def(StringAttr)
    function_type: FunctionType = prop_def(FunctionType)
    specifiers: builtin.ArrayAttr = opt_prop_def(builtin.ArrayAttr[StringAttr])
    arg_attrs = opt_prop_def(builtin.ArrayAttr[DictionaryAttr])
    res_attrs = opt_prop_def(builtin.ArrayAttr[DictionaryAttr])

    traits = frozenset(
        # AutomaticALlocationScope, CallableOpInterface,
        [IsolatedFromAbove(), SymbolOpInterface(), FuncOpCallableInterface()]
    )

    def __init__(
        self,
        name: str,
        function_type: FunctionType | tuple[Sequence[Attribute], Sequence[Attribute]],
        region: Region | type[Region.DEFAULT] = Region.DEFAULT,
        specifiers: builtin.ArrayAttr | list[str] | None = None,
        *,
        arg_attrs: builtin.ArrayAttr[DictionaryAttr] | None = None,
        res_attrs: builtin.ArrayAttr[DictionaryAttr] | None = None,
    ):
        if isinstance(specifiers, list):
            specifiers = builtin.ArrayAttr([StringAttr(i) for i in specifiers])
        if isinstance(function_type, tuple):
            inputs, outputs = function_type
            function_type = FunctionType.from_lists(inputs, outputs)
        if not isinstance(region, Region):
            region = Region(Block(arg_types=function_type.inputs))
        properties: dict[str, Attribute | None] = {
            "sym_name": StringAttr(name),
            "function_type": function_type,
            "specifiers": specifiers,
            "arg_attrs": arg_attrs,
            "res_attrs": res_attrs,
        }
        super().__init__(properties=properties, regions=[region])


@irdl_op_definition
class GetGlobal(IRDLOperation):
    name = "emitc.get_global"
    name_: SymbolRefAttr = prop_def(SymbolRefAttr, prop_name="name")
    result = result_def(Attribute)

    traits = frozenset(
        [
            # AlwaysSpeculative(),
            # NoMemoryEffect(),
            # ConditionallySpeculative(),
            # SymbolOpInterface(), # Can't use it, since name!= sym_name
        ]
    )

    def __init__(self, name: str | SymbolRefAttr, return_type: Attribute):
        if isinstance(name, str):
            name = SymbolRefAttr(name)
        super().__init__(result_types=[return_type], properties={"name": name})


@irdl_op_definition
class Global(IRDLOperation):
    name = "emitc.global"
    sym_name = prop_def(StringAttr)
    type_ = prop_def(TypeAttribute, prop_name="type")
    initial_value = prop_def(Attribute)
    extern_specifier = opt_prop_def(UnitAttr)
    static_specifier = opt_prop_def(UnitAttr)
    const_specifier = opt_prop_def(UnitAttr)

    traits = frozenset([SymbolOpInterface()])

    def __init__(
        self,
        name: str | StringAttr,
        type_: Attribute,
        initial_value: Attribute,
        extern_specifier: bool = False,
        static_specifier: bool = False,
        const_specifier: bool = False,
    ):
        if isinstance(name, str):
            name = StringAttr(name)
        properties = {
            "sym_name": name,
            "type": type_,
            "initial_value": initial_value,
        }
        if extern_specifier:
            properties["extern_specifier"] = UnitAttr()
        if static_specifier:
            properties["static_specifier"] = UnitAttr()
        if const_specifier:
            properties["const_specifier"] = UnitAttr()

        super().__init__(properties=properties)


@irdl_op_definition
class If(IRDLOperation):
    name = "emitc.if"
    condition = operand_def(IntegerType(1))
    true_region: Region = region_def()
    false_region: Region = opt_region_def()

    traits = frozenset(
        [
            # SingleBlockImplicitTerminator(Yield),
            RecursiveMemoryEffect()
        ]
    )

    def __init__(
        self,
        condition: SSAValue,
        true_region: Region | Sequence[Block] | Sequence[Operation],
        false_region: Region | Sequence[Block] | Sequence[Operation] | None = None,
    ):
        if false_region is None:
            false_region = Region()
        super().__init__(operands=[condition], regions=[true_region, false_region])

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print_ssa_value(self.condition)
        printer.print_string(" ")

        printer.print_region(
            self.true_region,
            print_entry_block_args=False,
            print_empty_block=False,
            print_block_terminators=False,
        )
        printer.print_region(
            self.false_region,
            print_entry_block_args=False,
            print_empty_block=False,
            print_block_terminators=False,
        )


@irdl_op_definition
class Include(IRDLOperation):
    name = "emitc.include"
    include = prop_def(StringAttr)
    is_standard_include = opt_prop_def(UnitAttr())

    traits = frozenset([HasParent(ModuleOp)])

    def __init__(self, include: str, is_standard_include: bool = False):
        properties = {"include": StringAttr(include)}
        if is_standard_include:
            properties["is_standard_include"] = UnitAttr()
        super().__init__(properties=properties)


@irdl_op_definition
class Literal(IRDLOperation):
    name = "emitc.literal"
    value = prop_def(Attribute)
    result = result_def(Attribute)

    def __init__(self, value: Attribute, result_type: Attribute):
        super().__init__(properties={"value": value}, result_types=[result_type])


@irdl_op_definition
class LogicalAnd(IRDLOperation):
    name = "emitc.logical_and"
    lhs = operand_def(Attribute)
    rhs = operand_def(Attribute)
    result = result_def(BoolType)

    traits = frozenset([CExpression()])

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        super().__init__(
            operands=[lhs, rhs],
            result_types=[BoolType()],
        )


@irdl_op_definition
class LogicalNot(IRDLOperation):
    name = "emitc.logical_not"
    operand = operand_def(Attribute)
    result = result_def(BoolType)

    traits = frozenset([CExpression()])

    def __init__(self, operand: SSAValue):
        super().__init__(
            operands=[operand],
            result_types=[BoolType()],
        )


@irdl_op_definition
class LogicalOr(IRDLOperation):
    name = "emitc.logical_or"
    lhs = operand_def(Attribute)
    rhs = operand_def(Attribute)
    result = result_def(BoolType)

    traits = frozenset([CExpression()])

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        super().__init__(
            operands=[lhs, rhs],
            result_types=[BoolType()],
        )


@irdl_op_definition
class Mul(IRDLOperation):
    name = "emitc.mul"
    lhs = operand_def(Attribute)
    rhs = operand_def(Attribute)
    result = result_def(Attribute)

    traits = frozenset([CExpression()])

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        typ = SSAValue.get(lhs).type
        super().__init__(
            operands=[lhs, rhs],
            result_types=[typ],
        )


@irdl_op_definition
class Rem(IRDLOperation):
    name = "emitc.rem"
    lhs = operand_def(Attribute)
    rhs = operand_def(Attribute)
    result = result_def(Attribute)

    traits = frozenset([CExpression()])

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        typ = SSAValue.get(lhs).type
        super().__init__(
            operands=[lhs, rhs],
            result_types=[typ],
        )


@irdl_op_definition
class Return(IRDLOperation):
    name = "emitc.return"
    operand: VarOperand = var_operand_def(AnyAttr())
    # TODO Check if this can extend from AbstractYieldOperation

    traits = frozenset([HasParent(Func), IsTerminator()])

    def __init__(self, operand: SSAValue | None = None):
        if operand is not None:
            super().__init__(operands=[operand])
        else:
            super().__init__(operands=[])


@irdl_op_definition
class Sub(IRDLOperation):
    name = "emitc.sub"
    lhs = operand_def(Attribute)
    rhs = operand_def(Attribute)
    result = result_def(Attribute)

    traits = frozenset([CExpression()])

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        typ = SSAValue.get(lhs).type
        super().__init__(
            operands=[lhs, rhs],
            result_types=[typ],
        )


@irdl_op_definition
class Subscript(IRDLOperation):
    name = "emitc.subscript"
    value = operand_def(Attribute)
    indices = var_operand_def(AnyAttr())
    result = result_def(Attribute)

    def __init__(self, operand: SSAValue, indices: Sequence[SSAValue]):

        typ = SSAValue.get(operand).type.element_type
        super().__init__(
            operands=[operand, indices],
            result_types=[typ],
        )


@irdl_op_definition
class Verbatim(IRDLOperation):
    name = "emitc.verbatim"
    value = prop_def(StringAttr)

    def __init__(self, value: str | StringAttr):
        if isinstance(value, str):
            value = StringAttr(value)
        super().__init__(properties={"value": value})


@irdl_op_definition
class Yield(AbstractYieldOperation[Attribute]):
    name = "emitc.yield"

    traits = traits_def(lambda: frozenset([IsTerminator(), HasParent(For, If), Pure()]))


@irdl_op_definition
class For(IRDLOperation):
    name = "emitc.for"
    lowerBound = operand_def(Attribute)
    upperBound = operand_def(Attribute)
    step = operand_def(Attribute)

    body: Region = region_def("single_block")

    traits = frozenset(
        [
            SingleBlockImplicitTerminator(Yield),
            # ForOpHasCanonicalizationPatternsTrait(),
            RecursiveMemoryEffect(),
        ]
    )

    def __init__(
        self, lowerBound: SSAValue, upperBound: SSAValue, step: SSAValue, region: Region
    ):
        super().__init__(operands=[lowerBound, upperBound, step], regions=[region])

    def print(self, printer: Printer):
        block = self.body.block
        indvar, *iter_args = block.args
        printer.print_string(" ")
        printer.print_ssa_value(indvar)
        printer.print_string(" = ")
        printer.print_ssa_value(self.lowerBound)
        printer.print_string(" to ")
        printer.print_ssa_value(self.upperBound)
        printer.print_string(" step ")
        printer.print_ssa_value(self.step)
        printer.print_string(" ")

        if not isinstance(indvar.type, builtin.IndexType):
            printer.print_string(": ")
            printer.print_attribute(indvar.type)
            printer.print_string(" ")
        printer.print_region(
            self.body,
            print_entry_block_args=False,
            print_empty_block=False,
            print_block_terminators=bool(iter_args),
        )


Emitc = Dialect(
    "emitc",
    [
        Add,
        Apply,
        ArrayType,
        Cast,
        Cmp,
        Constant,
        DeclareFunction,
        Div,
        Assign,
        CallOpaque,
        GetGlobal,
        Global,
        If,
        Include,
        Literal,
        LogicalAnd,
        LogicalNot,
        LogicalOr,
        Mul,
        Rem,
        Return,
        Sub,
        Subscript,
        Verbatim,
        Yield,
        For,
        Func,
        Variable,
    ],
    [
        OpaqueType,
        OpaqueAttr,
        PointerType,
    ],
)
