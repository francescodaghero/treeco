import numpy as np
from typing import Mapping
from treeco.dialects import emitc
from xdsl.dialects.builtin import (
    ArrayAttr,
    StringAttr,
    IntegerType,
    Float32Type,
    Signedness,
    FloatAttr,
    IntegerAttr,
    IndexType,
    DenseIntOrFPElementsAttr,
    DenseResourceAttr,
    TensorType,
    f32,
)


def convert_np_to_emitcarray(np_array: np.ndarray):
    out_val = ArrayAttr([FloatAttr(i.item(), f32) for i in np_array.reshape(-1)])
    return out_val


def convert_np_to_arrayattr(np_array: np.ndarray, is_index: bool = False) -> ArrayAttr:
    tipo = np_array.dtype.name

    if is_index:
        xdsl_attr = IntegerAttr
        xdsl_type = IndexType()
    elif "int" in tipo:
        xdsl_type = IntegerType(
            np_array.dtype.itemsize * 4,
            signedness=(
                Signedness.SIGNED if tipo.startswith("i") else Signedness.UNSIGNED
            ),
        )
        xdsl_attr = IntegerAttr
    elif "float" in tipo:
        # The only supported atm
        xdsl_type = Float32Type()
        xdsl_attr = FloatAttr
    elif "str" in tipo:
        xdsl_type = None
        xdsl_attr = StringAttr
    else:
        raise ValueError(f"Unsupported type {tipo}")

    if xdsl_type is not None:
        out = ArrayAttr([xdsl_attr(i.item(), xdsl_type) for i in np_array.reshape(-1)])
    else:
        out = ArrayAttr([xdsl_attr(i.item()) for i in np_array])
    return out


def convert_np_to_tensor(
    np_array: np.ndarray, is_index: bool = False, is_signless: bool = False
):
    tipo = np_array.dtype.name
    if is_signless:
        sign = Signedness.SIGNLESS
    else:
        sign = Signedness.SIGNED if tipo.startswith("i") else Signedness.UNSIGNED

    if is_index:
        xdsl_attr = IntegerAttr
        xdsl_type = IndexType()
    elif "int" in tipo:
        xdsl_type = IntegerType(
            np_array.dtype.itemsize * 8,
            signedness=sign,
        )
        xdsl_attr = IntegerAttr
    elif "float" in tipo:
        # The only supported atm
        xdsl_type = Float32Type()
        xdsl_attr = FloatAttr
    elif "str" in tipo:
        raise ValueError(f"Unsupported type {tipo}")
        xdsl_type = None
        xdsl_attr = StringAttr
    else:
        raise ValueError(f"Unsupported type {tipo}")

    out = DenseIntOrFPElementsAttr.tensor_from_list(
        np_array.reshape(-1).tolist(), data_type=xdsl_type, shape=np_array.shape
    )

    return out
