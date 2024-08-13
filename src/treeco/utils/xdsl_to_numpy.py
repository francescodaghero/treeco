import numpy as np
from typing import Mapping
from xdsl.dialects.builtin import (
    ArrayAttr,
    StringAttr,
    IntegerType,
    AnyFloat,
    IndexType,
    DenseIntOrFPElementsAttr,
)


def convert_arrayattr_to_np(array_attr):
    output = []
    for i in array_attr.data:
        if isinstance(i, StringAttr):
            output.append(i.data)
        elif hasattr(i, "data"):
            output.append(i.data)
        else:
            output.append(i.value.data)
    return np.asarray(output)


def convert_dense_to_np(dense_attr):
    output = []
    for i in dense_attr.data:
        if hasattr(i, "data"):
            output.append(i.data)
        else:
            output.append(i.value.data)

    new_dtype = convert_xdsl_to_np_type(dense_attr.type.element_type)
    return np.asarray(output, dtype=new_dtype)


def convert_attrdict_to_lists(dict_value: Mapping):
    new_dict = {}
    for k, v in dict_value.items():
        if isinstance(v, ArrayAttr):
            new_dict[k] = convert_arrayattr_to_np(v)
        elif isinstance(v, DenseIntOrFPElementsAttr):
            new_dict[k] = convert_dense_to_np(v)
        elif hasattr(v, "data"):
            new_dict[k] = v.data
        elif hasattr(v, "value"):
            new_dict[k] = v.value.data
    return new_dict


def convert_xdsl_to_np_type(tipo):
    if isinstance(tipo, IndexType):
        return np.int64
    elif isinstance(tipo, IntegerType):
        return np.dtype(
            f'{"u" if tipo.signedness.data.name == "UNSIGNED" else ""}int{tipo.width.data}'
        )
    elif isinstance(tipo, AnyFloat):
        return np.float32
    else:
        raise ValueError(f"Unsupported type {tipo}")
