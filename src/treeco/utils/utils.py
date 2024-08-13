""" 
General utility files for the treeco package.
Contains functions that might be used across multiple modules and not 
dependant on XDSL.
"""

import numpy as np
from typing import Tuple, Literal


def quantize(
    *,
    data: np.ndarray,
    min_val: float,
    max_val: float,
    precision: Literal[8, 16, 32] = 32,
    method: str = "clip",
) -> Tuple[np.ndarray, float, float]:
    """
    Base unsigned quantization function for alphas and inputs.

    Parameters
    ----------
    data : np.ndarray
        Data to be quantized
    min_val , max_val : float
        Range of the data to be quantized
    precision : Literal[8,16,32]
        Bit width desired.
    method : str
        Function to apply after scale and zero point, default is clip

    Returns
    -------
    np.ndarray
        A quantized copy of data, with the right dtype
    float
        Scale factor
    float
        zero point
    """
    data = np.copy(data)
    qmin = 0
    qmax = 2 ** (precision) - 1
    scale = (max_val - min_val) / (2**precision - 1)
    zero_point = -(round(min_val / scale) - qmin)
    if method == "clip":
        data = np.round(data / scale + zero_point).astype(int)
    else:
        data = np.trunc(data / scale + zero_point).astype(int)
    data = np.clip(a=data, a_min=qmin, a_max=qmax).astype(np.min_scalar_type(qmax))
    return data, scale, zero_point
