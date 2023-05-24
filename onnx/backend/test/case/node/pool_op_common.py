# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import itertools
from typing import Sequence, Tuple

import numpy as np


def get_pad_shape(
    auto_pad: str,
    input_spatial_shape: Sequence[int],
    kernel_spatial_shape: Sequence[int],
    strides_spatial: Sequence[int],
    output_spatial_shape: Sequence[int],
) -> Sequence[int]:
    pad_shape = [0] * len(input_spatial_shape)
    if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
        for i in range(len(input_spatial_shape)):
            pad_shape[i] = (
                (output_spatial_shape[i] - 1) * strides_spatial[i]
                + kernel_spatial_shape[i]
                - input_spatial_shape[i]
            )
    elif auto_pad == "VALID":
        pass
    return pad_shape

# Use pads to calculate output shape. Use output shape in turn to calculate the actual pads
# that are used to pad the input tensor so that computation in pool() will not cause out of bound error.
def get_output_shape_update_pads(
    pads_spatial_shape: Sequence[int] | None,
    input_spatial_shape: Sequence[int],
    dilations: Sequence[int],
    kernel_spatial_shape: Sequence[int],
    strides_spatial: Sequence[int],
    ceil_mode: bool,
) -> Tuple[Sequence[int], Sequence[int]]:
    
    # compute output shape according to:
    # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html?highlight=max+pool#torch.nn.MaxPool1d
    output_spatial_shape = [0] * len(input_spatial_shape)
    dims = len(input_spatial_shape)
    for dim in range(dims):
        dim_size = (
            input_spatial_shape[dim] +
            pads_spatial_shape[dim] +
            pads_spatial_shape[dims + dim] -
            dilations[dim] * (kernel_spatial_shape[dim] - 1) - 1
        ) / strides_spatial[dim] + 1
        
        if ceil_mode:
            output_spatial_shape[dim] = int(np.ceil(dim_size))
        else:
            output_spatial_shape[dim] = int(np.floor(dim_size))
    # import numpy as np
    # import torch
    # from torch import nn
    # m = nn.MaxPool1d(2, stride=4, padding=0, dilation=3, ceil_mode=True)
    # input = torch.from_numpy(np.array([[[1, 2, 3, 4, 0]]])).float()
    # output = m(input)
    # output
    # tensor([[[4., 0.]]])
    # it shows that if extra padding (pad = 3 here) is needed, it is added to the right (tail) side of the input tensor.
    pads_spatial_shape_new = pads_spatial_shape[:]
    for dim in range(dims):
        sliding_window_size = (kernel_spatial_shape[dim] - 1) * dilations[dim] + 1
        actual_padded_input_size = (output_spatial_shape[dim] - 1) * strides_spatial[dim] + sliding_window_size
        extra_pad = actual_padded_input_size - input_spatial_shape[dim] - pads_spatial_shape[dim] - pads_spatial_shape[dims + dim]
        if extra_pad > 0:
            pads_spatial_shape_new[dims + dim] += extra_pad

    return output_spatial_shape, pads_spatial_shape_new    


def get_output_shape(
    auto_pad: str,
    input_spatial_shape: Sequence[int],
    kernel_spatial_shape: Sequence[int],
    strides_spatial: Sequence[int],
) -> Sequence[int]:
    out_shape = [0] * len(input_spatial_shape)
    if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
        for i in range(len(input_spatial_shape)):
            out_shape[i] = int(
                np.ceil(float(input_spatial_shape[i]) / float(strides_spatial[i]))
            )
    elif auto_pad == "VALID":
        for i in range(len(input_spatial_shape)):
            out_shape[i] = int(
                np.ceil(
                    float(input_spatial_shape[i] - (kernel_spatial_shape[i] - 1))
                    / float(strides_spatial[i])
                )
            )
    return out_shape


def lp_pool(x: np.array, p: int) -> float:
    y = 0
    for v in np.nditer(x):
        y += abs(v) ** p
    return y ** (1.0 / p)


def pool(
    padded: np.ndarray,
    x_shape: Sequence[int],
    kernel_shape: Sequence[int],
    strides_shape: Sequence[int],
    out_shape: Sequence[int],
    pads: Sequence[int] | None = None,
    pooling_type: str | None = None,
    count_include_pad: int = 0,
    dilations: Sequence[int] | None = None,
    p: int = 1,
) -> np.ndarray:
    spatial_size = len(x_shape) - 2
    y = np.zeros([x_shape[0], x_shape[1], *list(out_shape)])
    if dilations is None:
        dilations = np.ones([spatial_size * 2], dtype=np.int64)
    if pads is None:
        pads = np.zeros([spatial_size * 2], dtype=np.int64)
    elif len(pads) == 1:
        pads = pads * spatial_size * 2

    def lp_pool_p(x):
        return lp_pool(x, p)

    for shape in itertools.product(
        range(x_shape[0]),
        range(x_shape[1]),
        *[
            range(
                int(
                    (x_shape[i + 2] + pads[i] + pads[i + spatial_size] - (1 + (kernel_shape[i] - 1) * dilations[i]) ) / strides_shape[i]
                    + 1
                )
            )
            for i in range(spatial_size)
        ],
    ):
        window = padded[shape[0], shape[1]]
        window_vals = np.array(
            [
                window[i]
                for i in list(
                    itertools.product(
                        *[
                            range(
                                strides_shape[i] * shape[i + 2],
                                strides_shape[i] * shape[i + 2] + (1 + (kernel_shape[i]  - 1) * dilations[i]),
                                dilations[i],
                            )
                            for i in range(spatial_size)
                        ]
                    )
                )
            ]
        )
        if pooling_type == "AVG":
            f = np.average
        elif pooling_type == "MAX":
            f = np.max
        elif pooling_type == "LPPOOL":
            f = lp_pool_p
        else:
            raise NotImplementedError(
                f"Pooling type {pooling_type} does not support. Should be AVG, MAX"
            )

        if count_include_pad == 1 and (
            pooling_type == "AVG" or pooling_type == "LPPOOL"
        ):
            y[shape] = f(window_vals)
        else:
            y[shape] = f(window_vals[np.where(~np.isnan(window_vals))])
    return y.astype(np.float32)
