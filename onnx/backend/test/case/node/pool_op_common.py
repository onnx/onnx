import numpy as np  # type: ignore
import itertools
from typing import Text, Sequence


def get_pad_shape(auto_pad,  # type: Text
                  input_spatial_shape,  # type: Sequence[int]
                  kernel_spatial_shape,  # type: Sequence[int]
                  strides_spatial,  # type: Sequence[int]
                  output_spatial_shape  # type: Sequence[int]
                  ):  # type: (...) -> Sequence[int]
    pad_shape = [0] * len(input_spatial_shape)
    if auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
        for i in range(len(input_spatial_shape)):
            pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial[i] + \
                kernel_spatial_shape[i] - input_spatial_shape[i]
    elif auto_pad == 'VALID':
        pass
    return pad_shape


def get_output_shape(auto_pad,  # type: Text
                     input_spatial_shape,  # type: Sequence[int]
                     kernel_spatial_shape,  # type: Sequence[int]
                     strides_spatial  # type: Sequence[int]
                     ):  # type: (...) -> Sequence[int]
    out_shape = [0] * len(input_spatial_shape)
    if auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
        for i in range(len(input_spatial_shape)):
            out_shape[i] = int(
                np.ceil(
                    float(
                        input_spatial_shape[i])
                    / float(
                        strides_spatial[i])))
    elif auto_pad == 'VALID':
        for i in range(len(input_spatial_shape)):
            out_shape[i] = int(np.ceil(float(input_spatial_shape[i] - (kernel_spatial_shape[i] - 1)) / float(strides_spatial[i])))
    return out_shape


def pool(padded,  # type: np.ndarray
         x_shape,  # type: Sequence[int]
         kernel_shape,  # type: Sequence[int]
         strides_shape,  # type: Sequence[int]
         out_shape,  # type: Sequence[int]
         pad_shape,  # type: Sequence[int]
         pooling_type,  # type: Text
         count_include_pad=0  # type: int
         ):  # type: (...) -> np.ndarray
    spatial_size = len(x_shape) - 2
    y = np.zeros([x_shape[0], x_shape[1]] + list(out_shape))

    for shape in itertools.product(range(x_shape[0]), range(x_shape[1]), *[range(int(
            (x_shape[i + 2] + pad_shape[i] - kernel_shape[i]) / strides_shape[i] + 1)) for i in range(spatial_size)]):
        window = padded[shape[0], shape[1]]
        window_vals = np.array([window[i] for i in list(
            itertools.product(
                *[range(strides_shape[i] * shape[i + 2], strides_shape[i] * shape[i + 2] + kernel_shape[i]) for i in
                  range(spatial_size)])
        )])
        if pooling_type == 'AVG':
            f = np.average
        elif pooling_type == 'MAX':
            f = np.max
        else:
            raise NotImplementedError(
                'Pooling type {} does not support. Should be AVG, MAX'.format(pooling_type))

        if count_include_pad == 1 and pooling_type == 'AVG':
            y[shape] = f(window_vals)
        else:
            y[shape] = f(window_vals[np.where(~np.isnan(window_vals))])
    return y.astype(np.float32)
