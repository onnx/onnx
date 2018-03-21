import numpy as np
import itertools


def get_pad_shape(auto_pad, input_spatial_shape, kernel_spatial_shape, strides_spatial, output_spatial_shape):
    pad_shape = [0] * len(input_spatial_shape)
    if auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
        for i in range(len(input_spatial_shape)):
            pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial[i] + kernel_spatial_shape[i] - \
                input_spatial_shape[i]
    elif auto_pad == 'VALID':
        pass
    return pad_shape


def get_output_shape(auto_pad, input_spatial_shape, kernel_spatial_shape, strides_spatial):
    out_shape = [0] * len(input_spatial_shape)
    if auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
        for i in range(len(input_spatial_shape)):
            out_shape[i] = int(np.ceil(float(input_spatial_shape[i]) / float(strides_spatial[i])))
    elif auto_pad == 'VALID':
        for i in range(len(input_spatial_shape)):
            out_shape[i] = int(
                np.ceil(float(input_spatial_shape[i] - (kernel_spatial_shape[i] - 1)) / float(strides_spatial[i])))
    return out_shape


def pool(padded, x_shape, kernel_shape, strides_shape, out_shape, pad_shape, pooling_type):
    spatial_size = len(x_shape) - 2
    y = np.zeros([x_shape[0], x_shape[1]] + list(out_shape))

    for shape in itertools.product(range(x_shape[0]),
                                   range(x_shape[1]),
                                   *[range(
                                       int((x_shape[i + 2] + pad_shape[i] - kernel_shape[i]) / strides_shape[i] + 1))
                                       for i in range(spatial_size)]):
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
            raise NotImplementedError('Pooling type {} does not support. Should be AVG, MAX'.format(pooling_type))
        y[shape] = f(window_vals[np.where(~np.isnan(window_vals))])
    return y.astype(np.float32)
