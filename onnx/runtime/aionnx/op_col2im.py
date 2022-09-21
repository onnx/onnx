# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0913,R0914,W0221

import numpy  # type: ignore

from ..op_run import OpRun


def _get_indices(i, shape):  # type: ignore
    res = numpy.empty((len(shape),), dtype=numpy.int64)
    k = len(shape) - 1
    while k > 0:
        m = i % shape[k]
        res[k] = m
        i -= m
        i /= shape[k]
        k -= 1
    res[0] = i
    return res


def _is_out(ind, shape):  # type: ignore
    for i, s in zip(ind, shape):
        if i < 0:
            return True
        if i >= s:
            return True
    return False


def col2im_naive_implementation(data, image_shape, kernel_shape, dilations, pads, strides):  # type: ignore
    """
    Naive implementation for `col2im`.
    """
    if not isinstance(kernel_shape, tuple):
        raise TypeError(f"Unexpected type {type(kernel_shape)!r} for kernel_shape.")
    if len(data.shape) != len(kernel_shape):
        raise ValueError(f"Shape mismatch {data.shape!r} and {kernel_shape!r}.")
    n_dims = len(pads) // 2
    new_pads = numpy.array([(pads[i], pads[i + n_dims]) for i in range(n_dims)])

    img = numpy.zeros(image_shape, dtype=data.dtype)
    kernel_size = numpy.prod(kernel_shape)
    data_size = numpy.prod(data.shape)
    for i in range(data_size):
        i_data = _get_indices(i, data.shape)
        t_data = tuple(i_data)
        for j in range(kernel_size):
            i_kernel = _get_indices(j, kernel_shape)
            t_kernel = tuple(i_kernel)

            i_img = i_data * strides - new_pads[:, 0] + i_kernel * dilations
            t_img = tuple(i_img)
            if not _is_out(t_img, img.shape):
                img[t_img] += data[tuple(t_data)]
    return img


class Col2Im(OpRun):
    def _run(self, data, image_shape, block_shape, dilations=None, pads=None, strides=None):  # type: ignore
        dilations = dilations or getattr(self, "dilations", None)
        pads = pads or getattr(self, "pads", None)
        strides = strides or getattr(self, "strides", None)

        if dilations is None:
            dilations = [1 for s in image_shape]
        if pads is None:
            pads = [0 for s in image_shape] * 2
        if strides is None:
            strides = [1 for s in image_shape]

        bl = numpy.prod(block_shape)
        C = data.shape[1] // bl
        data = data.reshape(data.shape[:1] + (C,) + (bl,) + data.shape[2:])

        ks = tuple(block_shape)
        res = None
        for n in range(data.shape[0]):
            for c in range(data.shape[1]):
                out = col2im_naive_implementation(
                    data[n, c, ...], image_shape, ks, dilations, pads, strides
                )
                if res is None:
                    new_shape = data.shape[:2] + out.shape
                    res = numpy.empty(new_shape, dtype=data.dtype)
                res[n, c, ...] = out
        return (res,)  # type: ignore
