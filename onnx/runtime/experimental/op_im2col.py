# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0913,W0221

import numpy  # type: ignore

from ._op_run_experimental import OpRunExperimental


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


def im2col_naive_implementation(data, kernel_shape, fill_value=0):  # type: ignore
    """
    Naive implementation for `im2col` (but with `padding=1`).

    :param image: image (float)
    :param kernel_shape: kernel shape
    :param fill_value: fill value
    :return: result
    """
    if not isinstance(kernel_shape, tuple):
        raise TypeError(f"Unexpected type {type(kernel_shape)!r} for kernel_shape.")
    if len(data.shape) != len(kernel_shape):
        raise ValueError(f"Shape mismatch {data.shape!r} and {kernel_shape!r}.")
    output_shape = data.shape + kernel_shape
    res = numpy.empty(output_shape, dtype=data.dtype)
    middle = numpy.array([-m / 2 for m in kernel_shape], dtype=numpy.int64)
    kernel_size = numpy.prod(kernel_shape)
    data_size = numpy.prod(data.shape)
    for i in range(data_size):
        for j in range(kernel_size):
            i_data = _get_indices(i, data.shape)
            i_kernel = _get_indices(j, kernel_shape)
            ind = i_data + i_kernel + middle
            t_data = tuple(i_data)
            t_kernel = tuple(i_kernel)
            i_out = t_data + t_kernel
            res[i_out] = fill_value if _is_out(ind, data.shape) else data[tuple(ind)]
    return res


class Im2Col(OpRunExperimental):
    def _run(self, img, kernel_shape, dilations=None, pads=None, strides=None):  # type: ignore
        dilations = dilations or getattr(self, "dilations", None)
        pads = pads or getattr(self, "pads", None)
        strides = strides or getattr(self, "strides", None)

        if dilations is None:
            dilations = [1 for s in img.shape[2:]]
        if pads is None:
            pads = [0 for s in img.shape[2:]] * 2
        if strides is None:
            strides = [1 for s in img.shape[2:]]

        if dilations[0] != 1 or min(dilations) != max(dilations):
            # Let's compute the dilated kernel.
            nd = len(dilations)
            new_kernel_shape = []
            for i, d in enumerate(dilations):
                di = len(kernel_shape) - nd + i
                new_kernel_shape.append(
                    kernel_shape[i] + (kernel_shape[i] - 1) * (d - 1)
                )
            kernel_shape = new_kernel_shape

            if pads[0] != 1 or min(pads) != max(pads):
                raise RuntimeError(
                    f"Not yet implemented for padding != 1 (pads={pads})."
                )
            if strides[0] != 1 or min(strides) != max(strides):
                raise RuntimeError(
                    f"Not yet implemented for strides != 1 (strides={strides})."
                )

        ks = tuple(kernel_shape[2:])
        res = None
        for n in range(img.shape[0]):
            for c in range(img.shape[1]):
                out = im2col_naive_implementation(img[n, c, ...], ks)
                if res is None:
                    new_shape = img.shape[:2] + out.shape
                    res = numpy.empty(new_shape, dtype=img.dtype)
                res[n, c, ...] = out
        new_shape = res.shape[: -len(ks)] + (-1,)  # type: ignore
        return (res.reshape(new_shape),)  # type: ignore
