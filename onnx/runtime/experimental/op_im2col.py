# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0913,R0914,W0221

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


def im2col_naive_implementation(data, kernel_shape, dilations, pads, strides):  # type: ignore
    """
    Naive implementation for `im2col` (but with `padding=1`).

    :param image: image (float)
    :param kernel_shape: kernel shape
    :param pads: pads
    :return: result
    """
    if not isinstance(kernel_shape, tuple):
        raise TypeError(f"Unexpected type {type(kernel_shape)!r} for kernel_shape.")
    if len(data.shape) != len(kernel_shape):
        raise ValueError(f"Shape mismatch {data.shape!r} and {kernel_shape!r}.")
    n_dims = len(pads) // 2
    new_pads = numpy.array([(pads[i], pads[i + n_dims]) for i in range(n_dims)])
    list_output_shape = list(data.shape + kernel_shape)
    for d in range(n_dims):
        kd = kernel_shape[d] + (kernel_shape[d] - 1) * (dilations[d] - 1)
        nd = int(
            ((list_output_shape[d] - kd + new_pads[d][0] + new_pads[d][1]) / strides[d])
            + 1
        )
        list_output_shape[d] = nd
    output_shape = tuple(list_output_shape)

    res = numpy.zeros(output_shape, dtype=data.dtype)
    kernel_size = numpy.prod(kernel_shape)
    res_size = numpy.prod(res.shape[:-n_dims])
    for i in range(res_size):
        i_res = _get_indices(i, res.shape[:-n_dims])
        t_res = tuple(i_res)
        for j in range(kernel_size):
            i_kernel = _get_indices(j, kernel_shape)
            t_kernel = tuple(i_kernel)

            i_img = i_res * strides - new_pads[:, 0] + i_kernel * dilations
            t_img = tuple(i_img)
            if _is_out(t_img, data.shape):
                res[t_res + t_kernel] = 0
            else:
                res[t_res + t_kernel] = data[tuple(t_img)]
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

        ks = tuple(kernel_shape[2:])
        res = None
        for n in range(img.shape[0]):
            for c in range(img.shape[1]):
                out = im2col_naive_implementation(
                    img[n, c, ...], ks, dilations, pads, strides
                )
                if res is None:
                    new_shape = img.shape[:2] + out.shape
                    res = numpy.empty(new_shape, dtype=img.dtype)
                res[n, c, ...] = out
        new_shape = res.shape[: -len(ks)] + (-1,)  # type: ignore
        return (res.reshape(new_shape),)  # type: ignore
