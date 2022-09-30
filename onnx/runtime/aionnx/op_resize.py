# SPDX-License-Identifier: Apache-2.0
# pylint: disable=C0123,R0912,R0913,R0914,W0221,W0613

import numpy as np  # type: ignore

from ..op_run import OpRun


def _cartesian(arrays, out=None):  # type: ignore
    """
    From https://stackoverflow.com/a/1235363
    Generate a cartesian product of input arrays.
    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.
    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n // arrays[0].size
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        _cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j * m : (j + 1) * m, 1:] = out[0:m, 1:]
    return out


def _nearest_coeffs(ratio, mode="round_prefer_floor"):  # type: ignore
    if type(ratio) == int or ratio.is_integer():
        return np.array([0, 1])
    if mode == "round_prefer_floor":
        return np.array([ratio <= 0.5, ratio > 0.5])
    if mode == "round_prefer_ceil":
        return np.array([ratio < 0.5, ratio >= 0.5])
    if mode == "floor":
        return np.array([1, 0])
    if mode == "ceil":
        return np.array([0, 1])
    raise ValueError(f"Unexpected value {mode!r}.")


def _cubic_coeffs(ratio, A=-0.75):  # type: ignore
    coeffs = [
        ((A * (ratio + 1) - 5 * A) * (ratio + 1) + 8 * A) * (ratio + 1) - 4 * A,
        ((A + 2) * ratio - (A + 3)) * ratio * ratio + 1,
        ((A + 2) * (1 - ratio) - (A + 3)) * (1 - ratio) * (1 - ratio) + 1,
        ((A * ((1 - ratio) + 1) - 5 * A) * ((1 - ratio) + 1) + 8 * A)
        * ((1 - ratio) + 1)
        - 4 * A,
    ]

    return np.array(coeffs)


def _linear_coeffs(ratio):  # type: ignore
    return np.array([1 - ratio, ratio])


def _get_neighbor_idxes(x, n, limit):  # type: ignore
    idxes = sorted(range(limit), key=lambda idx: (abs(x - idx), idx))[:n]
    idxes = sorted(idxes)
    return np.array(idxes)


def _get_neighbor(x, n, data):  # type: ignore
    pad_width = np.ceil(n / 2).astype(int)
    padded = np.pad(data, pad_width, mode="edge")
    x += pad_width

    idxes = _get_neighbor_idxes(x, n, len(padded))
    ret = padded[idxes]
    return idxes - pad_width, ret


def _interpolate_1d_with_x(  # type: ignore
    data,
    scale_factor,
    x,
    get_coeffs,
    roi=None,
    extrapolation_value=0.0,
    coordinate_transformation_mode="half_pixel",
    exclude_outside=False,
):

    input_width = len(data)
    output_width = scale_factor * input_width
    if coordinate_transformation_mode == "align_corners":
        if output_width == 1:
            x_ori = 0.0
        else:
            x_ori = x * (input_width - 1) / (output_width - 1)
    elif coordinate_transformation_mode == "asymmetric":
        x_ori = x / scale_factor
    elif coordinate_transformation_mode == "tf_crop_and_resize":
        if output_width == 1:
            x_ori = (roi[1] - roi[0]) * (input_width - 1) / 2
        else:
            x_ori = x * (roi[1] - roi[0]) * (input_width - 1) / (output_width - 1)
        x_ori += roi[0] * (input_width - 1)
        # Return extrapolation_value directly as what TF CropAndResize does
        if x_ori < 0 or x_ori > input_width - 1:
            return extrapolation_value
    elif coordinate_transformation_mode == "pytorch_half_pixel":
        if output_width == 1:
            x_ori = -0.5
        else:
            x_ori = (x + 0.5) / scale_factor - 0.5
    elif coordinate_transformation_mode == "half_pixel":
        x_ori = (x + 0.5) / scale_factor - 0.5
    else:
        raise ValueError(
            f"Invalid coordinate_transformation_mode: {coordinate_transformation_mode!r}."
        )
    x_ori_int = np.floor(x_ori).astype(int).item()

    # ratio must be in (0, 1] since we prefer the pixel on the left of `x_ori`
    if x_ori.is_integer():
        ratio = 1
    else:
        ratio = x_ori - x_ori_int

    coeffs = get_coeffs(ratio)
    n = len(coeffs)

    idxes, points = _get_neighbor(x_ori, n, data)

    if exclude_outside:
        for i, idx in enumerate(idxes):
            if idx < 0 or idx >= input_width:
                coeffs[i] = 0
        coeffs /= sum(coeffs)

    return np.dot(coeffs, points).item()


def _interpolate_nd_with_x(data, n, scale_factors, x, get_coeffs, roi=None, **kwargs):  # type: ignore
    if n == 1:
        return _interpolate_1d_with_x(
            data, scale_factors[0], x[0], get_coeffs, roi=roi, **kwargs
        )
    return _interpolate_1d_with_x(
        [
            _interpolate_nd_with_x(
                data[i],
                n - 1,
                scale_factors[1:],
                x[1:],
                get_coeffs,
                roi=None if roi is None else np.concatenate([roi[1:n], roi[n + 1 :]]),
                **kwargs,
            )
            for i in range(data.shape[0])
        ],
        scale_factors[0],
        x[0],
        get_coeffs,
        roi=None if roi is None else [roi[0], roi[n]],
        **kwargs,
    )


def _get_all_coords(data):  # type: ignore
    return _cartesian([list(range(data.shape[i])) for i in range(len(data.shape))])


def _interpolate_nd(  # type: ignore
    data, get_coeffs, output_size=None, scale_factors=None, roi=None, **kwargs
):

    assert output_size is not None or scale_factors is not None
    if output_size is not None:
        scale_factors = np.array(output_size) / np.array(data.shape)
    else:
        output_size = (scale_factors * np.array(data.shape)).astype(int)
    assert scale_factors is not None

    ret = np.zeros(output_size)
    for x in _get_all_coords(ret):
        ret[tuple(x)] = _interpolate_nd_with_x(
            data, len(data.shape), scale_factors, x, get_coeffs, roi=roi, **kwargs
        )
    return ret


class Resize(OpRun):
    def _run(  # type: ignore
        self,
        X,
        roi,
        scales=None,
        sizes=None,
        antialias=None,
        axes=None,
        coordinate_transformation_mode=None,
        cubic_coeff_a=None,
        exclude_outside=None,
        extrapolation_value=None,
        keep_aspect_ratio_policy=None,
        mode=None,
        nearest_mode=None,
    ):

        if mode == "nearest":  # type: ignore
            if nearest_mode is not None:
                fct = lambda x: _nearest_coeffs(x, mode=nearest_mode)  # noqa
            else:
                fct = _nearest_coeffs
        elif mode == "cubic":
            fct = _cubic_coeffs
        elif mode == "linear":
            fct = _linear_coeffs
        else:
            raise ValueError(f"Unexpected value {mode!r} for mode.")

        output = _interpolate_nd(
            X,
            fct,
            scale_factors=scales,
            output_size=sizes,
            roi=roi,
            coordinate_transformation_mode=coordinate_transformation_mode,  # type: ignore
            extrapolation_value=extrapolation_value,  # type: ignore
        ).astype(X.dtype)
        return (output,)
