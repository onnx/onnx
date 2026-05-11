# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

import onnx
from onnx.reference.op_run import OpRun

if TYPE_CHECKING:
    from collections.abc import Callable


def _cartesian(arrays: list[np.ndarray], out: np.ndarray | None = None) -> np.ndarray:
    """From https://stackoverflow.com/a/1235363
    Generate a cartesian product of input arrays.
    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns:
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples:
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


def _nearest_coeffs(
    ratio: float | int | np.ndarray, mode: str = "round_prefer_floor"
) -> np.ndarray:
    if isinstance(ratio, int) or ratio.is_integer():
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


def _cubic_coeffs(
    ratio: float, scale: float | None = None, A: float = -0.75
) -> np.ndarray:
    del scale  # Unused
    coeffs = [
        ((A * (ratio + 1) - 5 * A) * (ratio + 1) + 8 * A) * (ratio + 1) - 4 * A,
        ((A + 2) * ratio - (A + 3)) * ratio * ratio + 1,
        ((A + 2) * (1 - ratio) - (A + 3)) * (1 - ratio) * (1 - ratio) + 1,
        ((A * ((1 - ratio) + 1) - 5 * A) * ((1 - ratio) + 1) + 8 * A)
        * ((1 - ratio) + 1)
        - 4 * A,
    ]
    return np.array(coeffs)


def _cubic_coeffs_antialias(ratio: float, scale: float, A: float = -0.75) -> np.ndarray:
    # Antialias is applied when downsampling
    scale = min(scale, 1.0)

    def compute_coeff(x: float) -> float:
        x = abs(x)
        x_2 = x * x
        x_3 = x * x_2
        if x <= 1:
            return (A + 2) * x_3 - (A + 3) * x_2 + 1
        if x < 2:
            return A * x_3 - 5 * A * x_2 + 8 * A * x - 4 * A
        return 0.0

    i_start = int(np.floor(-2 / scale) + 1)
    i_end = 2 - i_start
    args = [scale * (i - ratio) for i in range(i_start, i_end)]
    coeffs = [compute_coeff(x) for x in args]
    return np.array(coeffs) / sum(coeffs)


def _linear_coeffs(ratio: float, scale: float | None = None) -> np.ndarray:
    del scale  # unused
    return np.array([1 - ratio, ratio])


def _linear_coeffs_antialias(ratio: float, scale: float) -> np.ndarray:
    # Antialias is applied when downsampling
    scale = min(scale, 1.0)

    start = int(np.floor(-1 / scale) + 1)
    footprint = 2 - 2 * start
    args = (np.arange(start, start + footprint) - ratio) * scale
    coeffs = np.clip(1 - np.abs(args), 0, 1)
    return np.array(coeffs) / sum(coeffs)  # type: ignore[no-any-return]


def _get_neighbor_idxes(x: float, n: int, limit: int) -> np.ndarray:
    """Return the n nearest indexes to x among `[0, limit)`,
    prefer the indexes smaller than x.
    As a result, the ratio must be in `(0, 1]`.

    Examples::

        get_neighbor_idxes(4, 2, 10) == [3, 4]
        get_neighbor_idxes(4, 3, 10) == [3, 4, 5]
        get_neighbor_idxes(4.4, 3, 10) == [3, 4, 5]
        get_neighbor_idxes(4.5, 3, 10) == [3, 4, 5]
        get_neighbor_idxes(4.6, 3, 10) == [4, 5, 6]
        get_neighbor_idxes(4.4, 1, 10) == [4]
        get_neighbor_idxes(4.6, 1, 10) == [5]

    Args:
        x: float.
        n: the number of the wanted indexes.
        limit: the maximum value of index.

    Returns:
        An np.array containing n nearest indexes in ascending order
    """
    idxes = sorted(range(limit), key=lambda idx: (abs(x - idx), idx))[:n]
    idxes = sorted(idxes)
    return np.array(idxes)


def _get_neighbor(x: float, n: int, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pad `data` in 'edge' mode, and get n nearest elements in the padded array and their indexes in the original array.

    Args:
        x: Center index (in the unpadded coordinate system) of the found
            nearest elements.
        n: The number of neighbors.
        data: The array.

    Returns:
        A tuple containing the indexes of neighbor elements (the index
        can be smaller than 0 or higher than len(data)) and the value of
        these elements.
    """
    pad_width = np.ceil(n / 2).astype(int)
    padded = np.pad(data, pad_width, mode="edge")
    x += pad_width

    idxes = _get_neighbor_idxes(x, n, len(padded))
    ret = padded[idxes]
    return idxes - pad_width, ret


def _interpolate_1d_with_x(
    data: np.ndarray,
    scale_factor: float,
    output_width_int: int,
    x: float,
    get_coeffs: Callable[[float, float], np.ndarray],
    roi: np.ndarray | None = None,
    extrapolation_value: float = 0.0,
    coordinate_transformation_mode: str = "half_pixel",
    exclude_outside: bool = False,
) -> np.ndarray:
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
        if roi is None:
            raise ValueError("roi cannot be None.")
        if output_width == 1:
            x_ori = (roi[1] - roi[0]) * (input_width - 1) / 2
        else:
            x_ori = x * (roi[1] - roi[0]) * (input_width - 1) / (output_width - 1)
        x_ori += roi[0] * (input_width - 1)
        # Return extrapolation_value directly as what TF CropAndResize does
        if x_ori < 0 or x_ori > input_width - 1:
            return np.array(extrapolation_value)
    elif coordinate_transformation_mode == "pytorch_half_pixel":
        if output_width == 1:
            x_ori = -0.5
        else:
            x_ori = (x + 0.5) / scale_factor - 0.5
    elif coordinate_transformation_mode == "half_pixel":
        x_ori = (x + 0.5) / scale_factor - 0.5
    elif coordinate_transformation_mode == "half_pixel_symmetric":
        # Maps the center of the implicit ROI to the center of the output canvas.
        # The difference with `half_pixel` will be only relevant
        # when output_width_int != output_width
        adjustment = output_width_int / output_width
        center = input_width / 2
        offset = center * (1 - adjustment)
        x_ori = offset + (x + 0.5) / scale_factor - 0.5
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

    coeffs = get_coeffs(ratio, scale_factor)
    n = len(coeffs)

    idxes, points = _get_neighbor(x_ori, n, data)

    if exclude_outside:
        for i, idx in enumerate(idxes):
            if idx < 0 or idx >= input_width:
                coeffs[i] = 0
        coeffs /= sum(coeffs)

    return np.dot(coeffs, points).item()  # type: ignore[no-any-return]


def _interpolate_nd_with_x(
    data: np.ndarray,
    n: int,
    scale_factors: list[float],
    output_size: list[int],
    x: list[float],
    get_coeffs: Callable[[float, float], np.ndarray],
    roi: np.ndarray | None = None,
    exclude_outside: bool = False,
    **kwargs: Any,
) -> np.ndarray:
    if n == 1:
        return _interpolate_1d_with_x(
            data,
            scale_factors[0],
            output_size[0],
            x[0],
            get_coeffs,
            roi=roi,
            exclude_outside=exclude_outside,
            **kwargs,
        )
    res1d = []
    for i in range(data.shape[0]):
        r = _interpolate_nd_with_x(
            data[i],
            n - 1,
            scale_factors[1:],
            output_size[1:],
            x[1:],
            get_coeffs,
            roi=None if roi is None else np.concatenate([roi[1:n], roi[n + 1 :]]),
            exclude_outside=exclude_outside,
            **kwargs,
        )
        res1d.append(r)

    return _interpolate_1d_with_x(
        res1d,  # type: ignore[arg-type]  # FIXME
        scale_factors[0],
        output_size[0],
        x[0],
        get_coeffs,
        roi=None if roi is None else [roi[0], roi[n]],  # type: ignore[arg-type]  # FIXME
        exclude_outside=exclude_outside,
        **kwargs,
    )


def _get_all_coords(data: np.ndarray) -> np.ndarray:
    # FIXME: Fix input type
    return _cartesian(
        [list(range(data.shape[i])) for i in range(len(data.shape))]  # type: ignore[arg-type,misc]
    )


def _interpolate_1d_along_axis(
    data: np.ndarray,
    axis: int,
    scale_factor: float,
    output_size_axis: int,
    get_coeffs: Callable[[float, float], np.ndarray],
    roi: np.ndarray | None = None,
    extrapolation_value: float = 0.0,
    coordinate_transformation_mode: str = "half_pixel",
    exclude_outside: bool = False,
) -> np.ndarray:
    """Resize data along a single axis using vectorized operations.

    Computes coordinate mappings and interpolation coefficients for all
    output positions along the axis, then gathers neighbor values and
    computes weighted sums using vectorized NumPy operations.
    """
    input_width = data.shape[axis]
    output_width = scale_factor * input_width

    # Compute x_ori for all output positions along this axis
    x = np.arange(output_size_axis, dtype=np.float64)

    if coordinate_transformation_mode == "align_corners":
        if output_width == 1:
            x_ori = np.zeros_like(x)
        else:
            x_ori = x * (input_width - 1) / (output_width - 1)
    elif coordinate_transformation_mode == "asymmetric":
        x_ori = x / scale_factor
    elif coordinate_transformation_mode == "tf_crop_and_resize":
        if roi is None:
            raise ValueError("roi cannot be None.")
        if output_width == 1:
            x_ori = np.full_like(x, (roi[1] - roi[0]) * (input_width - 1) / 2)
        else:
            x_ori = x * (roi[1] - roi[0]) * (input_width - 1) / (output_width - 1)
        x_ori = x_ori + roi[0] * (input_width - 1)
    elif coordinate_transformation_mode == "pytorch_half_pixel":
        if output_width == 1:
            x_ori = np.full_like(x, -0.5)
        else:
            x_ori = (x + 0.5) / scale_factor - 0.5
    elif coordinate_transformation_mode == "half_pixel":
        x_ori = (x + 0.5) / scale_factor - 0.5
    elif coordinate_transformation_mode == "half_pixel_symmetric":
        adjustment = output_size_axis / output_width
        center = input_width / 2
        offset = center * (1 - adjustment)
        x_ori = offset + (x + 0.5) / scale_factor - 0.5
    else:
        raise ValueError(
            f"Invalid coordinate_transformation_mode: {coordinate_transformation_mode!r}."
        )

    # Compute ratios (in (0, 1], with 1 for integer positions)
    x_ori_int = np.floor(x_ori).astype(int)
    ratios = np.where(x_ori == x_ori_int, 1.0, x_ori - x_ori_int)

    # Compute coefficients for each output position
    coeff_list = [get_coeffs(float(r), scale_factor) for r in ratios]
    n_coeffs = len(coeff_list[0])
    coeffs_arr = np.stack(coeff_list)  # (output_size_axis, n_coeffs)

    # Compute neighbor indices in padded coordinate space
    pad_width = int(np.ceil(n_coeffs / 2))
    padded_len = input_width + 2 * pad_width

    neighbor_idxes = np.stack(
        [
            _get_neighbor_idxes(float(xo) + pad_width, n_coeffs, padded_len)
            for xo in x_ori
        ]
    )  # (output_size_axis, n_coeffs)

    # Apply exclude_outside
    if exclude_outside:
        orig_idxes = neighbor_idxes - pad_width
        outside_mask = (orig_idxes < 0) | (orig_idxes >= input_width)
        coeffs_arr = coeffs_arr.copy()
        coeffs_arr[outside_mask] = 0
        coeff_sums = coeffs_arr.sum(axis=-1, keepdims=True)
        coeffs_arr = coeffs_arr / coeff_sums

    # Identify extrapolation positions for tf_crop_and_resize
    extrap_mask = None
    if coordinate_transformation_mode == "tf_crop_and_resize":
        extrap_mask = (x_ori < 0) | (x_ori > input_width - 1)

    # Vectorized gather and weighted sum
    # Move target axis to last position
    data_t = np.moveaxis(data, axis, -1)  # (..., input_width)

    # Pad along the last axis
    pad_config = [(0, 0)] * (data_t.ndim - 1) + [(pad_width, pad_width)]
    padded = np.pad(data_t, pad_config, mode="edge")  # (..., padded_len)

    # Gather neighbors: (..., output_size_axis, n_coeffs)
    gathered = padded[..., neighbor_idxes]

    # Weighted sum: (..., output_size_axis)
    result = np.einsum("...jk,jk->...j", gathered, coeffs_arr)

    # Apply extrapolation for tf_crop_and_resize
    if extrap_mask is not None:
        result[..., extrap_mask] = extrapolation_value

    # Move axis back to original position
    return np.moveaxis(result, -1, axis)


def _interpolate_nd(
    data: np.ndarray,
    get_coeffs: Callable[[float, float], np.ndarray],
    output_size: list[int] | None = None,
    scale_factors: list[float] | None = None,
    axes: list[int] | None = None,
    roi: np.ndarray | None = None,
    keep_aspect_ratio_policy: str | None = "stretch",
    exclude_outside: bool = False,
    **kwargs: Any,
) -> np.ndarray:
    if output_size is None and scale_factors is None:
        raise ValueError("output_size is None and scale_factors is None.")

    r = len(data.shape)
    if axes is not None:
        if scale_factors is not None:
            new_scale_factors = [1.0] * r
            for i, d in enumerate(axes):
                new_scale_factors[d] = scale_factors[i]
            scale_factors = new_scale_factors

        if output_size is not None:
            new_output_size = [data.shape[i] for i in range(r)]
            for i, d in enumerate(axes):
                new_output_size[d] = output_size[i]
            output_size = new_output_size

        if roi is not None:
            new_roi = ([0.0] * r) + ([1.0] * r)
            naxes = len(axes)
            for i, d in enumerate(axes):
                new_roi[d] = roi[i]
                new_roi[r + d] = roi[naxes + i]
            roi = new_roi  # type: ignore[assignment]  # FIXME
    else:
        axes = list(range(r))

    if output_size is not None:
        scale_factors = [output_size[i] / data.shape[i] for i in range(r)]
        if keep_aspect_ratio_policy != "stretch":
            if keep_aspect_ratio_policy == "not_larger":
                scale = np.array(scale_factors)[axes].min()
            elif keep_aspect_ratio_policy == "not_smaller":
                scale = np.array(scale_factors)[axes].max()
            else:
                raise ValueError(
                    f"Invalid keep_aspect_ratio_policy={keep_aspect_ratio_policy!r}"
                )

            scale_factors = [scale if i in axes else 1.0 for i in range(r)]

            def round_half_up(x: float) -> int:
                return int(x + 0.5)

            output_size = [
                round_half_up(scale * data.shape[i]) if i in axes else data.shape[i]
                for i in range(r)
            ]

    else:
        output_size = (scale_factors * np.array(data.shape)).astype(int)  # type: ignore[union-attr]

    if scale_factors is None:
        raise ValueError("scale_factors is None.")
    if output_size is None:
        raise ValueError("output_size is None.")

    # Vectorized separable interpolation: resize one dimension at a time.
    # This replaces the element-by-element Python loop with per-axis
    # vectorized operations, reducing complexity from O(prod(output_shape))
    # Python iterations to O(ndim) vectorized passes.
    result = np.array(data, dtype=np.float64)
    for d in range(r):
        if scale_factors[d] == 1.0 and output_size[d] == result.shape[d]:
            continue
        roi_1d = None if roi is None else np.array([roi[d], roi[r + d]])
        result = _interpolate_1d_along_axis(
            result,
            d,
            scale_factors[d],
            output_size[d],
            get_coeffs,
            roi=roi_1d,
            exclude_outside=exclude_outside,
            **kwargs,
        )
    return result


class Resize(OpRun):
    def _run(
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
        mode: str | None = None,
        nearest_mode=None,
    ):
        if mode == "nearest":
            if antialias:
                raise RuntimeError(
                    f"antilias={antialias!r} is not supported for mode={mode!r}."
                )
            if nearest_mode is not None:

                def fct(x, scale_factor):
                    del scale_factor  # unused
                    return _nearest_coeffs(x, mode=nearest_mode)

            else:
                fct = _nearest_coeffs
        elif mode == "cubic":
            fct_ = _cubic_coeffs_antialias if antialias else _cubic_coeffs

            def fct(x, scale):
                return fct_(x, scale, A=cubic_coeff_a)

        elif mode == "linear":
            fct = _linear_coeffs_antialias if antialias else _linear_coeffs
        else:
            raise ValueError(f"Unexpected value {mode!r} for mode.")

        if axes is None:
            output = onnx.numpy_helper.saturate_cast(
                _interpolate_nd(
                    X,
                    fct,
                    scale_factors=scales,
                    output_size=sizes,
                    roi=roi,
                    keep_aspect_ratio_policy=keep_aspect_ratio_policy,
                    exclude_outside=exclude_outside,
                    coordinate_transformation_mode=coordinate_transformation_mode,
                    extrapolation_value=extrapolation_value,
                ),
                X.dtype,
            )
            return (output,)

        # axes is not None
        not_axes = [a for a in range(len(X.shape)) if a not in axes]
        perm = tuple(not_axes + axes)
        permuted = np.transpose(X, perm)
        new_shape = (-1, *tuple(X.shape[a] for a in axes))
        reshaped = permuted.reshape(new_shape)
        res = None
        for i in range(reshaped.shape[0]):
            output = _interpolate_nd(
                reshaped[i],
                fct,
                scale_factors=scales,
                output_size=sizes,
                roi=roi,
                keep_aspect_ratio_policy=keep_aspect_ratio_policy,
                exclude_outside=exclude_outside,
                coordinate_transformation_mode=coordinate_transformation_mode,
                extrapolation_value=extrapolation_value,
            )
            if res is None:
                res = np.empty((reshaped.shape[0], *output.shape), dtype=X.dtype)
            res[i] = onnx.numpy_helper.saturate_cast(output, X.dtype)

        res_reshaped = res.reshape(tuple(X.shape[a] for a in not_axes) + res[0].shape)
        new_perm = list(perm)
        for i, a in enumerate(perm):
            new_perm[a] = i
        final = np.transpose(res_reshaped, tuple(new_perm))
        return (final,)
