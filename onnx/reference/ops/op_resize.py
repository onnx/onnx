# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
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


def _compute_x_ori(
    coordinate_transformation_mode: str,
    y: np.ndarray,
    scale_factor: float,
    input_width: int,
    output_width: float,
    output_width_int: int,
    roi: np.ndarray | list[float] | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Map output indices ``y`` to source coordinates per the chosen mode.

    Returns ``(x_ori, is_extrapolated)``. ``is_extrapolated`` is ``None`` for
    every mode except ``tf_crop_and_resize``, where it flags output positions
    whose source coordinate falls outside the input range.
    """
    is_extrapolated: np.ndarray | None = None
    if coordinate_transformation_mode == "align_corners":
        x_ori = (
            np.zeros(output_width_int, dtype=np.float64)
            if output_width == 1
            else y * (input_width - 1) / (output_width - 1)
        )
    elif coordinate_transformation_mode == "asymmetric":
        x_ori = y / scale_factor
    elif coordinate_transformation_mode == "tf_crop_and_resize":
        if roi is None:
            raise ValueError("roi cannot be None.")
        if output_width == 1:
            x_ori = np.full(
                output_width_int,
                (roi[1] - roi[0]) * (input_width - 1) / 2,
                dtype=np.float64,
            )
        else:
            x_ori = y * (roi[1] - roi[0]) * (input_width - 1) / (output_width - 1)
        x_ori = x_ori + roi[0] * (input_width - 1)
        is_extrapolated = (x_ori < 0) | (x_ori > input_width - 1)
    elif coordinate_transformation_mode == "pytorch_half_pixel":
        x_ori = (
            np.full(output_width_int, -0.5, dtype=np.float64)
            if output_width == 1
            else (y + 0.5) / scale_factor - 0.5
        )
    elif coordinate_transformation_mode == "half_pixel":
        x_ori = (y + 0.5) / scale_factor - 0.5
    elif coordinate_transformation_mode == "half_pixel_symmetric":
        adjustment = output_width_int / output_width
        center = input_width / 2
        offset = center * (1 - adjustment)
        x_ori = offset + (y + 0.5) / scale_factor - 0.5
    else:
        raise ValueError(
            f"Invalid coordinate_transformation_mode: {coordinate_transformation_mode!r}."
        )
    return x_ori, is_extrapolated


def _interpolate_1d_along_axis(
    data: np.ndarray,
    axis: int,
    scale_factor: float,
    output_width_int: int,
    get_coeffs: Callable[[float, float], np.ndarray],
    roi: np.ndarray | list[float] | None = None,
    extrapolation_value: float = 0.0,
    coordinate_transformation_mode: str = "half_pixel",
    exclude_outside: bool = False,
) -> np.ndarray:
    """Vectorized 1-D resize along a single axis.

    Computes the same result as calling :func:`_interpolate_1d_with_x` for
    every output coordinate along ``axis``, but in a single batched numpy
    operation. Resize interpolation is separable along axes, so resizing an
    N-D tensor reduces to applying this routine once per axis.
    """
    if output_width_int == 0:
        # Zero-sized output along this axis — nothing to interpolate, but
        # downstream code indexes ratios[0] / coeffs[0], so bail out early.
        empty_shape = list(data.shape)
        empty_shape[axis] = 0
        return np.empty(empty_shape, dtype=data.dtype)

    input_width = data.shape[axis]
    output_width = scale_factor * input_width
    y = np.arange(output_width_int, dtype=np.float64)

    x_ori, is_extrapolated = _compute_x_ori(
        coordinate_transformation_mode,
        y,
        scale_factor,
        input_width,
        output_width,
        output_width_int,
        roi,
    )

    x_ori_int = np.floor(x_ori).astype(np.int64)
    # Match the scalar code: prefer the pixel on the left of x_ori by
    # forcing ratio = 1 when x_ori is an exact integer.
    integer_mask = x_ori == x_ori_int
    ratios = np.where(integer_mask, 1.0, x_ori - x_ori_int)

    # Kernel size is fixed per axis: for non-antialias modes it doesn't
    # depend on ratio at all, and for antialias modes it depends only on
    # scale. Compute coefficients per output position.
    sample_coeffs = np.asarray(get_coeffs(float(ratios[0]), scale_factor))
    n = len(sample_coeffs)
    coeffs = np.empty((output_width_int, n), dtype=np.float64)
    coeffs[0] = sample_coeffs
    for i in range(1, output_width_int):
        coeffs[i] = get_coeffs(float(ratios[i]), scale_factor)

    # Replicate _get_neighbor_idxes semantics: take the n indices closest to
    # x_ori (in padded-edge coordinates) with ties broken by smaller index.
    pad_width = int(np.ceil(n / 2))
    x_padded = x_ori + pad_width
    p_padded = np.floor(x_padded).astype(np.int64)
    frac = x_padded - p_padded

    if n % 2 == 0:
        # Even kernel: window centers are half-integers. At an integer
        # x_padded, the tie between p-0.5 and p+0.5 goes to the smaller
        # center (p-0.5), so the window starts at p - n/2.
        offset_from_p = np.where(frac == 0, -(n // 2), -(n // 2) + 1)
    else:
        # Odd kernel: window centered on the nearest integer. Ties
        # (frac == 0.5) go to the smaller center.
        base = -((n - 1) // 2)
        offset_from_p = np.where(frac <= 0.5, base, base + 1)

    start_padded = p_padded + offset_from_p
    neighbor_idxes = start_padded[:, None] + np.arange(n)[None, :] - pad_width

    if exclude_outside:
        outside_mask = (neighbor_idxes < 0) | (neighbor_idxes >= input_width)
        coeffs = np.where(outside_mask, 0.0, coeffs)
        coeff_sum = coeffs.sum(axis=-1, keepdims=True)
        # Guard against rows where every coefficient is zero.
        coeff_sum = np.where(coeff_sum == 0, 1.0, coeff_sum)
        coeffs = coeffs / coeff_sum

    # Edge-padding is equivalent to clamping indices into the valid range.
    clamped = np.clip(neighbor_idxes, 0, input_width - 1)
    gathered = np.take(data, clamped, axis=axis)

    coeff_shape = (1,) * axis + coeffs.shape + (1,) * (data.ndim - axis - 1)
    weighted = gathered * coeffs.reshape(coeff_shape)
    result = weighted.sum(axis=axis + 1)

    if is_extrapolated is not None and is_extrapolated.any():
        mask_shape = (1,) * axis + (output_width_int,) + (1,) * (data.ndim - axis - 1)
        mask = is_extrapolated.reshape(mask_shape)
        result = np.where(mask, extrapolation_value, result)

    return result  # type: ignore[no-any-return]


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

    # roi is only meaningful for tf_crop_and_resize; for other modes it may
    # arrive as an empty tensor. Normalize to None so downstream indexing
    # (roi[axis], roi[axis + r]) doesn't fault on an empty array.
    if roi is not None and np.asarray(roi).size == 0:
        roi = None

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

    # Separable interpolation: resize one axis at a time. This avoids the
    # O(prod(output_shape)) scan of the old implementation, which called the
    # recursive _interpolate_nd_with_x once per output element.
    result = data if data.dtype == np.float64 else data.astype(np.float64)
    for axis in axes:
        axis_scale = float(scale_factors[axis])
        axis_output = int(output_size[axis])
        if (
            math.isclose(axis_scale, 1.0)
            and axis_output == result.shape[axis]
            and (
                roi is None
                or (
                    math.isclose(float(roi[axis]), 0.0)
                    and math.isclose(float(roi[axis + r]), 1.0)
                )
            )
        ):
            # Identity along this axis — skip to avoid unnecessary work.
            continue
        axis_roi = None if roi is None else [roi[axis], roi[axis + r]]
        result = _interpolate_1d_along_axis(
            result,
            axis,
            axis_scale,
            axis_output,
            get_coeffs,
            roi=axis_roi,
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
