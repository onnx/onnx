# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221,R0913,R0914

import itertools
from typing import Optional, Tuple

import numpy as np  # type: ignore

from ..op_run import OpRun


def _get_pad_shape(
    auto_pad: str,
    input_spatial_shape: Tuple[int],
    kernel_spatial_shape: Tuple[int],
    strides_spatial: Tuple[int],
    output_spatial_shape: Tuple[int],
) -> Tuple[int]:
    pad_shape = [0] * len(input_spatial_shape)
    if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
        for i in range(len(input_spatial_shape)):  # pylint: disable=C0200
            pad_shape[i] = (
                (output_spatial_shape[i] - 1) * strides_spatial[i]
                + kernel_spatial_shape[i]
                - input_spatial_shape[i]
            )
    elif auto_pad == "VALID":
        pass
    if len(pad_shape) == 0:
        raise RuntimeError(
            f"Unable to compute pad shape, auto_pad={auto_pad!r}, "
            f"input_spatial_shape={input_spatial_shape!r}, "
            f"kernel_spatial_shape={kernel_spatial_shape!r}, "
            f"strides_spatial={strides_spatial!r}."
        )
    return tuple(pad_shape)  # type: ignore


def _get_output_shape_no_ceil(
    auto_pad: str,
    input_spatial_shape: Tuple[int],
    kernel_spatial_shape: Tuple[int],
    strides_spatial: Tuple[int],
) -> Tuple[int]:
    out_shape = [0] * len(input_spatial_shape)
    if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
        for i in range(len(input_spatial_shape)):  # pylint: disable=C0200
            out_shape[i] = int(
                np.ceil(float(input_spatial_shape[i]) / float(strides_spatial[i]))
            )
    elif auto_pad == "VALID":
        for i in range(len(input_spatial_shape)):  # pylint: disable=C0200
            out_shape[i] = int(
                np.ceil(
                    float(input_spatial_shape[i] - (kernel_spatial_shape[i] - 1))
                    / float(strides_spatial[i])
                )
            )
    return tuple(out_shape)  # type: ignore


def _get_output_shape(
    auto_pad: str,
    input_spatial_shape: Tuple[int],
    kernel_spatial_shape: Tuple[int],
    strides_spatial: Tuple[int],
    pad_shape: Optional[Tuple[int]] = None,
    ceil_mode: Optional[int] = 0,
) -> Tuple[int]:
    if not ceil_mode:
        out_shape = _get_output_shape_no_ceil(
            auto_pad, input_spatial_shape, kernel_spatial_shape, strides_spatial
        )
    else:
        round_fct = np.ceil if ceil_mode else np.floor
        out_shape = [0] * len(input_spatial_shape)  # type: ignore
        if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
            for i in range(len(input_spatial_shape)):  # pylint: disable=C0200
                out_shape[i] = int(  # type: ignore
                    round_fct(float(input_spatial_shape[i]) / float(strides_spatial[i]))
                )
        elif auto_pad == "VALID":
            if pad_shape is None:
                raise ValueError(  # pragma: no cogitver
                    "pad_shape cannot be None if auto_pad is "
                    "'VALID' and ceil_mode is 1."
                )
            for i in range(len(input_spatial_shape)):  # pylint: disable=C0200
                out_shape[i] = int(  # type: ignore
                    round_fct(
                        float(
                            input_spatial_shape[i]
                            + pad_shape[i]
                            - kernel_spatial_shape[i]
                        )
                        / float(strides_spatial[i])
                        + 1
                    )
                )
    if len(out_shape) == 0:
        raise RuntimeError(
            f"Unable to compute output shape, auto_pad={auto_pad!r}, "
            f"input_spatial_shape={input_spatial_shape!r}, "
            f"kernel_spatial_shape={kernel_spatial_shape!r}, "
            f"strides_spatial={strides_spatial!r}, ceil_mode={ceil_mode!r}."
        )
    if min(out_shape) <= 0:
        raise RuntimeError(
            f"output shape cannot be null or negative, out_shape={out_shape!r}, "
            f"auto_pad={auto_pad!r}, input_spatial_shape={input_spatial_shape!r}, "
            f"kernel_spatial_shape={kernel_spatial_shape!r}, "
            f"strides_spatial={strides_spatial!r}, ceil_mode={ceil_mode!r}."
        )
    return tuple(out_shape)  # type: ignore


def _pool(
    padded: np.ndarray,
    x_shape: Tuple[int],
    kernel_shape: Tuple[int],
    strides_shape: Tuple[int],
    out_shape: Tuple[int],
    pad_shape: Tuple[int],
    pooling_type: str,
    count_include_pad: Optional[int] = 0,
    ceil_mode: Optional[int] = 0,
) -> np.ndarray:
    if pooling_type == "AVG":
        fpool = np.average
    elif pooling_type == "MAX":
        fpool = np.max
    else:
        raise NotImplementedError(
            f"Pooling type {pooling_type!r} does not support. Should be AVG, MAX."
        )
    spatial_size = len(x_shape) - 2
    y = np.zeros([x_shape[0], x_shape[1]] + list(out_shape))  # type: ignore
    round_fct = np.ceil if ceil_mode else np.floor

    def loop_range():  # type: ignore
        return [
            range(
                int(
                    round_fct(
                        float(x_shape[i + 2] + pad_shape[i] - kernel_shape[i])
                        / float(strides_shape[i])
                        + 1
                    )
                )
            )
            for i in range(spatial_size)
        ]

    for shape in itertools.product(range(x_shape[0]), range(x_shape[1]), *loop_range()):  # type: ignore
        window = padded[shape[0], shape[1]]
        listi = [
            range(
                strides_shape[i] * shape[i + 2],
                strides_shape[i] * shape[i + 2] + kernel_shape[i],
            )
            for i in range(spatial_size)
        ]
        listi2 = list(itertools.product(*listi))
        values = []
        for i in listi2:
            try:
                values.append(window[i])
            except IndexError:
                continue
        window_vals = np.array(values)

        if count_include_pad == 1 and pooling_type == "AVG":
            y[shape] = fpool(window_vals)
        else:
            y[shape] = fpool(window_vals[np.where(~np.isnan(window_vals))])
    return y.astype(np.float32)


class AveragePool(OpRun):
    def _run(self, x):  # type: ignore
        # TODO: support overridden attributes.
        if self.strides is None or len(self.strides) == 0:  # type: ignore
            strides = [1] * (len(x.shape) - 2)
        else:
            strides = self.strides  # type: ignore
        kernel_shape = list(self.kernel_shape)  # type: ignore
        auto_pad = (
            "VALID" if self.auto_pad == "NOTSET" else self.auto_pad  # type: ignore
        )

        if self.pads is None or len(self.pads) == 0:  # type: ignore
            pad_shape = [0] * (len(x.shape) - 2)
            x_shape = x.shape[2:]
            padded = x
        elif len(self.pads) == 4:  # type: ignore
            pad_top, pad_bottom, pad_left, pad_right = self.pads  # type: ignore
            pad_shape = [pad_top + pad_bottom, pad_left + pad_right]
            x_shape = np.array(x.shape[2:]) + np.array(pad_shape)
            const = np.nan if self.count_include_pad == 0 else 0  # type: ignore
            padded = np.pad(
                x,
                ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
                constant_values=const,
            )
        else:
            pad_shape = self.pads  # type: ignore
            x_shape = x.shape[2:]
            padded = x

        if auto_pad in ("SAME_LOWER", "SAME_UPPER"):
            const = np.nan if self.count_include_pad == 0 else 0  # type: ignore
            out_shape = _get_output_shape(
                auto_pad, x_shape, kernel_shape, strides, pad_shape, self.ceil_mode  # type: ignore
            )
            pad_shape = _get_pad_shape(  # type: ignore
                auto_pad, x_shape, kernel_shape, strides, out_shape  # type: ignore
            )
            if auto_pad == "SAME_LOWER":
                pad_bottom = pad_shape[0] // 2
                pad_top = pad_shape[0] - pad_bottom
                pad_right = pad_shape[1] // 2
                pad_left = pad_shape[1] - pad_right
            else:
                pad_top = pad_shape[0] // 2
                pad_bottom = pad_shape[0] - pad_top
                pad_left = pad_shape[1] // 2
                pad_right = pad_shape[1] - pad_left
            padded = np.pad(
                padded,
                ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
                constant_values=const,
            )
        else:
            out_shape = _get_output_shape(
                auto_pad, x_shape, kernel_shape, strides, pad_shape, self.ceil_mode  # type: ignore
            )

        pooling_type = "AVG"
        res = _pool(
            padded,
            x.shape,
            kernel_shape,  # type: ignore
            strides,  # type: ignore
            out_shape,
            pad_shape,  # type: ignore
            pooling_type,
            count_include_pad=self.count_include_pad,  # type: ignore
            ceil_mode=self.ceil_mode,  # type: ignore
        )
        return (res,)
