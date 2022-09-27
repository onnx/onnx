# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0913,R0914,R0916,W0221

import numpy as np

from ._op_common_pool import CommonPool


class MaxPool(CommonPool):
    def _run(  # type: ignore
        self,
        x,
        auto_pad=None,
        ceil_mode=None,
        dilations=None,
        kernel_shape=None,
        pads=None,
        storage_order=None,
        strides=None,
    ):
        auto_pad = auto_pad or self.auto_pad  # type: ignore
        ceil_mode = ceil_mode or self.ceil_mode  # type: ignore
        dilations = dilations or self.dilations  # type: ignore
        strides = strides or self.strides  # type: ignore
        storage_order = storage_order or self.storage_order  # type: ignore
        kernel_shape = kernel_shape or self.kernel_shape  # type: ignore
        pads = pads or self.pads  # type: ignore
        if (
            dilations is not None
            and (min(dilations) != max(dilations) or min(dilations) != 1)
        ) or (
            strides is not None and (min(strides) != max(strides) or min(strides) != 1)
        ):
            return self._max_pool(
                x,
                auto_pad=auto_pad,
                ceil_mode=ceil_mode,
                dilations=dilations,
                kernel_shape=kernel_shape,
                pads=pads,
                storage_order=storage_order,
                strides=strides,
            )

        return CommonPool._run(
            self,
            "MAX",
            0,
            x,
            auto_pad=auto_pad,
            ceil_mode=ceil_mode,
            dilations=dilations,
            kernel_shape=kernel_shape,
            pads=pads,
            storage_order=storage_order,
            strides=strides,
        )

    def _max_pool(  # type: ignore
        self,
        x,
        auto_pad,
        ceil_mode,
        dilations,
        kernel_shape,
        pads,
        storage_order,
        strides,
    ):
        if pads is None:
            pads = [0 for i in range(len(kernel_shape) * 2)]
        if strides is None:
            strides = [1 for i in range(len(kernel_shape))]
        if dilations is None:
            dilations = [1 for i in range(len(kernel_shape))]

        n_dims = len(kernel_shape)
        new_pads = np.array([(pads[i], pads[i + n_dims]) for i in range(n_dims)])

        input_spatial_shape = x.shape[2:]
        output_spatial_shape = [0 for s in input_spatial_shape]
        if ceil_mode:
            for i in range(len(input_spatial_shape)):
                output_spatial_shape[i] = int(
                    np.ceil(
                        (
                            input_spatial_shape[i]
                            + new_pads[i].sum()
                            - ((kernel_shape[i] - 1) * dilations[i] + 1)
                        )
                        / strides[i]
                        + 1
                    )
                )
        else:
            for i in range(len(input_spatial_shape)):
                output_spatial_shape[i] = int(
                    np.floor(
                        (
                            input_spatial_shape[i]
                            + new_pads[i].sum()
                            - ((kernel_shape[i] - 1) * dilations[i] + 1)
                        )
                        / strides[i]
                        + 1
                    )
                )

        if auto_pad and auto_pad != "NOTSET":
            # Deprecated attribute
            if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
                for i in range(len(input_spatial_shape)):
                    if auto_pad == "SAME_UPPER":
                        output_spatial_shape[i] = int(
                            np.ceil(input_spatial_shape[i] / strides[i])
                        )
                    else:
                        output_spatial_shape[i] = int(
                            np.floor(input_spatial_shape[i] / strides[i])
                        )
                    pad_i = (
                        (output_spatial_shape[i] - 1) * strides[i]
                        + ((kernel_shape[i] - 1) * dilations[i] + 1)
                        - input_spatial_shape[i]
                    )
                    new_pads[i, 0] = pad_i // 2
                    new_pads[i, 1] = pad_i - new_pads[i, 0]
            else:
                for i in range(len(input_spatial_shape)):
                    output_spatial_shape[i] = int(
                        np.ceil(
                            (
                                input_spatial_shape[i]
                                - ((kernel_shape[i] - 1) * dilations[i] + 1)
                                + 1
                            )
                            / strides[i]
                        )
                    )

        if len(input_spatial_shape) == 2:
            return self._max_pool_2d(
                x,
                auto_pad,
                ceil_mode,
                dilations,
                kernel_shape,
                new_pads,
                storage_order,
                strides,
                output_spatial_shape,
            )

        raise RuntimeError(f"Not implemented yet for shape {x.shape}.")

    def _max_pool_2d(  # type: ignore
        self,
        x,
        auto_pad,
        ceil_mode,
        dilations,
        kernel_shape,
        new_pads,
        storage_order,
        strides,
        output_spatial_shape,
    ):

        global_pooling = False
        y_dims = x.shape[:2] + tuple(output_spatial_shape)
        y = np.zeros(y_dims, dtype=x.dtype)
        indices = np.full(y_dims, dtype=np.int64, fill_value=-1)
        x_dims = x.shape
        channels = x_dims[1]
        height = x_dims[2]
        width = x_dims[3] if len(kernel_shape) > 1 else 1
        pooled_height = y_dims[2]
        pooled_width = y_dims[3] if len(kernel_shape) > 1 else 1
        total_channels = x_dims[0] * channels
        stride_h = 1 if global_pooling else strides[0]
        stride_w = 1 if global_pooling else strides[1]

        x_step = height * width
        y_step = pooled_height * pooled_width
        dilation_h = dilations[0]
        dilation_w = dilations[1]

        X_data = x.ravel()
        Y_data = y.ravel()
        I_data = indices.ravel()

        def iteration(c):  # type: ignore
            x_d = c * x_step  # X_data
            y_d = c * y_step  # Y_data
            for ph in range(pooled_height):
                hstart = ph * stride_h - new_pads[0, 0]
                hend = hstart + kernel_shape[0] * dilation_h
                for pw in range(pooled_width):
                    wstart = pw * stride_w - new_pads[1, 0]
                    wend = wstart + kernel_shape[1] * dilation_w
                    pool_index = ph * pooled_width + pw
                    Yh = None
                    h_index = -1
                    w_index = -1
                    for h in range(hstart, hend, dilation_h):
                        if h < height:
                            for w in range(wstart, wend, dilation_w):
                                if w < width:
                                    input_index = h * width + w
                                    if input_index < 0 or input_index > X_data.shape[0]:
                                        continue
                                    if Yh is None or X_data[x_d + input_index] > Yh:
                                        Yh = X_data[x_d + input_index]
                                        h_index = h
                                        w_index = w
                    if Yh is None:
                        continue
                    Y_data[y_d + pool_index] = Yh
                    I_data[y_d + pool_index] = (
                        c * x_step + h_index * width + w_index
                        if storage_order == 0
                        else c * x_step + h_index + w_index * height
                    )

        for c in range(total_channels):
            iteration(c)

        if len(self.output) == 1:  # type: ignore
            return (Y_data.reshape(y_dims),)
        return (Y_data.reshape(y_dims), I_data.reshape(y_dims))
