# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0913,R0914,W0221

import itertools

import numpy as np  # type: ignore

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
        dilations = dilations or self.dilations  # type: ignore
        strides = strides or self.strides  # type: ignore
        storage_order = storage_order or self.storage_order  # type: ignore
        if (
            dilations is not None
            and (min(dilations) != max(dilations) or min(dilations) != 1)
        ) or (
            strides is not None and (min(strides) != max(strides) or min(strides) != 1)
        ):
            raise NotImplementedError(
                f"Only default dilations or strides are implemented for MaxPool."
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
