# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# pylint: disable=C0123,C3001,R0912,R0913,R0914,R1730,W0221,W0613

import cv2
import numpy as np

from onnx.reference.op_run import OpRun


class ImageDecoder(OpRun):
    def _run(  # type: ignore
        self,
        encoded,
        pixel_format="RGB",
        dtype=None,
        chroma_upsampling="linear",
    ):
        if dtype is not None:
            raise RuntimeError(f"dtype={dtype!r} is not supported.")
        if chroma_upsampling != "linear":
            raise RuntimeError(
                f'Unsupported chroma_upsampling value "{chroma_upsampling}"'
            )
        decoded = None
        if pixel_format == "BGR":
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        elif pixel_format == "RGB":
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        elif pixel_format == "Grayscale":
            decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
            decoded = np.expand_dims(decoded, axis=2)  # (H, W) to (H, W, 1)
        else:
            raise RuntimeError(f"pixel_format={pixel_format!r} is not supported.")
        return (decoded,)
