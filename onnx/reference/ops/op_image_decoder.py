# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

import io

import numpy as np

from onnx.reference.op_run import OpRun


class ImageDecoder(OpRun):
    def _run(self, encoded: np.ndarray, pixel_format="RGB"):  # type: ignore
        try:
            import PIL.Image  # pylint: disable=import-outside-toplevel
        except ImportError as e:
            raise ImportError(
                "Pillow must be installed to use the reference implementation of the ImageDecoder operator"
            ) from e
        img = PIL.Image.open(io.BytesIO(encoded.tobytes()))
        if pixel_format == "BGR":
            decoded = np.array(img)[:, :, ::-1]
        elif pixel_format == "RGB":
            decoded = np.array(img)
        elif pixel_format == "Grayscale":
            img = img.convert("L")
            decoded = np.array(img)
            decoded = np.expand_dims(decoded, axis=2)  # (H, W) to (H, W, 1)
        else:
            raise ValueError(f"pixel_format={pixel_format!r} is not supported.")
        return (decoded,)
