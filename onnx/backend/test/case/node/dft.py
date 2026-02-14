# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class DFT(Base):
    @staticmethod
    def export_opset19() -> None:
        node = onnx.helper.make_node("DFT", inputs=["x"], outputs=["y"], axis=1)
        x = np.arange(0, 100).reshape(10, 10).astype(np.float32)
        y = np.fft.fft(x, axis=0)

        x = x.reshape(1, 10, 10, 1)
        y = np.stack((y.real, y.imag), axis=2).astype(np.float32).reshape(1, 10, 10, 2)
        expect(
            node,
            inputs=[x],
            outputs=[y],
            name="test_dft_opset19",
            opset_imports=[onnx.helper.make_opsetid("", 19)],
        )

        node = onnx.helper.make_node("DFT", inputs=["x"], outputs=["y"], axis=2)
        x = np.arange(0, 100).reshape(10, 10).astype(np.float32)
        y = np.fft.fft(x, axis=1)

        x = x.reshape(1, 10, 10, 1)
        y = np.stack((y.real, y.imag), axis=2).astype(np.float32).reshape(1, 10, 10, 2)
        expect(
            node,
            inputs=[x],
            outputs=[y],
            name="test_dft_axis_opset19",
            opset_imports=[onnx.helper.make_opsetid("", 19)],
        )

        node = onnx.helper.make_node(
            "DFT", inputs=["x"], outputs=["y"], inverse=1, axis=1
        )
        x = np.arange(0, 100, dtype=np.complex64).reshape(
            10,
            10,
        )
        y = np.fft.ifft(x, axis=0)

        x = np.stack((x.real, x.imag), axis=2).astype(np.float32).reshape(1, 10, 10, 2)
        y = np.stack((y.real, y.imag), axis=2).astype(np.float32).reshape(1, 10, 10, 2)
        expect(
            node,
            inputs=[x],
            outputs=[y],
            name="test_dft_inverse_opset19",
            opset_imports=[onnx.helper.make_opsetid("", 19)],
        )

        # Test RFFT (Real FFT): real input -> one-sided complex output
        node = onnx.helper.make_node(
            "DFT", inputs=["x"], outputs=["y"], onesided=1, axis=1
        )
        x = np.arange(0, 100).reshape(10, 10).astype(np.float32)
        y = np.fft.rfft(x, axis=0)

        x = x.reshape(1, 10, 10, 1)
        y = np.stack((y.real, y.imag), axis=2).astype(np.float32).reshape(1, 6, 10, 2)
        expect(
            node,
            inputs=[x],
            outputs=[y],
            name="test_dft_rfft_opset19",
            opset_imports=[onnx.helper.make_opsetid("", 19)],
        )

        # Test IRFFT (Inverse Real FFT): one-sided complex input -> real output
        node = onnx.helper.make_node(
            "DFT", inputs=["x"], outputs=["y"], onesided=1, inverse=1, axis=1
        )
        # Create one-sided complex input (6 bins for signal length 10)
        x = np.fft.rfft(np.arange(0, 100).reshape(10, 10), axis=0).astype(np.complex64)
        y = np.fft.irfft(x, n=10, axis=0)

        x = np.stack((x.real, x.imag), axis=2).astype(np.float32).reshape(1, 6, 10, 2)
        y = y.reshape(1, 10, 10, 1).astype(np.float32)
        expect(
            node,
            inputs=[x],
            outputs=[y],
            name="test_dft_irfft_opset19",
            opset_imports=[onnx.helper.make_opsetid("", 19)],
        )

    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node("DFT", inputs=["x", "", "axis"], outputs=["y"])
        x = np.arange(0, 100).reshape(10, 10).astype(np.float32)
        axis = np.array(1, dtype=np.int64)
        y = np.fft.fft(x, axis=0)

        x = x.reshape(1, 10, 10, 1)
        y = np.stack((y.real, y.imag), axis=2).astype(np.float32).reshape(1, 10, 10, 2)
        expect(node, inputs=[x, axis], outputs=[y], name="test_dft")

        node = onnx.helper.make_node("DFT", inputs=["x", "", "axis"], outputs=["y"])
        x = np.arange(0, 100).reshape(10, 10).astype(np.float32)
        axis = np.array(2, dtype=np.int64)
        y = np.fft.fft(x, axis=1)

        x = x.reshape(1, 10, 10, 1)
        y = np.stack((y.real, y.imag), axis=2).astype(np.float32).reshape(1, 10, 10, 2)
        expect(node, inputs=[x, axis], outputs=[y], name="test_dft_axis")

        node = onnx.helper.make_node(
            "DFT", inputs=["x", "", "axis"], outputs=["y"], inverse=1
        )
        x = np.arange(0, 100, dtype=np.complex64).reshape(10, 10)
        axis = np.array(1, dtype=np.int64)
        y = np.fft.ifft(x, axis=0)

        x = np.stack((x.real, x.imag), axis=2).astype(np.float32).reshape(1, 10, 10, 2)
        y = np.stack((y.real, y.imag), axis=2).astype(np.float32).reshape(1, 10, 10, 2)
        expect(node, inputs=[x, axis], outputs=[y], name="test_dft_inverse")

        # Test RFFT (Real FFT): real input -> one-sided complex output
        node = onnx.helper.make_node(
            "DFT", inputs=["x", "", "axis"], outputs=["y"], onesided=1
        )
        x = np.arange(0, 100).reshape(10, 10).astype(np.float32)
        axis = np.array(1, dtype=np.int64)
        y = np.fft.rfft(x, axis=0)

        x = x.reshape(1, 10, 10, 1)
        y = np.stack((y.real, y.imag), axis=2).astype(np.float32).reshape(1, 6, 10, 2)
        expect(node, inputs=[x, axis], outputs=[y], name="test_dft_rfft")

        # Test IRFFT (Inverse Real FFT): one-sided complex input -> real output
        node = onnx.helper.make_node(
            "DFT", inputs=["x", "", "axis"], outputs=["y"], onesided=1, inverse=1
        )
        # Create one-sided complex input (6 bins for signal length 10)
        x = np.fft.rfft(np.arange(0, 100).reshape(10, 10), axis=0).astype(np.complex64)
        axis = np.array(1, dtype=np.int64)
        y = np.fft.irfft(x, n=10, axis=0)

        x = np.stack((x.real, x.imag), axis=2).astype(np.float32).reshape(1, 6, 10, 2)
        y = y.reshape(1, 10, 10, 1).astype(np.float32)
        expect(node, inputs=[x, axis], outputs=[y], name="test_dft_irfft")
