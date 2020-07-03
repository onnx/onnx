from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class IFFT(Base):

    @staticmethod
    def export_dim_1d():  # type: () -> None
        node = onnx.helper.make_node(
            'FFT',
            inputs=['X'],
            outputs=['Y'],
        )

        input_data = np.array([[[1., 0.2], [0, 3], [1, 0]]], dtype=np.float64)

        # Convert to complex
        input_data_complex = input_data.view(dtype=np.complex128)[:,:,0]
        fft_result = np.fft.ifft(input_data_complex)
        expected_output = np.stack([fft_result.real, fft_result.imag], axis=2)

        expect(node, inputs=[input_data], outputs=[expected_output],
               name='test_fft_1d')

    @staticmethod
    def export_dim_2d():  # type: () -> None
        node = onnx.helper.make_node(
            'FFT',
            inputs=['X'],
            outputs=['Y'],
        )

        input_data = np.array([[[[1., 2.], [0, 0.5]], [[0, 0.7], [1.3, 0.4]]]], dtype=np.float64)

        # Convert to complex
        input_data_complex = input_data.view(dtype=np.complex128)[:,:,:,0]
        fft_result = np.fft.ifft2(input_data_complex)
        expected_output = np.stack([fft_result.real, fft_result.imag], axis=3)

        expect(node, inputs=[input_data], outputs=[expected_output],
               name='test_fft_2d')

    @staticmethod
    def export_dim_3d():  # type: () -> None
        node = onnx.helper.make_node(
            'FFT',
            inputs=['X'],
            outputs=['Y'],
        )

        input_data = np.random.randn(1,3,4,5,2).astype(np.float64)

        # Convert to complex
        input_data_complex = input_data.view(dtype=np.complex128)[:,:,:,:,0]
        fft_result = np.fft.ifftn(input_data_complex, (1, 2, 3))
        expected_output = np.stack([fft_result.real, fft_result.imag], axis=4)

        expect(node, inputs=[input_data], outputs=[expected_output],
               name='test_fft_3d')
