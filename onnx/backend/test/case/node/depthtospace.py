# SPDX-License-Identifier: Apache-2.0

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class DepthToSpace(Base):

    @staticmethod
    def export_default_mode_example() -> None:
        node = onnx.helper.make_node(
            'DepthToSpace',
            inputs=['x'],
            outputs=['y'],
            blocksize=2,
            mode='DCR'
        )

        # (1, 8, 2, 3) input tensor
        x = np.array([[[[0., 1., 2.],
                        [3., 4., 5.]],
                       [[9., 10., 11.],
                        [12., 13., 14.]],
                       [[18., 19., 20.],
                        [21., 22., 23.]],
                       [[27., 28., 29.],
                        [30., 31., 32.]],
                       [[36., 37., 38.],
                        [39., 40., 41.]],
                       [[45., 46., 47.],
                        [48., 49., 50.]],
                       [[54., 55., 56.],
                        [57., 58., 59.]],
                       [[63., 64., 65.],
                        [66., 67., 68.]]]]).astype(np.float32)

        # (1, 2, 4, 6) output tensor
        y = np.array([[[[0., 18., 1., 19., 2., 20.],
                        [36., 54., 37., 55., 38., 56.],
                        [3., 21., 4., 22., 5., 23.],
                        [39., 57., 40., 58., 41., 59.]],
                       [[9., 27., 10., 28., 11., 29.],
                        [45., 63., 46., 64., 47., 65.],
                        [12., 30., 13., 31., 14., 32.],
                        [48., 66., 49., 67., 50., 68.]]]]).astype(np.float32)
        expect(node, inputs=[x], outputs=[y],
               name='test_depthtospace_example')

    @staticmethod
    def export_crd_mode_example() -> None:
        node = onnx.helper.make_node(
            'DepthToSpace',
            inputs=['x'],
            outputs=['y'],
            blocksize=2,
            mode='CRD'
        )

        # (1, 8, 2, 3) input tensor
        x = np.array([[[[0., 1., 2.],
                        [3., 4., 5.]],
                       [[9., 10., 11.],
                        [12., 13., 14.]],
                       [[18., 19., 20.],
                        [21., 22., 23.]],
                       [[27., 28., 29.],
                        [30., 31., 32.]],
                       [[36., 37., 38.],
                        [39., 40., 41.]],
                       [[45., 46., 47.],
                        [48., 49., 50.]],
                       [[54., 55., 56.],
                        [57., 58., 59.]],
                       [[63., 64., 65.],
                        [66., 67., 68.]]]]).astype(np.float32)

        # (1, 2, 4, 6) output tensor
        y = np.array([[[[0., 9., 1., 10., 2., 11.],
                        [18., 27., 19., 28., 20., 29.],
                        [3., 12., 4., 13., 5., 14.],
                        [21., 30., 22., 31., 23., 32.]],
                       [[36., 45., 37., 46., 38., 47.],
                        [54., 63., 55., 64., 56., 65.],
                        [39., 48., 40., 49., 41., 50.],
                        [57., 66., 58., 67., 59., 68.]]]]).astype(np.float32)
        expect(node, inputs=[x], outputs=[y],
               name='test_depthtospace_crd_mode_example')
