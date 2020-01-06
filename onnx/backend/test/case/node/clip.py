from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Clip(Base):

    @staticmethod
    def export():  # type: () -> None
        signed_clip_dtypes = [
            np.int8, np.int16, np.int32, np.int64,
            np.float16, np.float32, np.float64
        ]
        for dtype in signed_clip_dtypes:
            dtype_name = np.dtype(dtype).name

            node = onnx.helper.make_node(
                'Clip',
                inputs=['x', 'min', 'max'],
                outputs=['y'],
            )

            x = np.array([-2, 0, 2]).astype(dtype)
            min_val = dtype(-1)
            max_val = dtype(1)
            y = np.clip(x, min_val, max_val)  # expected output [-1., 0., 1.]
            expect(node, inputs=[x, min_val, max_val], outputs=[y],
                   name='test_clip_example_{0}'.format(dtype_name))

            x = np.random.randn(3, 4, 5).astype(dtype)
            y = np.clip(x, min_val, max_val)
            expect(node, inputs=[x, min_val, max_val], outputs=[y],
                   name='test_clip_{0}'.format(dtype_name))
            node = onnx.helper.make_node(
                'Clip',
                inputs=['x', 'min', 'max'],
                outputs=['y'],
            )

            min_val = dtype(-5)
            max_val = dtype(5)

            x = np.array([-1, 0, 1]).astype(dtype)
            y = np.array([-1, 0, 1]).astype(dtype)
            expect(node, inputs=[x, min_val, max_val], outputs=[y],
                   name='test_clip_inbounds_{0}'.format(dtype_name))

            x = np.array([-6, 0, 6]).astype(dtype)
            y = np.array([-5, 0, 5]).astype(dtype)
            expect(node, inputs=[x, min_val, max_val], outputs=[y],
                   name='test_clip_outbounds_{0}'.format(dtype_name))

            x = np.array([-1, 0, 6]).astype(dtype)
            y = np.array([-1, 0, 5]).astype(dtype)
            expect(node, inputs=[x, min_val, max_val], outputs=[y],
                   name='test_clip_splitbounds_{0}'.format(dtype_name))

    @staticmethod
    def export_clip_default():  # type: () -> None
        clip_dtypes = [
            np.int8, np.int16, np.int32, np.int64,
            np.uint8, np.uint16, np.uint32, np.uint64,
            np.float16, np.float32, np.float64
        ]
        for dtype in clip_dtypes:
            dtype_name = np.dtype(dtype).name
            min_bound = -np.inf if np.issubdtype(dtype, np.floating) else np.iinfo(dtype).min
            max_bound = np.inf if np.issubdtype(dtype, np.floating) else np.iinfo(dtype).max

            node = onnx.helper.make_node(
                'Clip',
                inputs=['x', 'min'],
                outputs=['y'],
            )
            min_val = dtype(0)
            x = np.random.randn(3, 4, 5).astype(dtype)
            y = np.clip(x, min_val, max_bound)
            expect(node, inputs=[x, min_val], outputs=[y],
                   name='test_clip_default_min_{0}'.format(dtype_name))

            no_min = ""  # optional input, not supplied
            node = onnx.helper.make_node(
                'Clip',
                inputs=['x', no_min, 'max'],
                outputs=['y'],
            )
            max_val = dtype(0)
            x = np.random.randn(3, 4, 5).astype(dtype)
            y = np.clip(x, min_bound, max_val)
            expect(node, inputs=[x, max_val], outputs=[y],
                   name='test_clip_default_max_{0}'.format(dtype_name))

            no_max = ""  # optional input, not supplied
            node = onnx.helper.make_node(
                'Clip',
                inputs=['x', no_min, no_max],
                outputs=['y'],
            )

            # Note negative values will wrap around for unsigned types.
            x = np.array([-1, 0, 1]).astype(dtype)
            y = np.array([-1, 0, 1]).astype(dtype)
            expect(node, inputs=[x], outputs=[y],
                   name='test_clip_default_inbounds_{0}'.format(dtype_name))
