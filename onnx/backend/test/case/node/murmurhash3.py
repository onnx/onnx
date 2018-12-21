from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class MurmurHash3(Base):

    @staticmethod
    def export_murmurhash3_default_seed():  # type: () -> None
        node = onnx.helper.make_node(
            'MurmurHash3',
            inputs=['X'],
            outputs=['Y']

        )
        X = np.array([3]).astype(np.int32)
        Y = np.array([847579505]).astype(np.int32)

        expect(node, inputs=[X], outputs=[Y], name='test_murmurhash3_default_seed')
        
    @staticmethod
    def export_murmurhash3_zero_seed():  # type: () -> None
        node = onnx.helper.make_node(
            'MurmurHash3',
            inputs=['X'],
            outputs=['Y'],
            seed=0

        )
        X = np.array([3]).astype(np.int32)
        Y = np.array([847579505]).astype(np.int32)

        expect(node, inputs=[X], outputs=[Y], name='test_murmurhash3_zero_seed')
        
    @staticmethod
    def export_murmurhash3_zero_seed_uint_result():  # type: () -> None
        node = onnx.helper.make_node(
            'MurmurHash3',
            inputs=['X'],
            outputs=['Y'],
            seed=0

        )
        X = np.array([3]).astype(np.int32)
        Y = np.array([847579505]).astype(np.uint32)

        expect(node, inputs=[X], outputs=[Y], name='test_murmurhash3_zero_seed_uint_result')
        
    @staticmethod
    def export_murmurhash3_zero_seed_uint_result2():  # type: () -> None
        node = onnx.helper.make_node(
            'MurmurHash3',
            inputs=['X'],
            outputs=['Y'],
            seed=0

        )
        X = np.array([4]).astype(np.int32)
        Y = np.array([1889779975]).astype(np.uint32)

        expect(node, inputs=[X], outputs=[Y], name='test_murmurhash3_zero_seed_uint_result2')
        
    @staticmethod
    def export_murmurhash3_array_data():  # type: () -> None
        node = onnx.helper.make_node(
            'MurmurHash3',
            inputs=['X'],
            outputs=['Y'],
            seed=0

        )
        X = np.array([3, 4]).astype(np.int32)
        Y = np.array([847579505, 1889779975]).astype(np.uint32)

        expect(node, inputs=[X], outputs=[Y], name='test_murmurhash3_array_data')
        
    @staticmethod
    def export_murmurhash3_non_zero_seed():  # type: () -> None
        node = onnx.helper.make_node(
            'MurmurHash3',
            inputs=['X'],
            outputs=['Y'],
            seed=42

        )
        X = np.array([3]).astype(np.int32)
        Y = np.array([-1823081949]).astype(np.int32)

        expect(node, inputs=[X], outputs=[Y], name='test_murmurhash3_non_zero_seed')
        
    @staticmethod
    def export_murmurhash3_non_zero_seed_unint_result():  # type: () -> None
        node = onnx.helper.make_node(
            'MurmurHash3',
            inputs=['X'],
            outputs=['Y'],
            seed=42

        )
        X = np.array([3]).astype(np.int32)
        Y = np.array([2471885347]).astype(np.uint32)

        expect(node, inputs=[X], outputs=[Y], name='test_murmurhash3_non_zero_seed_unint_result')