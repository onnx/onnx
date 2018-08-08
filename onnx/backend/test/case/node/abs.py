from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect

def generate_test(x, name):
    node = onnx.helper.make_node(
        'Abs',
        inputs=['x'],
        outputs=['y'],
    )
    y = np.abs(x)
    expect(node, inputs=[x], outputs=[y],
           name=name)

def generate_test_unsigned(x, name):
    node = onnx.helper.make_node(
        'Abs',
        inputs=['x'],
        outputs=['y'],
    )
    expect(node, inputs=[x], outputs=[x],
           name=name)

class Abs(Base):
    @staticmethod
    def export_float():  # type: () -> None
        x = np.random.randn(3, 4, 5).astype(np.float32)
        generate_test(x, 'test_abs_float')

    @staticmethod
    def export_double():  # type: () -> None
        x = np.random.randn(3, 4, 5).astype(np.float64)
        generate_test(x, 'test_abs_double')

    @staticmethod
    def export_int8():  # type: () -> None
        x = np.int8([-127,-4,0,3,127])
        generate_test(x, 'test_abs_int8')
    
    @staticmethod
    def export_uint8():  # type: () -> None
        x = np.uint8([0,1,20,255])
        generate_test_unsigned(x, 'test_abs_uint8')

    @staticmethod
    def export_int16():  # type: () -> None
        x = np.int16([-32767,-4,0,3,32767])
        generate_test(x, 'test_abs_int16')
    
    @staticmethod
    def export_uint16():  # type: () -> None
        x = np.uint16([0,1,20,65535])
        generate_test_unsigned(x, 'test_abs_uint16')

    @staticmethod
    def export_int32():  # type: () -> None
        x = np.int32([-2147483647,-4,0,3,2147483647])
        generate_test(x, 'test_abs_int32')
    
    @staticmethod
    def export_uint32():  # type: () -> None
        x = np.uint32([0,1,20,4294967295])
        generate_test_unsigned(x, 'test_abs_uint32')
        
    @staticmethod
    def export_int64():  # type: () -> None
        number_info = np.iinfo(np.int64)
        x = np.int64([-number_info.max,-4,0,3,number_info.max])
        generate_test(x, 'test_abs_int64')
    
    @staticmethod
    def export_uint64():  # type: () -> None
        number_info = np.iinfo(np.int64)
        x = np.uint64([0,1,20,number_info.max])
        generate_test_unsigned(x, 'test_abs_uint64')