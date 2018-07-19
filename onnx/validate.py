"""onnx checker

This implements graphalities that allows us to check whether a serialized
proto is legal.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import onnx
from onnx import defs
from onnx.defs import OpSchema
import onnx.shape_inference

import argparse

def main():  # type: () -> None
    parser = argparse.ArgumentParser(
        description='Validates ONNX and ONNX-ML model files.')
    parser.add_argument('-m', '--ml', action='store_true', help='ML mode')
    parser.add_argument('files', nargs='*',
                        help='list of ONNX files ')
    args = parser.parse_args()

    for file in args.files:
        try:
            m = onnx.load(file)
            print('\n==== Validating ' + file + ' ====\n')
            onnx.checker.check_model(m)
        except onnx.onnx_cpp2py_export.checker.ValidationError as error:
            print(str(error))
        else:
            print('No errors found.')

    print('\n')

if __name__ == '__main__':
    main()
