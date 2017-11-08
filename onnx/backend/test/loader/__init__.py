from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import glob
import json
import os

import onnx
from ..case.node import TestCase as NodeTestCase
from ..case.model import TestCase as ModelTestCase

DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(os.path.dirname(__file__))),
    'data')


def load_node_tests(data_dir=os.path.join(DATA_DIR, 'node')):
    testcases = []

    for test_name in os.listdir(data_dir):
        case_dir = os.path.join(data_dir, test_name)

        node = onnx.NodeProto()
        with open(os.path.join(case_dir, 'node.pb'), 'rb') as f:
            node.ParseFromString(f.read())

        inputs = []
        for input_file in sorted(
                glob.glob(os.path.join(case_dir, 'input_*.pb'))):
            tensor = onnx.TensorProto()
            with open(input_file, 'rb') as f:
                tensor.ParseFromString(f.read())
            inputs.append(tensor)

        outputs = []
        for output_file in sorted(
                glob.glob(os.path.join(case_dir, 'output_*.pb'))):
            tensor = onnx.TensorProto()
            with open(output_file, 'rb') as f:
                tensor.ParseFromString(f.read())
            outputs.append(tensor)

        testcases.append(
            NodeTestCase(node, inputs, outputs, test_name))

    return testcases


def load_model_tests(data_dir=os.path.join(DATA_DIR, 'model')):
    testcases = []

    for test_name in os.listdir(data_dir):
        case_dir = os.path.join(data_dir, test_name)
        with open(os.path.join(case_dir, 'data.json')) as f:
            data = json.load(f)
            url = data['url']
            model_name = data['model_name']
        testcases.append(
            ModelTestCase(test_name, url, model_name))

    return testcases
