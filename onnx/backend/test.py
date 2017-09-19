from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

import numpy as np

import onnx
from onnx import helper
from .test_util import create_input, N

L = 20
M = 10
S = 5
const2_np = np.random.randn(S, S)
const2_onnx = onnx.helper.make_tensor("const2",
                                      onnx.TensorProto.FLOAT,
                                      (S, S),
                                      const2_np.flatten().astype(float))

# TODO: These Numpy specs will be generally useful to backend implementations,
# so they should get moved out of here at some point
node_tests = [
  ("test_abs", N("Abs"), np.abs, [(S, S, S)]),
  ("test_add", N("Add"), np.add, [(S, S, S), (S, S, S)]),
  ("test_add_bcast", N("Add", broadcast=1), np.add, [(S, M), (M,)]),
  ("test_constant", N("Constant", value=const2_onnx), lambda: const2_np, []),
  # TODO: Are we actually supporting other dot modes?  In that case, some fancy
  # footwork is necessary...
  ("test_dot", N("Dot"), np.dot, [(S, M), (M, L)]),
  ("test_relu", N("Relu"), lambda x: np.clip(x, 0, np.inf), [(S, S, S)]),
  # TODO: Add all the other operators
]

class BackendTest(object):
    def __init__(self, backend):
        class TestsContainer(unittest.TestCase):
            def setUp(self):
                np.random.seed(seed=0)

        self.backend = backend
        self.tests = TestsContainer

        for nt in node_tests:
            self.add_node_test(*nt)

    def add_node_test(self, name, node_spec, ref, inputs):
        """
        Add A test for a single ONNX node against a reference implementation.

        Arguments:
            name (string): Eventual name of the test.  Must be prefixed
                with 'test_'.
            node (NodeSpec): The ONNX node's name and attributes to be tested;
                inputs and outputs will be inferred from other arguments of this
                spec.
            np_impl (lambda): A function from any number of Numpy ndarrays,
                to a single ndarray or tuple of ndarrays.
            inputs (tuple of ndarrays or size tuples): A specification of
                the input to the operator.
        """
        # We don't prepend the 'test_' prefix to improve greppability
        if not name.startswith('test_'):
            raise ValueError('Test name must start with test_: {}'.format(name))
        if hasattr(self.tests, name):
            raise ValueError('Duplicated test name: {}'.format(name))

        def run(test_self):
            # TODO: In some cases we should generate multiple random inputs
            # and test (ala Hypothesis)
            args = create_input(inputs)
            ref_outputs = ref(*args)
            if not isinstance(ref_outputs, tuple):
                ref_outputs = (ref_outputs,)
            input_names = ['input_{}'.format(i) for i in range(len(args))]
            output_names = ['output_{}'.format(i) for i in range(len(ref_outputs))]
            node_def = helper.make_node(
                node_spec.name,
                input_names,
                output_names,
                **node_spec.kwargs)
            outputs = self.backend.run_node(node_def, args)
            test_self.assertEqual(len(ref_outputs), len(outputs))
            for i in range(len(output_names)):
                np.testing.assert_almost_equal(
                    ref_outputs[i],
                    outputs[i],
                    decimal=4)

        setattr(self.tests, name, run)
