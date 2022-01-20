from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import unittest

from onnx import checker, parser, utils

class TestFunction(unittest.TestCase):
    def test_extract_model_with_local_function(self):  # type: () -> None
        m = parser.parse_model('''
        <
        ir_version: 8,
        opset_import: [ "" : 14, "local" : 1],
        producer_name: "test",
        producer_version: "1.0",
        model_version: 1,
        doc_string: "Test"
        >
        agraph (uint8[H, W, C] x) => (uint8[H, W, C] y)
        {
            x1 = local.func(x)
            y = Identity(x1)
        }

        <
        opset_import: [ "" : 14 ],
        domain: "local",
        doc_string: "test"
        >
        func (a) => (b)
        {
            b = Identity(a)
        }
        ''')

        checker.check_model(m)
        extracted_with_funcion = utils.Extractor(m).extract_model(['x'], ['y'])
        checker.check_model(extracted_with_funcion)
        self.assertEqual(len(extracted_with_funcion.functions), 1)

        extracted_without_funcion = utils.Extractor(m).extract_model(['x1'], ['y'])
        checker.check_model(extracted_without_funcion)
        self.assertEqual(len(extracted_without_funcion.functions), 0)


if __name__ == '__main__':
    unittest.main()
