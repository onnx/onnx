from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import unittest

import onnx
from onnx import checker, parser, utils


class TestFunction(unittest.TestCase):
    def test_extract_model_with_local_function(self):  # type: () -> None

        # function common
        func_domain = 'local'
        func_opset_imports = [onnx.helper.make_opsetid("", 14)]
        func_nested_opset_imports = [
            onnx.helper.make_opsetid("", 14), onnx.helper.make_opsetid(func_domain, 1)]

        # add function
        func_add_name = 'func_add'
        func_add_inputs = ['a', 'b']
        func_add_outputs = ['c']
        func_add_nodes = [onnx.helper.make_node('Add', ['a', 'b'], ['c'])]
        func_add = onnx.helper.make_function(
            func_domain,
            func_add_name,
            func_add_inputs,
            func_add_outputs,
            func_add_nodes,
            func_opset_imports)

        # identity function
        func_identity_name = 'func_identity'
        func_identity_inputs = ['a']
        func_identity_outputs = ['b']
        func_identity_nodes = [onnx.helper.make_node('Identity', ['a'], ['b'])]
        func_identity = onnx.helper.make_function(
            func_domain,
            func_identity_name,
            func_identity_inputs,
            func_identity_outputs,
            func_identity_nodes,
            func_opset_imports)

        # nested identity/add function
        func_nested_identity_add_name = 'func_nested_identity_add'
        func_nested_identity_add_inputs = ['a', 'b']
        func_nested_identity_add_outputs = ['c']
        func_nested_identity_add_nodes = [
            onnx.helper.make_node('func_identity', ['a'], ['a1'], domain=func_domain),
            onnx.helper.make_node('func_identity', ['b'], ['b1'], domain=func_domain),
            onnx.helper.make_node('func_add', ['a1', 'b1'], ['c'], domain=func_domain)]
        func_nested_identity_add = onnx.helper.make_function(
            func_domain,
            func_nested_identity_add_name,
            func_nested_identity_add_inputs,
            func_nested_identity_add_outputs,
            func_nested_identity_add_nodes,
            func_nested_opset_imports)

        # create graph nodes
        node_func_add = onnx.helper.make_node(func_add_name, ['i0', 'i1'], ['t0'], domain=func_domain)
        node_add0 = onnx.helper.make_node('Add', ['i1', 'i2'], ['t2'])
        node_add1 = onnx.helper.make_node('Add', ['t0', 't2'], ['o_func_add'])
        node_func_identity = onnx.helper.make_node(func_identity_name, ['i1'], ['t1'], domain=func_domain)
        node_identity = onnx.helper.make_node('Identity', ['i1'], ['t3'])
        node_add2 = onnx.helper.make_node('Add', ['t3', 't2'], ['o_no_func'])
        node_func_nested0 = onnx.helper.make_node(
            func_nested_identity_add_name,
            ['t0', 't1'],
            ['o_all_func0'],
            domain=func_domain)
        node_func_nested1 = onnx.helper.make_node(
            func_nested_identity_add_name,
            ['t3', 't2'],
            ['o_all_func1'],
            domain=func_domain)

        graph_name = 'graph_with_imbedded_functions'
        ir_version = 8
        opset_imports = [onnx.helper.make_opsetid("", 14), onnx.helper.make_opsetid("local", 1)]
        tensor_type_proto = onnx.helper.make_tensor_type_proto(elem_type=2, shape=[5])

        graph = onnx.helper.make_graph(
            [node_func_add, node_add0, node_add1, node_func_identity, node_identity,
            node_func_nested0, node_func_nested1, node_add2],
            graph_name,
            [
                onnx.helper.make_value_info(name='i0', type_proto=tensor_type_proto),
                onnx.helper.make_value_info(name='i1', type_proto=tensor_type_proto),
                onnx.helper.make_value_info(name='i2', type_proto=tensor_type_proto)],
            [
                onnx.helper.make_value_info(name='o_no_func', type_proto=tensor_type_proto),
                onnx.helper.make_value_info(name='o_func_add', type_proto=tensor_type_proto),
                onnx.helper.make_value_info(name='o_all_func0', type_proto=tensor_type_proto),
                onnx.helper.make_value_info(name='o_all_func1', type_proto=tensor_type_proto)])

        meta = {
            'ir_version': ir_version,
            'opset_imports': opset_imports,
            'producer_name': 'test_extract_model_with_local_function',
            'functions': [func_identity, func_add, func_nested_identity_add],
        }
        model = onnx.helper.make_model(graph, **meta)

        checker.check_model(model)
        extracted_with_no_funcion = utils.Extractor(model).extract_model(['i0', 'i1', 'i2'], ['o_no_func'])
        checker.check_model(extracted_with_no_funcion)
        self.assertEqual(len(extracted_with_no_funcion.functions), 0)

        extracted_with_add_funcion = utils.Extractor(model).extract_model(['i0', 'i1', 'i2'], ['o_func_add'])
        checker.check_model(extracted_with_add_funcion)
        self.assertEqual(len(extracted_with_add_funcion.functions), 1)
        self.assertTrue(extracted_with_add_funcion.functions[0].name == func_add_name)

        extracted_with_o_all_funcion0 = utils.Extractor(model).extract_model(['i0', 'i1', 'i2'], ['o_all_func0'])
        checker.check_model(extracted_with_o_all_funcion0)
        self.assertEqual(len(extracted_with_o_all_funcion0.functions), 3)
        self.assertIsNotNone(
            next((f for f in extracted_with_o_all_funcion0.functions
            if f.name == func_add_name and f.domain == func_domain), None))
        self.assertIsNotNone(
            next((f for f in extracted_with_o_all_funcion0.functions
            if f.name == func_identity_name and f.domain == func_domain), None))
        self.assertIsNotNone(
            next((f for f in extracted_with_o_all_funcion0.functions
            if f.name == func_nested_identity_add_name and f.domain == func_domain), None))

        extracted_with_o_all_funcion1 = utils.Extractor(model).extract_model(['i0', 'i1', 'i2'], ['o_all_func1'])
        checker.check_model(extracted_with_o_all_funcion1)
        self.assertEqual(len(extracted_with_o_all_funcion1.functions), 3)
        self.assertIsNotNone(
            next((f for f in extracted_with_o_all_funcion1.functions
            if f.name == func_add_name and f.domain == func_domain), None))
        self.assertIsNotNone(
            next((f for f in extracted_with_o_all_funcion1.functions
            if f.name == func_identity_name and f.domain == func_domain), None))
        self.assertIsNotNone(
            next((f for f in extracted_with_o_all_funcion1.functions
            if f.name == func_nested_identity_add_name and f.domain == func_domain), None))


if __name__ == '__main__':
    unittest.main()
