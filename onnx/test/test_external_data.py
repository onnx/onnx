import tempfile
import unittest
import uuid

import numpy as np
import shutil

import os

import onnx
from onnx import helper, load_from_disk, save_to_disk
from onnx.external_data_helper import get_all_tensors, \
    generate_persistence_value
from onnx.numpy_helper import to_array


class TestExternalData(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def get_temp_model_filename(self):
        return os.path.join(self.temp_dir, str(uuid.uuid4()) + '.onnx')

    def create_test_model(self):
        attribute_value = np.random.rand(2, 3).astype(np.float32)
        constant_node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['values'],
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=attribute_value.shape,
                vals=attribute_value.flatten().astype(float),
            ),
        )

        initializer_value = np.random.rand(3, 2).astype(np.float32)
        initializers = [onnx.helper.make_tensor(
            name='input_value',
            data_type=onnx.TensorProto.FLOAT,
            dims=initializer_value.shape,
            vals=initializer_value.flatten().astype(float),
        )]

        inputs = [helper.make_tensor_value_info('input_value', 
                                                onnx.TensorProto.FLOAT, 
                                                initializer_value.shape)]
        graph = helper.make_graph([constant_node], 'test_graph', 
                                  inputs=inputs, outputs=[], 
                                  initializer=initializers)
        model = helper.make_model(graph)
        return model

    def test_internal_to_external(self):
        model = self.create_test_model()
        initializer_value = to_array(model.graph.initializer[0])
        attribute_value = to_array(model.graph.node[0].attribute[0].t)

        # Save model and all tensor data in external files
        # To do this, first add a Persistence value to the external_data field
        # of each tensor we want to store in an external file
        for tensor in get_all_tensors(model):
            tensor.external_data = generate_persistence_value(tensor.name)

        # Now we can save the model
        filename = self.get_temp_model_filename()
        save_to_disk(model, filename)

        # Load model and external data from disk, verify
        loaded_model = load_from_disk(filename)

        initializer_tensor = loaded_model.graph.initializer[0]
        self.assertTrue(initializer_tensor.external_data.startswith('runtime'))
        self.assertTrue(np.allclose(to_array(initializer_tensor),
                                    initializer_value))

        attribute_tensor = loaded_model.graph.node[0].attribute[0].t
        self.assertTrue(attribute_tensor.external_data.startswith('runtime'))
        self.assertTrue(np.allclose(to_array(attribute_tensor),
                                    attribute_value))

    def test_external_to_internal(self):
        # Create a model with data stored in external files and save to disk
        model = self.create_test_model()
        initializer_value = to_array(model.graph.initializer[0])
        attribute_value = to_array(model.graph.node[0].attribute[0].t)

        for tensor in get_all_tensors(model):
            tensor.external_data = generate_persistence_value(tensor.name)
        filename = self.get_temp_model_filename()
        save_to_disk(model, filename)

        # Load model with external data from disk
        saved_model = load_from_disk(filename)

        # Load all tensors and clear the `external_data` field for all tensors
        for tensor in get_all_tensors(saved_model):
            to_array(tensor)
            tensor.ClearField('external_data')

        # Save model with internal data to disk
        filename = self.get_temp_model_filename()
        save_to_disk(saved_model, filename)

        # Load model with internal data, verify
        loaded_model = load_from_disk(filename)

        initializer_tensor = loaded_model.graph.initializer[0]
        self.assertFalse(initializer_tensor.HasField('external_data'))
        self.assertTrue(np.allclose(to_array(initializer_tensor),
                                    initializer_value))

        attribute_tensor = loaded_model.graph.node[0].attribute[0].t
        self.assertFalse(attribute_tensor.HasField('external_data'))
        self.assertTrue(np.allclose(to_array(attribute_tensor),
                                    attribute_value))

    def test_external_data_eager_loading(self):
        # Create a model with data stored in external files and save to disk
        model = self.create_test_model()
        initializer_value = to_array(model.graph.initializer[0])
        attribute_value = to_array(model.graph.node[0].attribute[0].t)

        for tensor in get_all_tensors(model):
            tensor.external_data = generate_persistence_value(tensor.name)
        filename = self.get_temp_model_filename()
        save_to_disk(model, filename)

        # Load model and eagerly load all external data from disk
        loaded_model = load_from_disk(filename, lazy_loading=False)
        initializer_tensor = loaded_model.graph.initializer[0]
        self.assertTrue(initializer_tensor.HasField('raw_data'))
        self.assertTrue(np.allclose(to_array(initializer_tensor),
                                    initializer_value))

        attribute_tensor = loaded_model.graph.node[0].attribute[0].t
        self.assertTrue(attribute_tensor.HasField('raw_data'))
        self.assertTrue(np.allclose(to_array(attribute_tensor),
                                    attribute_value))


if __name__ == '__main__':
    unittest.main()
