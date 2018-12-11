import tempfile
import unittest
import uuid

import numpy as np
import shutil

import os

import onnx
from onnx import helper
from onnx.helper import set_external_data
from onnx.numpy_helper import to_array, from_array


class TestLoadExternalData(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.initializer_value = np.arange(6).reshape(3, 2).astype(np.float32) + 512
        self.attribute_value = np.arange(6).reshape(2, 3).astype(np.float32) + 256
        self.model_filename = self.create_test_model()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def get_temp_model_filename(self):
        return os.path.join(self.temp_dir, str(uuid.uuid4()) + '.onnx')

    def create_external_data_tensor(self, value, tensor_name):
        tensor = from_array(np.array(value))
        tensor.name = tensor_name
        tensor_filename = "{}.bin".format(tensor_name)
        set_external_data(tensor, location=tensor_filename)

        with open(os.path.join(self.temp_dir, tensor_filename), 'wb') as data_file:
            data_file.write(tensor.raw_data)
        tensor.ClearField('raw_data')
        tensor.data_location = "external"
        return tensor

    def create_test_model(self):

        constant_node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['values'],
            value=self.create_external_data_tensor(self.attribute_value, "attribute_value")
        )

        initializers = [self.create_external_data_tensor(self.initializer_value, "input_value")]
        inputs = [helper.make_tensor_value_info("input_value",
                                                onnx.TensorProto.FLOAT,
                                                self.initializer_value.shape)]

        graph = helper.make_graph([constant_node], "test_graph",
                                  inputs=inputs, outputs=[],
                                  initializer=initializers)
        model = helper.make_model(graph)

        model_filename = os.path.join(self.temp_dir, "model.onnx")
        with open(model_filename, "wb") as model_file:
            model_file.write(model.SerializeToString())

        return model_filename

    def test_load_external_data(self):
        model = onnx.load_model(self.model_filename)
        initializer_tensor = model.graph.initializer[0]
        self.assertTrue(np.allclose(to_array(initializer_tensor), self.initializer_value))

        attribute_tensor = model.graph.node[0].attribute[0].t
        self.assertTrue(np.allclose(to_array(attribute_tensor), self.attribute_value))

    def test_save_external_data(self):
        model = onnx.load_model(self.model_filename)

        temp_dir = os.path.join(self.temp_dir, "save_copy")
        os.mkdir(temp_dir)
        new_model_filename = os.path.join(temp_dir, 'model.onnx')
        onnx.save_model(model, new_model_filename)

        new_model = onnx.load_model(new_model_filename)
        initializer_tensor = new_model.graph.initializer[0]
        self.assertTrue(np.allclose(to_array(initializer_tensor), self.initializer_value))

        attribute_tensor = new_model.graph.node[0].attribute[0].t
        self.assertTrue(np.allclose(to_array(attribute_tensor), self.attribute_value))


class TestLoadExternalDataSingleFile(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.initializer_value = np.arange(6).reshape(3, 2).astype(np.float32) + 512
        self.attribute_value = np.arange(6).reshape(2, 3).astype(np.float32) + 256
        self.model_filename = self.create_test_model()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def get_temp_model_filename(self):
        return os.path.join(self.temp_dir, str(uuid.uuid4()) + '.onnx')

    def pad_file_to_size(self, data_file, required_size):
        data_file.seek(0, 2)
        file_size = data_file.tell()
        if required_size > file_size:
            data_file.write(b"\0" * (required_size - file_size))

    def create_external_data_tensors(self, tensors_data):
        tensor_filename = "tensors.bin"
        tensors = []
        next_offset = 0

        with open(os.path.join(self.temp_dir, tensor_filename), 'ab') as data_file:
            for (value, tensor_name) in tensors_data:
                tensor = from_array(np.array(value))
                self.pad_file_to_size(data_file, next_offset)
                length_written = data_file.write(tensor.raw_data)

                tensor.name = tensor_name
                set_external_data(tensor, location=tensor_filename, offset=next_offset, length=length_written)
                tensor.ClearField("raw_data")
                tensor.data_location = "external"
                tensors.append(tensor)
                next_offset = ((data_file.tell() // 4096) + 1) * 4096

        return tensors

    def create_test_model(self):
        tensors = self.create_external_data_tensors([
            (self.attribute_value, "attribute_value"),
            (self.initializer_value, "input_value"),
        ])

        constant_node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['values'],
            value=tensors[0]
        )

        inputs = [helper.make_tensor_value_info("input_value",
                                                onnx.TensorProto.FLOAT,
                                                self.initializer_value.shape)]

        graph = helper.make_graph([constant_node], "test_graph",
                                  inputs=inputs, outputs=[],
                                  initializer=[tensors[1]])
        model = helper.make_model(graph)

        model_filename = os.path.join(self.temp_dir, 'model.onnx')
        with open(model_filename, "wb") as model_file:
            model_file.write(model.SerializeToString())
        return model_filename

    def test_load_external_single_file_data(self):
        model = onnx.load_model(self.model_filename)

        initializer_tensor = model.graph.initializer[0]
        self.assertTrue(np.allclose(to_array(initializer_tensor), self.initializer_value))

        attribute_tensor = model.graph.node[0].attribute[0].t
        self.assertTrue(np.allclose(to_array(attribute_tensor), self.attribute_value))

    def test_save_external_single_file_data(self):
        model = onnx.load_model(self.model_filename)

        temp_dir = os.path.join(self.temp_dir, "save_copy")
        os.mkdir(temp_dir)
        new_model_filename = os.path.join(temp_dir, 'model.onnx')
        onnx.save_model(model, new_model_filename)

        new_model = onnx.load_model(new_model_filename)
        initializer_tensor = new_model.graph.initializer[0]
        self.assertTrue(np.allclose(to_array(initializer_tensor), self.initializer_value))

        attribute_tensor = new_model.graph.node[0].attribute[0].t
        self.assertTrue(np.allclose(to_array(attribute_tensor), self.attribute_value))


if __name__ == '__main__':
    unittest.main()
