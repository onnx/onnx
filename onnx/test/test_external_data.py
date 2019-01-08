import tempfile
import unittest
import uuid

import numpy as np  # type: ignore
import shutil

import os
import os.path as Path

import onnx
from onnx import helper
from onnx import TensorProto
from onnx.helper import set_external_data
from onnx.numpy_helper import to_array, from_array
from typing import Any, AnyStr, Tuple, Text, List, IO, BinaryIO


class TestLoadExternalData(unittest.TestCase):

    def setUp(self):  # type: () -> None
        self.temp_dir = tempfile.mkdtemp()  # type: Text
        self.initializer_value = np.arange(6).reshape(3, 2).astype(np.float32) + 512
        self.attribute_value = np.arange(6).reshape(2, 3).astype(np.float32) + 256
        self.model_filename = self.create_test_model()

    def tearDown(self):  # type: () -> None
        shutil.rmtree(self.temp_dir)

    def get_temp_model_filename(self):  # type: () -> Text
        return os.path.join(self.temp_dir, str(uuid.uuid4()) + '.onnx')

    def create_external_data_tensor(self, value, tensor_name):  # type: (List[Any], Text) -> TensorProto
        tensor = from_array(np.array(value))
        tensor.name = tensor_name
        tensor_filename = "{}.bin".format(tensor_name)
        set_external_data(tensor, location=tensor_filename)

        with open(os.path.join(self.temp_dir, tensor_filename), 'wb') as data_file:
            data_file.write(tensor.raw_data)
        tensor.ClearField('raw_data')
        tensor.data_location = onnx.TensorProto.EXTERNAL
        return tensor

    def create_test_model(self):  # type: () -> Text

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

    def test_load_external_data(self):  # type: () -> None
        model = onnx.load_model(self.model_filename)
        initializer_tensor = model.graph.initializer[0]
        self.assertTrue(np.allclose(to_array(initializer_tensor), self.initializer_value))

        attribute_tensor = model.graph.node[0].attribute[0].t
        self.assertTrue(np.allclose(to_array(attribute_tensor), self.attribute_value))

    def test_save_external_data(self):  # type: () -> None
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

    def setUp(self):  # type: () -> None
        self.temp_dir = tempfile.mkdtemp()  # type: Text
        self.initializer_value = np.arange(6).reshape(3, 2).astype(np.float32) + 512
        self.attribute_value = np.arange(6).reshape(2, 3).astype(np.float32) + 256
        self.model_filename = self.create_test_model()

    def tearDown(self):  # type: () -> None
        shutil.rmtree(self.temp_dir)

    def get_temp_model_filename(self):  # type: () -> Text
        return os.path.join(self.temp_dir, str(uuid.uuid4()) + '.onnx')

    def create_external_data_tensors(self, tensors_data):  # type: (List[Tuple[List[Any],Any]]) -> List[TensorProto]
        tensor_filename = "tensors.bin"
        tensors = []

        with open(os.path.join(self.temp_dir, tensor_filename), 'ab') as data_file:
            for (value, tensor_name) in tensors_data:
                tensor = from_array(np.array(value))
                offset = data_file.tell()
                if offset % 4096 != 0:
                    data_file.write(b"\0" * (4096 - offset % 4096))
                    offset = offset + 4096 - offset % 4096

                data_file.write(tensor.raw_data)
                set_external_data(tensor, location=tensor_filename, offset=offset, length=data_file.tell() - offset)
                tensor.name = tensor_name
                tensor.ClearField("raw_data")
                tensor.data_location = onnx.TensorProto.EXTERNAL
                tensors.append(tensor)

        return tensors

    def create_test_model(self):  # type: () -> Text
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

    def test_load_external_single_file_data(self):  # type: () -> None
        model = onnx.load_model(self.model_filename)

        initializer_tensor = model.graph.initializer[0]
        self.assertTrue(np.allclose(to_array(initializer_tensor), self.initializer_value))

        attribute_tensor = model.graph.node[0].attribute[0].t
        self.assertTrue(np.allclose(to_array(attribute_tensor), self.attribute_value))

    def test_save_external_single_file_data(self):  # type: () -> None
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
