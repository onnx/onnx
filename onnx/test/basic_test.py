from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnx import AttributeProto, NodeProto, GraphProto, ModelProto, TensorProto, IR_VERSION

import io
import onnx
import os
import tempfile
import unittest

from onnx import helper


class TestBasicFunctions(unittest.TestCase):

    def _simple_model(self):
        # Create a ModelProto.
        model = ModelProto()
        model.ir_version = IR_VERSION
        return model

    def _simple_tensor(self):
        # Create a TensorProto.
        tensor = helper.make_tensor(
            name='test-tensor',
            data_type=TensorProto.FLOAT,
            dims=(2, 3, 4),
            vals=[x + 0.5 for x in range(24)]
        )
        return tensor

    def test_serialize_and_deserialize(self):
        opts = {
            '_simple_model': ModelProto,
            '_simple_tensor': TensorProto,
        }
        for gen in opts:
            obj = getattr(self, gen)()
            cls = opts[gen]
            se_obj = onnx.serialize(obj)
            de_obj = onnx.deserialize(se_obj, cls)
            se_obj2 = onnx.serialize(de_obj)
            de_obj2 = onnx.deserialize(se_obj, cls())
            self.assertTrue(obj == de_obj)
            self.assertTrue(se_obj == se_obj2)
            self.assertTrue(de_obj == de_obj2)
            self.assertRaises(ValueError, onnx.serialize, object())
            self.assertRaises(ValueError, onnx.deserialize, se_obj, object())

    def test_save_and_load(self):
        opts = {
            '_simple_model': ModelProto,
            '_simple_tensor': TensorProto,
        }
        for gen in opts:
            proto = getattr(self, gen)()
            cls = opts[gen]
            proto_string = onnx.serialize(proto)

            # Test if input is string
            loaded_proto = onnx.load_from_string(proto_string, cls)
            self.assertTrue(proto == loaded_proto)

            # Test if input has a read function
            f = io.BytesIO()
            onnx.save(proto_string, f)
            f = io.BytesIO(f.getvalue())
            loaded_proto = onnx.load(f, cls)
            self.assertTrue(proto == loaded_proto)

            # Test if input is a file name
            try:
                f = tempfile.NamedTemporaryFile(delete=False)
                onnx.save(proto, f)
                f.close()

                loaded_proto = onnx.load(f.name, cls)
                self.assertTrue(proto == loaded_proto)
            finally:
                os.remove(f.name)

    def test_existence(self):
        try:
            AttributeProto
            NodeProto
            GraphProto
            ModelProto
        except Exception as e:
            self.fail(
                'Did not find proper onnx protobufs. Error is: {}'
                .format(e))

    def test_version_exists(self):
        model = ModelProto()
        # When we create it, graph should not have a version string.
        self.assertFalse(model.HasField('ir_version'))
        # We should touch the version so it is annotated with the current
        # ir version of the running ONNX
        model.ir_version = IR_VERSION
        model_string = model.SerializeToString()
        model.ParseFromString(model_string)
        self.assertTrue(model.HasField('ir_version'))
        # Check if the version is correct.
        self.assertEqual(model.ir_version, IR_VERSION)


if __name__ == '__main__':
    unittest.main()
