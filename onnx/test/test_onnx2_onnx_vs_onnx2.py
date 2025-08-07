from __future__ import annotations

import unittest

import numpy as np

import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
import onnx.onnx2.cpu._onnx2py as onnx2
import onnx.onnx2.helper as oh2


class TestOnnx2(unittest.TestCase):
    def assertEmpty(self, obj):  # noqa: N802
        self.assertFalse(obj)

    def assertNotEmpty(self, obj):  # noqa: N802
        self.assertTrue(obj)

    def test_onnx2_tensorproto(self):
        a = onh.from_array(
            np.array([[10, 20, 30, 40, 50, 60]]).reshape((2, 3, 1, 1)).astype(np.int16),
            name="AAAê€AAA",
        )
        a.doc_string = "help"
        s = a.SerializeToString()
        self.assertEqual(onnx2.utils_onnx2_read_varint64(b"\xac\x02"), (300, 2))
        i = onnx2.utils_onnx2_read_varint64(s[0:])
        self.assertEqual(i, (8, 1))
        t2 = onnx2.TensorProto()
        t2.ParseFromString(s)
        self.assertEqual(a.name, t2.name)
        self.assertEqual(a.doc_string, t2.doc_string)
        self.assertEqual(tuple(a.dims), tuple(t2.dims))
        self.assertEqual(a.data_type, int(t2.data_type))
        self.assertEqual(a.raw_data, t2.raw_data)
        a.raw_data = b"012345"
        self.assertEqual(a.raw_data, b"012345")

        # way back
        s2 = t2.SerializeToString()
        t = onnx.TensorProto()
        t.ParseFromString(s2)
        self.assertEqual(t.raw_data, t2.raw_data)
        self.assertEqual(t.name, t2.name)
        self.assertEqual(tuple(t.dims), tuple(t2.dims))

    def test_onnx2_tensorproto_metadata(self):
        a = onh.from_array(
            np.array([[10, 20, 30, 40, 50, 60]]).reshape((2, 3, 1, 1)).astype(np.int16),
            name="AAAê€AAA",
        )
        a.doc_string = "help"
        entry = a.metadata_props.add()
        entry.key = "k1"
        entry.value = "vv1"
        entry = a.metadata_props.add()
        entry.key = "k2"
        entry.value = "vv2"
        s = a.SerializeToString()
        self.assertEqual(onnx2.utils_onnx2_read_varint64(b"\xac\x02"), (300, 2))
        i = onnx2.utils_onnx2_read_varint64(s[0:])
        self.assertEqual(i, (8, 1))
        t2 = onnx2.TensorProto()
        t2.ParseFromString(s)
        self.assertEqual(a.name, t2.name)
        self.assertEqual(a.doc_string, t2.doc_string)
        self.assertEqual(tuple(a.dims), tuple(t2.dims))
        self.assertEqual(a.data_type, int(t2.data_type))
        self.assertEqual(a.raw_data, t2.raw_data)
        a.raw_data = b"012345"
        self.assertEqual(a.raw_data, b"012345")
        kv = list(t2.metadata_props)
        self.assertEqual(len(kv), 2)
        self.assertEqual(
            [kv[0].key, kv[0].value, kv[1].key, kv[1].value], ["k1", "vv1", "k2", "vv2"]
        )
        del t2.metadata_props[:]

    def test_string_string_entry_proto(self):
        p = onnx.StringStringEntryProto()
        p.key = "hk"
        p.value = "zoo"
        s = p.SerializeToString()
        p2 = onnx2.StringStringEntryProto()
        p2.ParseFromString(s)
        self.assertEqual((p2.key, p2.value), (p.key, p.value))
        # way back
        s2 = p2.SerializeToString()
        p = onnx.StringStringEntryProto()
        p.ParseFromString(s2)
        self.assertEqual((p2.key, p2.value), (p.key, p.value))

    def test_tensor_shape_proto(self):
        vts = oh.make_tensor_value_info(
            "iname",
            onnx.TensorProto.FLOAT,
            (4, "dyndyn"),
            "hellohello",
            ["DDDDD1", "DDDD2"],
        )
        ts = vts.type.tensor_type.shape
        bina = ts.SerializeToString()
        ts2 = onnx2.TensorShapeProto()
        ts2.ParseFromString(bina)
        self.assertEqual(len(ts.dim), len(ts2.dim))
        for d1, d2 in zip(ts.dim, ts2.dim):
            self.assertEqual(d1.dim_value, d2.dim_value or 0)
            self.assertEqual(d1.dim_param, d2.dim_param)
            self.assertEqual(d1.denotation, d2.denotation)
        # way back
        s2 = ts2.SerializeToString()
        onnx2.TensorShapeProto().ParseFromString(s2)
        ts = onnx.TensorShapeProto()
        ts.ParseFromString(s2)
        self.assertEqual(len(ts.dim), len(ts2.dim))
        for d1, d2 in zip(ts.dim, ts2.dim):
            self.assertEqual(d1.dim_value, d2.dim_value or 0)
            self.assertEqual(d1.dim_param, d2.dim_param)
            self.assertEqual(d1.denotation, d2.denotation)

    def test_operator_set_id(self):
        p = oh.make_opsetid("ai.onnx.ml", 5)
        s = p.SerializeToString()
        p2 = onnx2.OperatorSetIdProto()
        p2.ParseFromString(s)
        self.assertEqual((p2.domain, p2.version), (p.domain, p.version))
        # way back
        s2 = p2.SerializeToString()
        p = onnx.OperatorSetIdProto()
        p.ParseFromString(s2)
        self.assertEqual((p2.domain, p2.version), (p.domain, p.version))

    def test_operator_set_id_negative(self):
        p = oh.make_opsetid("ai.onnx.ml", -7)
        s = p.SerializeToString()
        p0 = onnx.OperatorSetIdProto()
        p0.ParseFromString(s)
        self.assertEqual((p0.domain, p0.version), (p.domain, p.version))
        p2 = onnx2.OperatorSetIdProto()
        p2.ParseFromString(s)
        self.assertEqual((p2.domain, p2.version), (p.domain, p.version))
        # way back
        s2 = p2.SerializeToString()
        p = onnx.OperatorSetIdProto()
        p.ParseFromString(s2)
        self.assertEqual((p2.domain, p2.version), (p.domain, p.version))

    def test_tensor_proto_double_data(self):
        p = onnx.TensorProto()
        p.name = "test"
        p.dims.extend([2])
        p.double_data.extend((4.0, 5.0))
        p.data_type = onnx.TensorProto.DOUBLE
        s = p.SerializeToString()

        p2 = onnx2.TensorProto()
        p2.ParseFromString(s)
        self.assertEqual(tuple(p.dims), tuple(p2.dims))
        self.assertEqual(tuple(p.double_data), tuple(p2.double_data))

        s2 = p2.SerializeToString()
        p0 = onnx.TensorProto()
        p0.ParseFromString(s2)
        s0 = p0.SerializeToString()
        self.assertEqual(tuple(p0.dims), tuple(p2.dims))
        self.assertEqual(tuple(p0.double_data), tuple(p2.double_data))
        self.assertEqual(s, s0)

    def test_tensor_proto_float_data(self):
        p = onnx.TensorProto()
        p.name = "test"
        p.dims.extend([2])
        p.float_data.extend((4.0, 5.0))
        p.data_type = onnx.TensorProto.FLOAT
        s = p.SerializeToString()

        p2 = onnx2.TensorProto()
        p2.ParseFromString(s)
        self.assertEqual(tuple(p.dims), tuple(p2.dims))
        self.assertEqual(tuple(p.float_data), tuple(p2.float_data))

        s2 = p2.SerializeToString()
        p0 = onnx.TensorProto()
        p0.ParseFromString(s2)
        s0 = p0.SerializeToString()
        self.assertEqual(tuple(p0.dims), tuple(p2.dims))
        self.assertEqual(tuple(p0.float_data), tuple(p2.float_data))
        self.assertEqual(s, s0)

    def test_tensor_proto_int32_data(self):
        p = onnx.TensorProto()
        p.name = "test"
        p.dims.extend([7])
        p.int32_data.extend((4, 5, 6, 7, 8, 9, 10))
        p.data_type = onnx.TensorProto.INT32
        s = p.SerializeToString()

        p2 = onnx2.TensorProto()
        p2.ParseFromString(s)
        self.assertEqual(tuple(p.dims), tuple(p2.dims))
        self.assertEqual(tuple(p.int32_data), tuple(p2.int32_data))

        s2 = p2.SerializeToString()
        p0 = onnx.TensorProto()
        p0.ParseFromString(s2)
        s0 = p0.SerializeToString()
        self.assertEqual(tuple(p0.dims), tuple(p2.dims))
        self.assertEqual(tuple(p0.int32_data), tuple(p2.int32_data))
        self.assertEqual(s, s0)

    def test_tensor_proto_int64_data(self):
        p = onnx.TensorProto()
        p.name = "test"
        p.dims.extend([2])
        p.int64_data.extend([4, 5])
        p.data_type = onnx.TensorProto.INT64
        s = p.SerializeToString()

        p2 = onnx2.TensorProto()
        p2.ParseFromString(s)
        self.assertEqual(tuple(p.dims), tuple(p2.dims))
        self.assertEqual(tuple(p.int64_data), tuple(p2.int64_data))

        s2 = p2.SerializeToString()
        p0 = onnx.TensorProto()
        p0.ParseFromString(s2)
        s0 = p0.SerializeToString()
        self.assertEqual(tuple(p0.dims), tuple(p2.dims))
        self.assertEqual(tuple(p.int64_data), tuple(p2.int64_data))
        self.assertEqual(tuple(p0.int64_data), tuple(p2.int64_data))
        self.assertEqual(s, s0)

    def test_tensor_proto_uint64_data(self):
        p = onnx.TensorProto()
        p.name = "test"
        p.dims.extend([2])
        p.uint64_data.extend((4, 5))
        p.data_type = onnx.TensorProto.UINT64
        s = p.SerializeToString()

        p2 = onnx2.TensorProto()
        p2.ParseFromString(s)
        self.assertEqual(tuple(p.dims), tuple(p2.dims))
        self.assertEqual(tuple(p.uint64_data), tuple(p2.uint64_data))

        s2 = p2.SerializeToString()
        p0 = onnx.TensorProto()
        p0.ParseFromString(s2)
        s0 = p0.SerializeToString()
        self.assertEqual(tuple(p0.dims), tuple(p2.dims))
        self.assertEqual(tuple(p.uint64_data), tuple(p2.uint64_data))
        self.assertEqual(tuple(p0.uint64_data), tuple(p2.uint64_data))
        self.assertEqual(s, s0)

    def test_tensor_proto_uint64_data_reverse1(self):
        p = onnx2.TensorProto()
        p.name = "test"
        s = p.name
        self.assertEqual(s, "test")
        p.dims.extend([2])
        p.uint64_data.extend((4, 5))
        p.data_type = onnx.TensorProto.UINT64
        s = p.SerializeToString()
        del p
        self.assertNotEmpty(s)
        p = onnx2.TensorProto()
        p.ParseFromString(s)
        self.assertEqual(tuple(p.uint64_data), (4, 5))

    def test_tensor_proto_uint64_data_reverse2(self):
        p = onnx2.TensorProto()
        p.name = "test"
        s = p.name
        self.assertEqual(s, "test")
        p.dims.extend([2])
        p.uint64_data.extend((4, 5))
        p.data_type = onnx.TensorProto.UINT64
        s = p.SerializeToString()
        del p
        self.assertNotEmpty(s)
        px = onnx.TensorProto()
        px.ParseFromString(s)
        self.assertEqual(tuple(px.uint64_data), (4, 5))

    def test_tensor_proto_uint64_data_reverse3(self):
        p = onnx.TensorProto()
        p.name = "test"
        s = p.name
        self.assertEqual(s, "test")
        p.dims.extend([2])
        p.uint64_data.extend([4, 5])
        p.data_type = onnx.TensorProto.UINT64
        s = p.SerializeToString()
        del p
        self.assertNotEmpty(s)
        px = onnx2.TensorProto()
        px.ParseFromString(s)
        self.assertEqual(list(px.uint64_data), [4, 5])

    def test_tensor_proto_uint64_data_reverse_whole(self):
        p = onnx2.TensorProto()
        p.name = "test"
        p.dims.extend([2])
        p.uint64_data.extend((4, 5))
        p.data_type = onnx.TensorProto.UINT64
        s = p.SerializeToString()

        p2 = onnx.TensorProto()
        p2.ParseFromString(s)
        self.assertEqual(tuple(p.dims), tuple(p2.dims))
        self.assertEqual(tuple(p.uint64_data), tuple(p2.uint64_data))

        s2 = p2.SerializeToString()
        p0 = onnx2.TensorProto()
        p0.ParseFromString(s2)
        s0 = p0.SerializeToString()
        self.assertEqual(tuple(p0.dims), tuple(p2.dims))
        self.assertEqual(tuple(p0.uint64_data), tuple(p2.uint64_data))
        self.assertEqual(s, s0)

    def test_tensor_proto_string_data(self):
        p = onnx.TensorProto()
        p.name = "test"
        p.dims.extend([2])
        p.string_data.extend((b"s4", b"s5"))
        p.data_type = onnx.TensorProto.STRING
        s = p.SerializeToString()

        p2 = onnx2.TensorProto()
        p2.ParseFromString(s)
        self.assertEqual(tuple(p.dims), tuple(p2.dims))
        self.assertEqual(tuple(p.string_data), tuple(p2.string_data))

        s2 = p2.SerializeToString()
        p0 = onnx.TensorProto()
        p0.ParseFromString(s2)
        s0 = p0.SerializeToString()
        self.assertEqual(tuple(p0.dims), tuple(p2.dims))
        self.assertEqual(tuple(p0.string_data), tuple(p.string_data))
        self.assertEqual(s, s0)

    def test_tensor_proto_string_data_reverse(self):
        p = onnx2.TensorProto()
        p.name = "test"
        p.dims.extend([2])
        p.string_data.extend((b"s4", b"s5"))
        p.data_type = onnx.TensorProto.STRING
        s = p.SerializeToString()

        p2 = onnx.TensorProto()
        p2.ParseFromString(s)
        self.assertEqual(tuple(p.dims), tuple(p2.dims))
        self.assertEqual(tuple(p.string_data), tuple(p2.string_data))

        s2 = p2.SerializeToString()
        p0 = onnx2.TensorProto()
        p0.ParseFromString(s2)
        s0 = p0.SerializeToString()
        self.assertEqual(tuple(p0.dims), tuple(p2.dims))
        self.assertEqual(tuple(p0.string_data), tuple(p.string_data))
        self.assertEqual(s, s0)

    def test_sparse_tensor_proto(self):
        dense_shape = [3, 3]
        sparse_values = [1.764052391052246, 0.40015721321105957, 0.978738009929657]
        values_tensor = oh.make_tensor(
            name="sparse_values",
            data_type=onnx.TensorProto.FLOAT,
            dims=[len(sparse_values)],
            vals=np.array(sparse_values).astype(np.float32),
            raw=False,
        )

        linear_indices = [2, 3, 5]
        indices_tensor = oh.make_tensor(
            name="indices",
            data_type=onnx.TensorProto.INT64,
            dims=[len(linear_indices)],
            vals=np.array(linear_indices).astype(np.int64),
            raw=False,
        )
        p = oh.make_sparse_tensor(values_tensor, indices_tensor, dense_shape)
        s = p.SerializeToString()
        self.assertEqual(p.__class__.__name__, "SparseTensorProto")

        p2 = onnx2.SparseTensorProto()
        p2.ParseFromString(s)
        self.assertEqual(tuple(p.dims), tuple(p2.dims))

        s2 = p2.SerializeToString()
        p0 = onnx.SparseTensorProto()
        p0.ParseFromString(s2)
        s0 = p0.SerializeToString()
        self.assertEqual(tuple(p0.dims), tuple(p2.dims))
        self.assertEqual(len(s), len(s0))
        self.assertEqual(s, s0)

    def test_tensor_annotation(self):
        p = onnx.TensorAnnotation()
        p.tensor_name = "T"
        e = p.quant_parameter_tensor_names.add()
        e.key = "K"
        e.value = "V"
        e = p.quant_parameter_tensor_names.add()
        e.key = "K2"
        e.value = "V2"

        s = p.SerializeToString()
        p2 = onnx2.TensorAnnotation()
        p2.ParseFromString(s)
        self.assertEqual(p.tensor_name, p2.tensor_name)

        s2 = p2.SerializeToString()
        p0 = onnx.TensorAnnotation()
        p0.ParseFromString(s2)
        self.assertEqual(p0.tensor_name, p2.tensor_name)
        self.assertEqual(p.SerializeToString(), p0.SerializeToString())

    def test_tensor_annotation_reverse(self):
        p = onnx2.TensorAnnotation()
        p.tensor_name = "T"
        e = p.quant_parameter_tensor_names.add()
        e.key = "K"
        e.value = "V"
        e = p.quant_parameter_tensor_names.add()
        e.key = "K2"
        e.value = "V2"

        s = p.SerializeToString()
        p2 = onnx.TensorAnnotation()
        p2.ParseFromString(s)
        self.assertEqual(p.tensor_name, p2.tensor_name)

        s2 = p2.SerializeToString()
        p0 = onnx2.TensorAnnotation()
        p0.ParseFromString(s2)
        self.assertEqual(p0.tensor_name, p2.tensor_name)
        self.assertEqual(p.SerializeToString(), p0.SerializeToString())

    def test_int_int_list_entry_proto(self):
        p = onnx.IntIntListEntryProto()
        p.key = 1
        p.value.extend([3, 4])

        s = p.SerializeToString()
        p2 = onnx2.IntIntListEntryProto()
        p2.ParseFromString(s)
        self.assertEqual(p.key, p2.key)

        s2 = p2.SerializeToString()
        p0 = onnx.IntIntListEntryProto()
        p0.ParseFromString(s2)
        self.assertEqual(p0.key, p2.key)
        self.assertEqual(p.SerializeToString(), p0.SerializeToString())

    def test_int_int_list_entry_proto_reverse(self):
        p = onnx2.IntIntListEntryProto()
        p.key = 1
        p.value.extend([3, 4])

        s = p.SerializeToString()
        p2 = onnx.IntIntListEntryProto()
        p2.ParseFromString(s)
        self.assertEqual(p.key, p2.key)

        s2 = p2.SerializeToString()
        p0 = onnx2.IntIntListEntryProto()
        p0.ParseFromString(s2)
        self.assertEqual(p0.key, p2.key)
        self.assertEqual(p.SerializeToString(), p0.SerializeToString())

    def test_device_configuration_proto(self):
        for x, x2 in [(onnx, onnx2), (onnx2, onnx)]:
            with self.subTest(start=x.__name__):
                p = x.DeviceConfigurationProto()
                p.name = "R"
                p.num_devices = 3
                p.device.extend(["T3", "G4"])

                s = p.SerializeToString()
                p2 = x2.DeviceConfigurationProto()
                p2.ParseFromString(s)

                s2 = p2.SerializeToString()
                p0 = x.DeviceConfigurationProto()
                p0.ParseFromString(s2)
                self.assertEqual(p.SerializeToString(), p0.SerializeToString())

    def test_simple_shared_dim_proto(self):
        for x, x2 in [(onnx, onnx2), (onnx2, onnx)]:
            with self.subTest(start=x.__name__):
                p = x.SimpleShardedDimProto()
                p.dim_value = 3
                # p.dim_param = "rt"
                p.num_shards = 4
                self.assertEqual(p.dim_value, 3)
                self.assertEqual(p.dim_param, "")

                s = p.SerializeToString()
                p2 = x2.SimpleShardedDimProto()
                p2.ParseFromString(s)

                s2 = p2.SerializeToString()
                p0 = x.SimpleShardedDimProto()
                p0.ParseFromString(s2)
                self.assertEqual(p.SerializeToString(), p0.SerializeToString())

        for x, x2 in [(onnx, onnx2), (onnx2, onnx)]:
            with self.subTest(start=x.__name__, case="dim_param"):
                p = x.SimpleShardedDimProto()
                # p.dim_value = 3
                p.dim_param = "rt"
                p.num_shards = 4
                self.assertIn(p.dim_value, (0, None))
                self.assertEqual(p.dim_param, "rt")

                s = p.SerializeToString()
                p2 = x2.SimpleShardedDimProto()
                p2.ParseFromString(s)

                s2 = p2.SerializeToString()
                p0 = x.SimpleShardedDimProto()
                p0.ParseFromString(s2)
                self.assertEqual(p.SerializeToString(), p0.SerializeToString())

    def test_sharded_dim_proto(self):
        for x, x2 in [(onnx, onnx2), (onnx2, onnx)]:
            with self.subTest(start=x.__name__, case="dim_value"):
                p = x.ShardedDimProto()
                p.axis = 3
                a = p.simple_sharding.add()
                a.dim_value = 4
                a.num_shards = 5

                s = p.SerializeToString()
                p2 = x2.ShardedDimProto()
                p2.ParseFromString(s)

                s2 = p2.SerializeToString()
                p0 = x.ShardedDimProto()
                p0.ParseFromString(s2)
                self.assertEqual(p.SerializeToString(), p0.SerializeToString())

    def test_sharding_spec_proto(self):
        for x, x2 in [(onnx, onnx2), (onnx2, onnx)]:
            with self.subTest(start=x.__name__):
                p = x.ShardingSpecProto()
                p.tensor_name = "erty"
                p.device.extend([4, 5])
                a = p.index_to_device_group_map.add()
                a.key = 10
                a.value.extend([6, 7])
                a = p.index_to_device_group_map.add()
                a.key = 11
                a.value.extend([61, 71])
                b = p.sharded_dim.add()
                b.axis = 3
                c = b.simple_sharding.add()
                c.dim_value = 4
                c.num_shards = 5

                s = p.SerializeToString()
                p2 = x2.ShardingSpecProto()
                p2.ParseFromString(s)

                s2 = p2.SerializeToString()
                p0 = x.ShardingSpecProto()
                p0.ParseFromString(s2)
                self.assertEqual(p.SerializeToString(), p0.SerializeToString())

    def test_node_device_configuration_proto(self):
        for x, x2 in [(onnx, onnx2), (onnx2, onnx)]:
            with self.subTest(start=x.__name__):
                p = x.NodeDeviceConfigurationProto()
                p.configuration_id = "cid"
                p.pipeline_stage = 5

                ps = p.sharding_spec.add()
                ps.tensor_name = "erty"
                ps.device.extend([4, 5])
                a = ps.index_to_device_group_map.add()
                a.key = 10
                a.value.extend([6, 7])
                a = ps.index_to_device_group_map.add()
                a.key = 11
                a.value.extend([61, 71])
                b = ps.sharded_dim.add()
                b.axis = 3
                c = b.simple_sharding.add()
                c.dim_value = 4
                c.num_shards = 5
                self.assertNotEmpty(p.sharding_spec)
                self.assertEqual(len(p.sharding_spec), 1)

                s = p.SerializeToString()
                p2 = x2.NodeDeviceConfigurationProto()
                p2.ParseFromString(s)

                s2 = p2.SerializeToString()
                p0 = x.NodeDeviceConfigurationProto()
                p0.ParseFromString(s2)
                self.assertEqual(p.SerializeToString(), p0.SerializeToString())

    def test_tensor_proto_data_type(self):
        self.assertEqual(onnx2.TensorProto.UNDEFINED, onnx.TensorProto.UNDEFINED)
        self.assertEqual(onnx2.TensorProto.FLOAT, onnx.TensorProto.FLOAT)
        self.assertEqual(onnx2.TensorProto.UINT8, onnx.TensorProto.UINT8)
        self.assertEqual(onnx2.TensorProto.INT8, onnx.TensorProto.INT8)
        self.assertEqual(onnx2.TensorProto.UINT16, onnx.TensorProto.UINT16)
        self.assertEqual(onnx2.TensorProto.INT16, onnx.TensorProto.INT16)
        self.assertEqual(onnx2.TensorProto.INT32, onnx.TensorProto.INT32)
        self.assertEqual(onnx2.TensorProto.INT64, onnx.TensorProto.INT64)
        self.assertEqual(onnx2.TensorProto.STRING, onnx.TensorProto.STRING)
        self.assertEqual(onnx2.TensorProto.BOOL, onnx.TensorProto.BOOL)
        self.assertEqual(onnx2.TensorProto.FLOAT16, onnx.TensorProto.FLOAT16)
        self.assertEqual(onnx2.TensorProto.DOUBLE, onnx.TensorProto.DOUBLE)
        self.assertEqual(onnx2.TensorProto.UINT32, onnx.TensorProto.UINT32)
        self.assertEqual(onnx2.TensorProto.UINT64, onnx.TensorProto.UINT64)
        self.assertEqual(onnx2.TensorProto.COMPLEX64, onnx.TensorProto.COMPLEX64)
        self.assertEqual(onnx2.TensorProto.COMPLEX128, onnx.TensorProto.COMPLEX128)
        self.assertEqual(onnx2.TensorProto.BFLOAT16, onnx.TensorProto.BFLOAT16)
        self.assertEqual(onnx2.TensorProto.FLOAT8E4M3FN, onnx.TensorProto.FLOAT8E4M3FN)
        self.assertEqual(
            onnx2.TensorProto.FLOAT8E4M3FNUZ, onnx.TensorProto.FLOAT8E4M3FNUZ
        )
        self.assertEqual(onnx2.TensorProto.FLOAT8E5M2, onnx.TensorProto.FLOAT8E5M2)
        self.assertEqual(
            onnx2.TensorProto.FLOAT8E5M2FNUZ, onnx.TensorProto.FLOAT8E5M2FNUZ
        )
        self.assertEqual(onnx2.TensorProto.UINT4, onnx.TensorProto.UINT4)
        self.assertEqual(onnx2.TensorProto.INT4, onnx.TensorProto.INT4)
        self.assertEqual(onnx2.TensorProto.FLOAT4E2M1, onnx.TensorProto.FLOAT4E2M1)
        # self.assertEqual(onnx2.TensorProto.FLOAT8E8M0, onnx.TensorProto.FLOAT8E8M0)

        for k in dir(onnx.TensorProto):
            if k[0] == "_":
                continue
            v = getattr(onnx.TensorProto, k)
            if isinstance(v, int):
                with self.subTest(attr=k):
                    self.assertEqual(v, getattr(onnx2.TensorProto, k))

    def test_type_proto_tensor_type(self):
        t = oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, ["a", "b", "c"])
        p = t.type
        p.denotation = "denot"

        s = p.SerializeToString()
        p2 = onnx2.TypeProto()
        p2.ParseFromString(s)

        s2 = p2.SerializeToString()
        p0 = onnx.TypeProto()
        p0.ParseFromString(s2)
        self.assertEqual(p.SerializeToString(), p0.SerializeToString())

    def test_type_proto_tensor_type_reverse(self):
        p = onnx2.TypeProto()
        p.add_tensor_type()
        p.tensor_type.elem_type = onnx2.TensorProto.FLOAT
        p.tensor_type.add_shape().dim.add().dim_param = "a"
        p.tensor_type.shape.dim.add().dim_param = "b"
        p.tensor_type.shape.dim.add().dim_param = "c"
        p.denotation = "denot"
        self.assertNotEmpty(p.tensor_type)
        self.assertNotEmpty(p.tensor_type.shape)
        self.assertNotEmpty(p.tensor_type.shape.dim)
        self.assertEqual(len(p.tensor_type.shape.dim), 3)

        s = p.SerializeToString()
        p2 = onnx.TypeProto()
        p2.ParseFromString(s)

        s2 = p2.SerializeToString()
        p0 = onnx2.TypeProto()
        p0.ParseFromString(s2)
        self.assertEqual(p.SerializeToString(), p0.SerializeToString())

    def test_type_proto_sparse_tensor_type_reverse(self):
        p = onnx2.TypeProto()
        p.add_sparse_tensor_type()
        p.sparse_tensor_type.elem_type = onnx2.TensorProto.FLOAT
        p.sparse_tensor_type.add_shape().dim.add().dim_param = "a"
        p.sparse_tensor_type.shape.dim.add().dim_param = "b"
        p.sparse_tensor_type.shape.dim.add().dim_param = "c"
        self.assertNotEmpty(p.sparse_tensor_type)
        self.assertNotEmpty(p.sparse_tensor_type.shape)
        self.assertNotEmpty(p.sparse_tensor_type.shape.dim)
        self.assertEqual(len(p.sparse_tensor_type.shape.dim), 3)
        p.denotation = "denot"

        s = p.SerializeToString()
        p2 = onnx.TypeProto()
        p2.ParseFromString(s)

        s2 = p2.SerializeToString()
        p0 = onnx2.TypeProto()
        p0.ParseFromString(s2)
        self.assertEqual(p.SerializeToString(), p0.SerializeToString())

    def test_tensor_shape_proto_reverse(self):
        p = onnx2.TensorShapeProto()
        d = p.dim.add()
        d.dim_param = "aaa"
        p.dim.add().dim_value = 1
        p.dim.add().dim_param = "ccc"
        self.assertEqual(len(p.dim), 3)
        self.assertEqual(p.dim[0].dim_param, "aaa")
        self.assertEqual(p.dim[1].dim_value, 1)
        self.assertEqual(p.dim[2].dim_param, "ccc")
        self.assertEqual([d.dim_value for d in p.dim], [None, 1, None])
        self.assertEqual([d.dim_param for d in p.dim], ["aaa", "", "ccc"])

        s = p.SerializeToString()
        p2 = onnx.TensorShapeProto()
        p2.ParseFromString(s)

        s2 = p2.SerializeToString()
        p0 = onnx2.TensorShapeProto()
        p0.ParseFromString(s2)
        self.assertEqual(p.SerializeToString(), p0.SerializeToString())

    def test_type_proto_sequence_type_reverse(self):
        p = onnx2.TypeProto()
        p.denotation = "denot"
        # not working yet
        """
        et = onnx2.TypeProto()
        p.sequence_type.elem_type = et
        self.assertNotEmpty(et)
        self.assertNotEmpty(p.sequence_type.elem_type)
        self.assertTrue(p.sequence_type.has_elem_type())
        et.tensor_type = onnx2.TypeProto.Tensor()
        et.tensor_type.elem_type = onnx2.TensorProto.FLOAT
        et.tensor_type.add_shape().dim.add().dim_param = "a"
        et.tensor_type.shape.dim.add().dim_param = "b"
        et.tensor_type.shape.dim.add().dim_param = "c"
        self.assertEqual(len(et.tensor_type.shape.dim), 3)
        et.denotation = "denott"
        """
        p.add_sequence_type().add_elem_type().add_tensor_type().elem_type = (
            onnx2.TensorProto.FLOAT
        )
        p.sequence_type.elem_type.tensor_type.add_shape().dim.add().dim_param = "a"
        p.sequence_type.elem_type.tensor_type.shape.dim.add().dim_param = "b"
        p.sequence_type.elem_type.tensor_type.shape.dim.add().dim_param = "c"
        p.sequence_type.elem_type.denotation = "denott"
        self.assertEqual(len(p.sequence_type.elem_type.tensor_type.shape.dim), 3)

        s = p.SerializeToString()
        p2 = onnx.TypeProto()
        p2.ParseFromString(s)
        self.assertEqual(len(p2.sequence_type.elem_type.tensor_type.shape.dim), 3)

        s2 = p2.SerializeToString()
        p0 = onnx2.TypeProto()
        p0.ParseFromString(s2)
        self.assertEqual(p.SerializeToString(), p0.SerializeToString())

    def test_type_proto_optional_type_reverse(self):
        p = onnx2.TypeProto()
        p.denotation = "denot"
        p.add_optional_type().add_elem_type().add_tensor_type().elem_type = (
            onnx2.TensorProto.FLOAT
        )
        p.optional_type.elem_type.tensor_type.add_shape().dim.add().dim_param = "a"
        p.optional_type.elem_type.tensor_type.shape.dim.add().dim_param = "b"
        p.optional_type.elem_type.tensor_type.shape.dim.add().dim_param = "c"
        p.optional_type.elem_type.denotation = "denott"
        self.assertEqual(len(p.optional_type.elem_type.tensor_type.shape.dim), 3)

        s = p.SerializeToString()
        p2 = onnx.TypeProto()
        p2.ParseFromString(s)
        self.assertEqual(len(p2.optional_type.elem_type.tensor_type.shape.dim), 3)

        s2 = p2.SerializeToString()
        p0 = onnx2.TypeProto()
        p0.ParseFromString(s2)
        self.assertEqual(p.SerializeToString(), p0.SerializeToString())

    def test_type_proto_sequence_type_assign1(self):
        p = onnx2.TypeProto()
        p.denotation = "denot"
        et = onnx2.TypeProto()
        et.tensor_type = onnx2.TypeProto.Tensor()
        et.tensor_type.elem_type = onnx2.TensorProto.FLOAT
        et.tensor_type.add_shape().dim.add().dim_param = "a"
        et.tensor_type.shape.dim.add().dim_param = "b"
        et.tensor_type.shape.dim.add().dim_param = "c"
        et.denotation = "denott"
        p.add_sequence_type().elem_type = et
        self.assertNotEmpty(et)
        self.assertNotEmpty(p.sequence_type.elem_type)
        self.assertTrue(p.sequence_type.has_elem_type())
        self.assertEqual(
            p.sequence_type.elem_type.tensor_type.shape.dim[0].dim_param, "a"
        )
        self.assertEqual(
            p.sequence_type.elem_type.tensor_type.shape.dim[1].dim_param, "b"
        )

    def test_type_proto_sequence_type_assign2(self):
        p = onnx2.TypeProto()
        p.denotation = "denot"
        et = onnx2.TypeProto()
        et.add_tensor_type()
        et.tensor_type.elem_type = onnx2.TensorProto.FLOAT
        et.tensor_type.add_shape().dim.add().dim_param = "a"
        et.tensor_type.shape.dim.add().dim_param = "b"
        et.tensor_type.shape.dim.add().dim_param = "c"
        et.denotation = "denott"
        p.add_sequence_type()
        p.sequence_type.elem_type = et
        self.assertNotEmpty(et)
        self.assertNotEmpty(p.sequence_type.elem_type)
        self.assertTrue(p.sequence_type.has_elem_type())
        self.assertEqual(
            p.sequence_type.elem_type.tensor_type.shape.dim[0].dim_param, "a"
        )
        self.assertEqual(
            p.sequence_type.elem_type.tensor_type.shape.dim[1].dim_param, "b"
        )

    def test_make_attribute_serialization(self):
        p = onnx2.AttributeProto()
        p.type = onnx2.AttributeProto.INT
        p.i = 1

        s = p.SerializeToString()
        p2 = onnx.AttributeProto()
        p2.ParseFromString(s)
        self.assertEqual(p2.i, 1)

        s2 = p2.SerializeToString()
        p0 = onnx2.AttributeProto()
        p0.ParseFromString(s2)
        self.assertEqual(p.SerializeToString(), p0.SerializeToString())

    def test_make_attribute_int(self):
        t = onnx2.AttributeProto()
        t.type = onnx2.AttributeProto.INT
        t.i = 1
        t2 = onnx2.AttributeProto()
        t2.CopyFrom(t)
        self.assertEqual(str(t), str(t2))
        node = onnx2.NodeProto()
        node.attribute.append(t)
        self.assertEqual(str(t), str(node.attribute[0]))

    def test_make_attribute_string(self):
        t = onnx2.AttributeProto()
        t.type = onnx2.AttributeProto.STRING
        t.s = "s1"
        t2 = onnx2.AttributeProto()
        t2.CopyFrom(t)
        self.assertEqual(str(t), str(t2))
        node = onnx2.NodeProto()
        node.attribute.append(t)
        self.assertEqual(str(t), str(node.attribute[0]))

    def test_make_attribute_tensor(self):
        values_tensor = oh2.make_tensor(
            name="v",
            data_type=onnx.TensorProto.FLOAT,
            dims=[3],
            vals=np.array([5, 6, 7]).astype(np.float32),
            raw=False,
        )
        self.assertIsInstance(values_tensor, onnx2.TensorProto)
        t = onnx2.AttributeProto()
        t.type = onnx2.AttributeProto.TENSOR
        t.t = values_tensor
        self.assertFalse(t.has_sparse_tensor())
        t2 = onnx2.AttributeProto()
        t2.CopyFrom(t)
        self.assertFalse(t2.has_sparse_tensor())
        self.assertEqual(str(t), str(t2))
        node = onnx2.NodeProto()
        node.attribute.append(t)
        self.assertFalse(t.has_sparse_tensor())
        self.assertFalse(node.attribute[0].has_sparse_tensor())
        self.assertEqual(str(t), str(node.attribute[0]))

    def test_make_attribute_tensor_serialization(self):
        values_tensor = oh2.make_tensor(
            name="v",
            data_type=onnx.TensorProto.FLOAT,
            dims=[3],
            vals=np.array([5, 6, 7]).astype(np.float32),
            raw=False,
        )
        self.assertIsInstance(values_tensor, onnx2.TensorProto)
        p = onnx2.AttributeProto()
        p.type = onnx2.AttributeProto.TENSOR
        p.t = values_tensor

        s = p.SerializeToString()
        p2 = onnx.AttributeProto()
        p2.ParseFromString(s)

        s2 = p2.SerializeToString()
        p0 = onnx2.AttributeProto()
        p0.ParseFromString(s2)
        self.assertEqual(p.SerializeToString(), p0.SerializeToString())
        self.assertIn("float_data: [5, 6, 7],", str(p))
        self.assertIn("float_data: [5, 6, 7],", str(p0))
        self.assertFalse(p.has_sparse_tensor())
        self.assertFalse(p0.has_sparse_tensor())

    def test_make_attribute_serialization_ints(self):
        p = onnx.AttributeProto()
        p.name = "axes"
        p.type = onnx.AttributeProto.INTS
        p.ints.extend([1])
        self.assertEqual(list(p.ints), [1])

        s = p.SerializeToString()
        p2 = onnx2.AttributeProto()
        p2.ParseFromString(s)
        self.assertEqual(list(p2.ints), [1])

        s2 = p2.SerializeToString()
        p0 = onnx.AttributeProto()
        p0.ParseFromString(s2)
        self.assertEqual(list(p0.ints), [1])
        self.assertEqual(p.SerializeToString(), p0.SerializeToString())

    def test_make_attribute_serialization_ints_reverse(self):
        p = onnx2.AttributeProto()
        p.name = "axes"
        p.type = onnx2.AttributeProto.INTS
        p.ints.extend([1])
        self.assertEqual(list(p.ints), [1])

        s = p.SerializeToString()
        p2 = onnx.AttributeProto()
        p2.ParseFromString(s)
        self.assertEqual(p2.ints, [1])

        s2 = p2.SerializeToString()
        p0 = onnx2.AttributeProto()
        p0.ParseFromString(s2)
        self.assertEqual(list(p0.ints), [1])
        self.assertEqual(p.SerializeToString(), p0.SerializeToString())

    def test_make_constant_int(self):
        for dt in [
            onnx.TensorProto.INT32,
            onnx.TensorProto.INT64,
            onnx.TensorProto.UINT64,
        ]:
            with self.subTest(dt=dt):
                t = onnx.TensorProto()
                t.data_type = dt
                t.dims.extend([1])
                t.int64_data.extend([-1])
                t.data_location = onnx.TensorProto.DataLocation.DEFAULT

                t2 = onnx2.TensorProto()
                t2.data_type = dt
                t2.dims.extend([1])
                t2.int64_data.extend([-1])
                t2.data_location = onnx2.TensorProto.DataLocation.DEFAULT

                node = oh.make_node(
                    "Constant", [], ["AAA"], value=t, domain="M", name="NN"
                )
                node.attribute[0].doc_string = "DOC"
                node.attribute[0].ref_attr_name = "REF"

                nnn2 = oh2.make_node(
                    "Constant", [], ["AAA"], value=t2, domain="M", name="NN"
                )
                nnn2.attribute[0].doc_string = "DOC"
                nnn2.attribute[0].ref_attr_name = "REF"

                s = node.SerializeToString()
                snn2 = nnn2.SerializeToString()
                onnx2.NodeProto().ParseFromString(snn2)

                node2 = onnx2.NodeProto()
                node2.ParseFromString(s)
                self.assertEqual(node2.name, node.name)
                self.assertEqual(list(node2.output), list(node.output))
                self.assertEqual(node2.domain, node.domain)
                self.assertEqual(len(node2.attribute), len(node.attribute))
                a1 = node.attribute[0]
                a2 = node2.attribute[0]
                self.assertEqual(a1.name, a2.name)
                self.assertEqual(a1.type, int(a2.type))
                self.assertEqual(list(a1.t.dims), list(a2.t.dims))
                self.assertEqual(list(a1.t.int64_data), list(a2.t.int64_data))
                self.assertEqual(len(s), len(snn2))

    def test_make_tensor_int(self):
        for dt in [
            onnx.TensorProto.INT32,
            onnx.TensorProto.INT64,
            onnx.TensorProto.UINT64,
        ]:
            with self.subTest(dt=dt):
                t = onnx.TensorProto()
                t.data_type = dt
                t.dims.extend([5])
                t.int64_data.extend([-1, 2, -3, 4, -5])
                t.data_location = onnx.TensorProto.DataLocation.DEFAULT

                t2 = onnx2.TensorProto()
                t2.data_type = dt
                t2.dims.extend([5])
                t2.int64_data.extend([-1, 2, -3, 4, -5])
                t2.data_location = onnx2.TensorProto.DataLocation.DEFAULT

                s = t.SerializeToString()
                snn2 = t2.SerializeToString()
                read2 = onnx2.TensorProto()
                read2.ParseFromString(s)
                self.assertEqual(read2.name, t.name)
                self.assertEqual(list(t.dims), list(read2.dims))
                self.assertEqual(list(t.int64_data), list(read2.int64_data))
                self.assertEqual(len(s), len(snn2))

    def test_make_constant_float(self):
        for dt in [onnx.TensorProto.FLOAT]:
            with self.subTest(dt=dt):
                t = onnx.TensorProto()
                t.data_type = dt
                t.dims.extend([5])
                t.float_data.extend([-1.0, 2.0, -3.0, 4.0, -5.0])
                t.data_location = onnx.TensorProto.DataLocation.DEFAULT

                t2 = onnx2.TensorProto()
                t2.data_type = dt
                t2.dims.extend([5])
                t2.float_data.extend([-1.0, 2.0, -3.0, 4.0, -5.0])
                t2.data_location = onnx2.TensorProto.DataLocation.DEFAULT
                self.assertEqual(
                    len(t.SerializeToString()), len(t2.SerializeToString())
                )
                self.assertEqual(
                    sorted(t.SerializeToString()),
                    sorted(t2.SerializeToString()),
                )

                node = oh.make_node(
                    "Constant",
                    [],
                    ["AAA"],
                    domain="M",
                    name="NN",
                    value=t,
                )
                node.attribute[0].doc_string = "DOC"
                node.attribute[0].ref_attr_name = "REF"

                nnn2 = oh2.make_node(
                    "Constant",
                    [],
                    ["AAA"],
                    domain="M",
                    name="NN",
                    value=t2,
                )
                nnn2.attribute[0].doc_string = "DOC"
                nnn2.attribute[0].ref_attr_name = "REF"

                s = node.SerializeToString()
                snn2 = nnn2.SerializeToString()
                self.assertEqual(len(s), len(snn2))

                node2 = onnx2.NodeProto()
                node2.ParseFromString(s)
                self.assertEqual(node2.name, node.name)
                self.assertEqual(list(node2.output), list(node.output))
                self.assertEqual(node2.domain, node.domain)
                self.assertEqual(len(node2.attribute), len(node.attribute))
                a1 = node.attribute[0]
                a2 = node2.attribute[0]
                self.assertEqual(a1.name, a2.name)
                self.assertEqual(a1.type, int(a2.type))
                self.assertEqual(list(a1.t.dims), list(a2.t.dims))
                self.assertEqual(list(a1.t.float_data), list(a2.t.float_data))
                self.assertEqual(len(s), len(snn2))

    def test_make_constant_double(self):
        for dt in [onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE]:
            with self.subTest(dt=dt):
                t = onnx.TensorProto()
                t.data_type = dt
                t.dims.extend([1])
                t.double_data.extend([-1.0])
                t.data_location = onnx.TensorProto.DataLocation.DEFAULT

                t2 = onnx2.TensorProto()
                t2.data_type = dt
                t2.dims.extend([1])
                t2.double_data.extend([-1.0])
                t2.data_location = onnx2.TensorProto.DataLocation.DEFAULT
                self.assertEqual(
                    sorted(t.SerializeToString()),
                    sorted(t2.SerializeToString()),
                )

                node = oh.make_node(
                    "Constant", [], ["AAA"], value=t, domain="M", name="NN"
                )
                node.attribute[0].doc_string = "DOC"
                node.attribute[0].ref_attr_name = "REF"

                nnn2 = oh2.make_node(
                    "Constant", [], ["AAA"], value=t2, domain="M", name="NN"
                )
                nnn2.attribute[0].doc_string = "DOC"
                nnn2.attribute[0].ref_attr_name = "REF"

                self.assertEqual(
                    len(node.attribute[0].SerializeToString()),
                    len(nnn2.attribute[0].SerializeToString()),
                )

                s = node.SerializeToString()
                snn2 = nnn2.SerializeToString()

                node2 = onnx2.NodeProto()
                node2.ParseFromString(s)
                self.assertEqual(node2.name, node.name)
                self.assertEqual(list(node2.output), list(node.output))
                self.assertEqual(node2.domain, node.domain)
                self.assertEqual(len(node2.attribute), len(node.attribute))
                a1 = node.attribute[0]
                a2 = node2.attribute[0]
                self.assertEqual(a1.name, a2.name)
                self.assertEqual(a1.type, int(a2.type))
                self.assertEqual(list(a1.t.dims), list(a2.t.dims))
                self.assertEqual(list(a1.t.double_data), list(a2.t.double_data))
                self.assertEqual(len(s), len(snn2))

    def test_make_node_attention(self):
        node = oh.make_node(
            "Attention",
            ["A", "B", "C"],
            ["D"],
            kv_num_heads=3,
            q_num_heads=3,
            scale=0.01,
            name="NAME",
        )

        for att in node.attribute:
            s = att.SerializeToString()
            a_ = onnx.AttributeProto()
            a_.ParseFromString(s)
            a2 = oh2.AttributeProto()
            a2.ParseFromString(s)
            s2 = a2.SerializeToString()
            a3 = onnx.AttributeProto()
            a3.ParseFromString(s2)

        s = node.SerializeToString()
        node_ = onnx.NodeProto()
        node_.ParseFromString(s)
        node2 = oh2.NodeProto()
        node2.ParseFromString(s)
        s2 = node2.SerializeToString()
        node3 = onnx.NodeProto()
        node3.ParseFromString(s2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
