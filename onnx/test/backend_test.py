# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import itertools
import platform
from typing import TYPE_CHECKING, Any

import numpy

import onnx.backend.base
import onnx.backend.test
import onnx.shape_inference
import onnx.version_converter
from onnx import ModelProto, NodeProto, TensorProto
from onnx.backend.base import Device, DeviceType
from onnx.backend.test.case.node.unique import unique_output_types
from onnx.backend.test.runner import BackendIsNotSupposedToImplementIt

if TYPE_CHECKING:
    from collections.abc import Sequence

# The following just executes the fake backend through the backend test
# infrastructure. Since we don't have full reference implementation of all ops
# in ONNX repo, it's impossible to produce the proper results. However, we can
# run 'checker' (that's what base Backend class does) to verify that all tests
# fed are actually well-formed ONNX models.
#
# If everything is fine, all the tests would be marked as "skipped".
#
# We don't enable report in this test because the report collection logic itself
# fails when models are mal-formed.


class DummyBackend(onnx.backend.base.Backend):
    @classmethod
    def prepare(
        cls, model: ModelProto, device: str = "CPU", **kwargs: Any
    ) -> onnx.backend.base.BackendRep | None:
        super().prepare(model, device, **kwargs)

        onnx.checker.check_model(model)

        # by default test strict shape inference
        kwargs = {"check_type": True, "strict_mode": True, **kwargs}
        model = onnx.shape_inference.infer_shapes(model, **kwargs)

        value_infos = {
            vi.name: vi
            for vi in itertools.chain(model.graph.value_info, model.graph.output)
        }

        if do_enforce_test_coverage_safelist(model):
            for node in model.graph.node:
                for i, output in enumerate(node.output):
                    if node.op_type == "Dropout" and i != 0:
                        continue
                    assert output in value_infos
                    tt = value_infos[output].type.tensor_type
                    assert tt.elem_type != TensorProto.UNDEFINED
                    for dim in tt.shape.dim:
                        assert dim.WhichOneof("value") == "dim_value"

        raise BackendIsNotSupposedToImplementIt(
            "This is the dummy backend test that doesn't verify the results but does run the checker"
        )

    @classmethod
    def run_node(
        cls,
        node: NodeProto,
        inputs: Any,
        device: str = "CPU",
        outputs_info: Sequence[tuple[numpy.dtype, tuple[int, ...]]] | None = None,
        **kwargs: Any,  # noqa: ARG003
    ) -> tuple[Any, ...] | None:
        super().run_node(node, inputs, device=device, outputs_info=outputs_info)
        raise BackendIsNotSupposedToImplementIt(
            "This is the dummy backend test that doesn't verify the results but does run the checker"
        )

    @classmethod
    def supports_device(cls, device: str) -> bool:
        d = Device(device)
        return d.type == DeviceType.CPU


test_coverage_safelist = {
    "bvlc_alexnet",
    "densenet121",
    "inception_v1",
    "inception_v2",
    "resnet50",
    "shufflenet",
    "SingleRelu",
    "squeezenet_old",
    "vgg19",
    "zfnet",
}


def do_enforce_test_coverage_safelist(model: ModelProto) -> bool:
    if model.graph.name not in test_coverage_safelist:
        return False
    return all(node.op_type not in {"RNN", "LSTM", "GRU"} for node in model.graph.node)


def test_unique_output_types() -> None:
    x = numpy.empty((2, 3, 4), dtype=numpy.float32)
    for axis, y_shape in ((None, [None]), (-1, [2, 3, None])):
        expected = [
            onnx.helper.make_tensor_type_proto(TensorProto.FLOAT, y_shape),
            *[
                onnx.helper.make_tensor_type_proto(TensorProto.INT64, [None])
                for _ in range(3)
            ],
        ]
        assert unique_output_types(x, axis) == expected


test_kwargs = {
    # https://github.com/onnx/onnx/issues/5510 (test_mvn fails with backend_test.py)
    "test_mvn": {"strict_mode": False},
}

backend_test = onnx.backend.test.BackendTest(
    DummyBackend, __name__, test_kwargs=test_kwargs
)
if platform.architecture()[0] == "32bit":
    backend_test.exclude(r"(test_vgg19|test_zfnet|test_bvlc_alexnet)")

# import all test cases at global scope to make them visible to python.unittest
globals().update(backend_test.test_cases)
