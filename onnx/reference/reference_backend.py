# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import numpy

import onnx.backend.base
import onnx.backend.test
import onnx.shape_inference
import onnx.version_converter
from onnx import ModelProto
from onnx.backend.base import Device, DeviceType
from onnx.reference import ReferenceEvaluator


class ReferenceEvaluatorBackendRep(onnx.backend.base.BackendRep):
    def __init__(self, session):
        self._session = session

    def run(self, inputs, **kwargs):
        if isinstance(inputs, numpy.ndarray):
            inputs = [inputs]
        if isinstance(inputs, list):
            if len(inputs) == len(self._session.input_names):
                feeds = dict(zip(self._session.input_names, inputs))
            else:
                feeds = {}
                pos_inputs = 0
                for inp, tshape in zip(
                    self._session.input_names, self._session.input_types
                ):
                    shape = tuple(d.dim_value for d in tshape.tensor_type.shape.dim)
                    if shape == inputs[pos_inputs].shape:
                        feeds[inp] = inputs[pos_inputs]
                        pos_inputs += 1
                        if pos_inputs >= len(inputs):
                            break
        elif isinstance(inputs, dict):
            feeds = inputs
        else:
            raise TypeError(f"Unexpected input type {type(inputs)!r}.")
        outs = self._session.run(None, feeds)
        return outs


class ReferenceEvaluatorBackend(onnx.backend.base.Backend):
    cls_inference = ReferenceEvaluator

    @classmethod
    def __class_getitem__(cls, cls_inference: type, name: str | None = None):
        """
        Creates a new class inheriting from this one but with
        static attribute `cls_inference` equal to *cls_inference*.
        The goal is to make it easier to evaluate a runtime
        sharing the same API as the :class:`ReferenceEvaluator`
        on CPU.
        """
        if name is None:
            name = f"{cls.__name__}{cls_inference.__name__}"
        return type(name, (cls,), {"cls_inference": cls_inference})

    @classmethod
    def is_opset_supported(cls, model):  # pylint: disable=unused-argument
        return True, ""

    @classmethod
    def supports_device(cls, device: str) -> bool:
        d = Device(device)
        return d.type == DeviceType.CPU  # type: ignore[no-any-return]

    @classmethod
    def create_inference_session(cls, model):
        return cls.cls_inference(model)

    @classmethod
    def prepare(
        cls, model: Any, device: str = "CPU", **kwargs: Any
    ) -> ReferenceEvaluatorBackendRep:
        if isinstance(model, cls.cls_inference):
            return ReferenceEvaluatorBackendRep(model)
        if isinstance(model, (str, bytes, ModelProto)):
            inf = cls.create_inference_session(model)
            return cls.prepare(inf, device, **kwargs)
        raise TypeError(f"Unexpected type {type(model)} for model.")

    @classmethod
    def run_model(cls, model, inputs, device=None, **kwargs):
        rep = cls.prepare(model, device, **kwargs)
        return rep.run(inputs, **kwargs)

    @classmethod
    def run_node(cls, node, inputs, device=None, outputs_info=None, **kwargs):
        raise NotImplementedError("Unable to run the model node by node.")


def create_reference_backend(
    backend: type | None = None,
    path_to_test: str | None = None,
    kind: str | None = None,
):
    return onnx.backend.test.BackendTest(
        backend or ReferenceEvaluatorBackend,
        __name__,
        path_to_test=path_to_test,
        kind=kind,
    )
