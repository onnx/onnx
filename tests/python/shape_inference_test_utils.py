# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import onnx.shape_inference
from onnx import (
    GraphProto,
    ModelProto,
    NodeProto,
    TensorProto,
    TypeProto,
    ValueInfoProto,
    checker,
    helper,
)
from onnx.helper import (
    make_empty_tensor_value_info,
    make_node,
    make_tensor_value_info,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class TestShapeInferenceHelper:
    def _make_graph(
        self,
        seed_values: Sequence[str | tuple[str, TensorProto.DataType, Any]],
        nodes: list[NodeProto],
        value_info: list[ValueInfoProto],
        initializer: Sequence[TensorProto] | None = None,
    ) -> GraphProto:
        if initializer is None:
            initializer = []
        names_in_initializer = {x.name for x in initializer}
        input_value_infos = []
        # If the starting values are not also initializers,
        # introduce the starting values as the output of reshape,
        # so that the sizes are guaranteed to be unknown
        for seed_value in seed_values:
            if isinstance(seed_value, tuple):
                seed_name, proto_type = seed_value[:2]
                seed_value_info = make_tensor_value_info(*seed_value)
            else:
                seed_name, proto_type = seed_value, TensorProto.UNDEFINED
                seed_value_info = make_empty_tensor_value_info(seed_value)
            if seed_name in names_in_initializer:
                input_value_infos.append(seed_value_info)
            else:
                value_info.append(seed_value_info)
                input_value_infos.append(
                    make_tensor_value_info("SEED_" + seed_name, proto_type, ())
                )
                input_value_infos.append(
                    make_tensor_value_info(
                        "UNKNOWN_SHAPE_" + seed_name, TensorProto.INT64, (None,)
                    )
                )
                nodes[:0] = [
                    make_node(
                        "Reshape",
                        ["SEED_" + seed_name, "UNKNOWN_SHAPE_" + seed_name],
                        [seed_name],
                    )
                ]
        return helper.make_graph(
            nodes,
            "test",
            input_value_infos,
            [],
            initializer=initializer,
            value_info=value_info,
        )

    def _inferred(
        self, graph_or_model: GraphProto | ModelProto, **kwargs: Any
    ) -> ModelProto:
        data_prop = kwargs.pop("data_prop", False)
        if isinstance(graph_or_model, GraphProto):
            kwargs["producer_name"] = "onnx-test"
            orig_model = helper.make_model(graph_or_model, **kwargs)
        else:
            orig_model = graph_or_model
        inferred_model = onnx.shape_inference.infer_shapes(
            orig_model, check_type=True, strict_mode=True, data_prop=data_prop
        )
        checker.check_model(inferred_model)
        return inferred_model

    def _assert_inferred(
        self,
        graph_or_model: GraphProto | ModelProto,
        inferred_value_infos: list[ValueInfoProto],
        **kwargs: Any,
    ) -> None:
        graph = (
            graph_or_model
            if isinstance(graph_or_model, GraphProto)
            else graph_or_model.graph
        )
        # "inferred_value_infos" specifies the expected delta produced by type/shape inference.
        # The types/shapes specified in inferred_value_infos should be inferred by the inference implementation,
        # while for names not in inferred_value_infos, the original type/shape in input model should be preserved.
        names_in_inferred_value_infos = {x.name for x in inferred_value_infos}
        # The types/shapes can be recorded in graph.output and/or graph.value_info.
        # For the input model, if a name is specified in both, verify the two records
        # agree (symmetric to the check applied to the inferred model below), to avoid
        # masking inconsistent test inputs.
        expected: dict[str, ValueInfoProto] = {}
        for x in [*graph.value_info, *graph.output]:
            if x.name in names_in_inferred_value_infos:
                continue
            if x.name in expected:
                self._compare_value_infos(expected[x.name].type, x.type)
            else:
                expected[x.name] = x
        expected.update({x.name: x for x in inferred_value_infos})
        inferred_model = self._inferred(graph_or_model, **kwargs)
        inferred_graph = inferred_model.graph
        # Inferred type info may be recorded either in value_info (intermediate
        # values, and outputs that were untyped in the input model) or directly on
        # the graph outputs (outputs that were already typed). Merge both by name.
        # An untyped graph output is recorded in BOTH value_info and output; when a
        # name appears in both, verify that the two records agree.
        inferred: dict[str, ValueInfoProto] = {}
        for x in [*inferred_graph.value_info, *inferred_graph.output]:
            if x.name in inferred:
                self._compare_value_infos(inferred[x.name].type, x.type)
            else:
                inferred[x.name] = x
        assert expected.keys() == inferred.keys(), (
            f"\nExpected value infos for: {sorted(expected)}"
            f"\nInferred value infos for: {sorted(inferred)}\n"
        )
        for name, expected_vi in expected.items():
            self._compare_value_infos(expected_vi.type, inferred[name].type)

    def _compare_value_infos(
        self, vi_type: TypeProto, inferred_vi_type: TypeProto
    ) -> None:
        if vi_type.HasField("tensor_type"):
            assert inferred_vi_type.HasField("tensor_type")
            assert vi_type.tensor_type.HasField("elem_type")
            assert inferred_vi_type.tensor_type.HasField("elem_type")
            assert (
                vi_type.tensor_type.elem_type == inferred_vi_type.tensor_type.elem_type
            )
            assert vi_type.tensor_type.HasField(
                "shape"
            ) == inferred_vi_type.tensor_type.HasField("shape")
            if vi_type.tensor_type.HasField("shape"):
                assert len(vi_type.tensor_type.shape.dim) == len(
                    inferred_vi_type.tensor_type.shape.dim
                )
                for dim_i, dim in enumerate(vi_type.tensor_type.shape.dim):
                    inferred_dim = inferred_vi_type.tensor_type.shape.dim[dim_i]
                    # if it is a symbolic shape, make sure the inferred symbol has generated (dim_param)
                    if dim.dim_param:
                        assert dim.dim_param == inferred_dim.dim_param, (
                            f"\n{vi_type}\n{inferred_vi_type}\n"
                        )
                    else:
                        assert dim.dim_value == inferred_dim.dim_value, (
                            f"\n{vi_type}\n{inferred_vi_type}\n"
                        )
        elif vi_type.HasField("sequence_type"):
            assert inferred_vi_type.HasField("sequence_type")
            vi = vi_type.sequence_type.elem_type
            inferred_vi = inferred_vi_type.sequence_type.elem_type
            self._compare_value_infos(vi, inferred_vi)
        elif vi_type.HasField("optional_type"):
            assert inferred_vi_type.HasField("optional_type")
            vi = vi_type.optional_type.elem_type
            inferred_vi = inferred_vi_type.optional_type.elem_type
            self._compare_value_infos(vi, inferred_vi)
        elif vi_type.HasField("map_type"):
            assert inferred_vi_type.HasField("map_type")
            assert vi_type.map_type.key_type == inferred_vi_type.map_type.key_type
            self._compare_value_infos(
                vi_type.map_type.value_type, inferred_vi_type.map_type.value_type
            )
        elif vi_type == onnx.TypeProto():
            assert inferred_vi_type == onnx.TypeProto()
        else:
            raise NotImplementedError(
                "Unrecognized value info type in _compare_value_infos: ", str(vi_type)
            )

    def skipIf(self, condition, reason):
        if condition:
            pytest.skip(reason)
