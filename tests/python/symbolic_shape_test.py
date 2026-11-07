# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import onnx.parser
import onnx.shape_inference
from onnx import ModelProto, TensorProto, TensorShapeProto, ValueInfoProto, helper
from onnx.helper import make_model, make_tensor_value_info


class TestSymbolicShape:
    @pytest.mark.parametrize("version", [1, 10, 11, 12, 18, 22])
    def test_pooling_floor_does_not_alias_integer_division(self, version: int) -> None:
        model = make_model(
            helper.make_graph(
                [
                    helper.make_node(
                        "MaxPool", ["x"], ["pooled"], kernel_shape=[2], strides=[2]
                    ),
                    helper.make_node("Shape", ["x"], ["shape"]),
                    helper.make_node("Sub", ["shape", "two"], ["gap"]),
                    helper.make_node("Div", ["gap", "two"], ["quotient"]),
                    helper.make_node("Add", ["quotient", "one"], ["target"]),
                    helper.make_node("ConstantOfShape", ["target"], ["arithmetic"]),
                ],
                "pooling_floor",
                [make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, "H"])],
                [
                    make_tensor_value_info(name, TensorProto.FLOAT, None)
                    for name in ("pooled", "arithmetic")
                ],
                initializer=[
                    helper.make_tensor("two", TensorProto.INT64, [1], [2]),
                    helper.make_tensor("one", TensorProto.INT64, [1], [1]),
                ],
            ),
            opset_imports=[helper.make_opsetid("", version)],
        )
        # ConstantOfShape and partial arithmetic are available together from 18.
        if version < 18:
            del model.graph.node[1:]
            del model.graph.output[1:]
        inferred = onnx.shape_inference.infer_shapes(
            model, strict_mode=True, data_prop=True
        )
        pool_dim = inferred.graph.output[0].type.tensor_type.shape.dim[2]
        assert pool_dim.dim_param.startswith("floor(")
        if version >= 18:
            arithmetic_dim = inferred.graph.output[1].type.tensor_type.shape.dim[2]
            assert arithmetic_dim.dim_param
            assert pool_dim.dim_param != arithmetic_dim.dim_param

    def test_expression_names_reserve_type_attributes(self) -> None:
        model = make_model(
            helper.make_graph(
                [
                    helper.make_node("Flatten", ["x"], ["flat"], axis=0),
                    helper.make_node(
                        "Optional",
                        [],
                        ["optional"],
                        type=helper.make_tensor_type_proto(TensorProto.FLOAT, ["N*2"]),
                    ),
                ],
                "type_attributes",
                [make_tensor_value_info("x", TensorProto.FLOAT, ["N", 2])],
                [
                    make_tensor_value_info("flat", TensorProto.FLOAT, None),
                    helper.make_value_info("optional", onnx.TypeProto()),
                ],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
        )
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
        flat, optional = inferred.graph.output
        symbol = flat.type.tensor_type.shape.dim[1].dim_param
        assert symbol and symbol != "N*2"
        assert (
            optional.type.optional_type.elem_type.tensor_type.shape.dim[0].dim_param
            == "N*2"
        )

    @pytest.mark.parametrize("reverse", [False, True])
    def test_generated_and_atomic_operands_with_identical_spelling(
        self, reverse: bool
    ) -> None:
        nodes = [
            helper.make_node("Tile", ["flat", "repeats"], ["generated"]),
            helper.make_node("Tile", ["other", "repeats"], ["atomic"]),
        ]
        if reverse:
            nodes.reverse()
        nodes.insert(0, helper.make_node("Flatten", ["x"], ["flat"], axis=0))
        model = make_model(
            helper.make_graph(
                nodes,
                "operand_provenance",
                [
                    make_tensor_value_info("x", TensorProto.FLOAT, ["N", 2]),
                    make_tensor_value_info("other", TensorProto.FLOAT, [1, "N*2"]),
                ],
                [
                    make_tensor_value_info(name, TensorProto.FLOAT, None)
                    for name in ("generated", "atomic")
                ],
                initializer=[
                    helper.make_tensor("repeats", TensorProto.INT64, [2], [1, 3])
                ],
            )
        )
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
        generated, atomic = inferred.graph.output
        generated_name = generated.type.tensor_type.shape.dim[1].dim_param
        atomic_name = atomic.type.tensor_type.shape.dim[1].dim_param
        assert generated_name and atomic_name
        assert generated_name != atomic_name
        assert "(N*2)*3" in {generated_name, atomic_name}

    def test_expression_registry_is_shared_with_local_functions(self) -> None:
        function = helper.make_function(
            "local",
            "Flatten",
            ["x"],
            ["y"],
            [helper.make_node("Flatten", ["x"], ["y"], axis=0)],
            [helper.make_opsetid("", 18)],
            value_info=[make_tensor_value_info("reserved", TensorProto.FLOAT, ["N*2"])],
        )
        model = make_model(
            helper.make_graph(
                [
                    helper.make_node("Flatten", ["x"], ["flat"], axis=0),
                    helper.make_node("Flatten", ["x"], ["function"], domain="local"),
                    helper.make_node("Flatten", ["x"], ["again"], domain="local"),
                ],
                "local_functions",
                [make_tensor_value_info("x", TensorProto.FLOAT, ["N", 2])],
                [
                    make_tensor_value_info(name, TensorProto.FLOAT, None)
                    for name in ("flat", "function", "again")
                ],
            ),
            functions=[function],
            opset_imports=[
                helper.make_opsetid("", 18),
                helper.make_opsetid("local", 1),
            ],
        )
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
        flat, called, again = inferred.graph.output
        symbol = flat.type.tensor_type.shape.dim[1].dim_param
        assert symbol and symbol != "N*2"
        assert called.type.tensor_type.shape == flat.type.tensor_type.shape
        assert again.type.tensor_type.shape == flat.type.tensor_type.shape

    @pytest.mark.parametrize("entrypoint", ["node", "function"])
    def test_context_free_expression_inference_is_conservative(
        self, entrypoint: str
    ) -> None:
        node = helper.make_node("Flatten", ["x"], ["y"], axis=0)
        input_type = helper.make_tensor_type_proto(TensorProto.FLOAT, ["N", 2])
        if entrypoint == "node":
            output = onnx.shape_inference.infer_node_outputs(
                onnx.defs.get_schema("Flatten", 18), node, {"x": input_type}
            )["y"]
        else:
            function = helper.make_function(
                "local",
                "Flatten",
                ["x"],
                ["y"],
                [node],
                [helper.make_opsetid("", 18)],
            )
            output = onnx.shape_inference.infer_function_output_types(
                function, [input_type], []
            )[0]
        assert output.tensor_type.shape.dim[0].dim_value == 1
        assert not output.tensor_type.shape.dim[1].HasField("dim_value")
        assert not output.tensor_type.shape.dim[1].HasField("dim_param")

    @pytest.mark.parametrize("version", [1, 9, 11, 13, 21, 23, 24, 25, 27])
    @pytest.mark.parametrize("reverse", [False, True])
    def test_expression_names_reserve_existing_symbols(
        self, version: int, reverse: bool
    ) -> None:
        inputs = [
            make_tensor_value_info("x", TensorProto.FLOAT, ["N", 2]),
            make_tensor_value_info("other", TensorProto.FLOAT, ["N*2"]),
            make_tensor_value_info("suffix", TensorProto.FLOAT, ["N*2__0"]),
        ]
        nodes = [
            helper.make_node("Flatten", ["x"], ["flat"], axis=0),
            helper.make_node("Identity", ["other"], ["independent"]),
        ]
        if reverse:
            inputs.reverse()
            nodes.reverse()
        nodes.append(helper.make_node("Flatten", ["x"], ["repeated"], axis=0))
        model = make_model(
            helper.make_graph(
                nodes,
                "collision",
                inputs,
                [
                    make_tensor_value_info(name, TensorProto.FLOAT, None)
                    for name in ("flat", "independent", "repeated")
                ],
            ),
            opset_imports=[helper.make_opsetid("", version)],
        )
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
        flat, independent, repeated = inferred.graph.output
        symbol = flat.type.tensor_type.shape.dim[1].dim_param
        assert symbol
        assert symbol not in {"N", "N*2", "N*2__0"}
        assert independent.type.tensor_type.shape.dim[0].dim_param == "N*2"
        assert repeated.type.tensor_type.shape == flat.type.tensor_type.shape

    @pytest.mark.parametrize("reverse", [False, True])
    def test_expression_operands_distinguish_symbols_and_integers(
        self, reverse: bool
    ) -> None:
        nodes = [
            helper.make_node("Flatten", ["symbolic"], ["a"], axis=0),
            helper.make_node("Flatten", ["concrete"], ["b"], axis=0),
        ]
        if reverse:
            nodes.reverse()
        model = make_model(
            helper.make_graph(
                nodes,
                "typed_operands",
                [
                    make_tensor_value_info("symbolic", TensorProto.FLOAT, ["N", "2"]),
                    make_tensor_value_info("concrete", TensorProto.FLOAT, ["N", 2]),
                ],
                [
                    make_tensor_value_info(name, TensorProto.FLOAT, None)
                    for name in ("a", "b")
                ],
            )
        )
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
        a, b = inferred.graph.output
        assert a.type.tensor_type.shape.dim[1].dim_param
        assert b.type.tensor_type.shape.dim[1].dim_param
        assert (
            a.type.tensor_type.shape.dim[1].dim_param
            != b.type.tensor_type.shape.dim[1].dim_param
        )
        first = b if reverse else a
        assert first.type.tensor_type.shape.dim[1].dim_param == "N*2"

    def test_expression_provenance_is_not_recovered_from_annotations(self) -> None:
        model = onnx.parser.parse_model("""
            <ir_version: 8, opset_import: ["" : 18]>
            graph (float[N, 2] x) => (flat, repeated) {
                flat = Flatten<axis = 0>(x)
                repeated = Flatten<axis = 0>(x)
            }
        """)
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
        assert inferred.graph.output[0].type.tensor_type.shape.dim[1].dim_param == "N*2"
        restored = ModelProto.FromString(inferred.SerializeToString())
        restored.graph.input.append(
            make_tensor_value_info("independent", TensorProto.FLOAT, ["N*2", 3])
        )
        restored.graph.node.extend(
            [
                helper.make_node("Flatten", ["x"], ["fresh"], axis=0),
                helper.make_node("Flatten", ["x"], ["fresh_repeated"], axis=0),
                helper.make_node("Flatten", ["independent"], ["atomic"], axis=0),
            ]
        )
        restored.graph.output.extend(
            make_tensor_value_info(name, TensorProto.FLOAT, None)
            for name in ("fresh", "fresh_repeated", "atomic")
        )
        reinferred = onnx.shape_inference.infer_shapes(restored, strict_mode=True)
        old, repeated, fresh, fresh_repeated, atomic = reinferred.graph.output
        assert old == inferred.graph.output[0]
        assert repeated == inferred.graph.output[1]
        assert fresh.type.tensor_type.shape.dim[1].dim_param != "N*2"
        assert fresh.type.tensor_type.shape == fresh_repeated.type.tensor_type.shape
        assert atomic.type.tensor_type.shape.dim[1].dim_param == "(N*2)*3"
        assert reinferred.metadata_props == model.metadata_props

    def test_expression_symbols_are_shared_across_subgraphs(self) -> None:
        model = onnx.parser.parse_model("""
            <ir_version: 8, opset_import: ["" : 18]>
            graph (float[N, 2] x, float[M] other, bool cond)
                => (flat, branches) {
                flat = Flatten<axis = 0>(x)
                branches = If(cond) <
                    then_branch = then_graph () => (out) {
                        out = Flatten<axis = 0>(x)
                    },
                    else_branch = else_graph () => (out)
                        <float["N*2"] reserved> {
                        reserved = Identity(other)
                        out = Flatten<axis = 0>(x)
                    }
                >
            }
        """)
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
        flat, branches = inferred.graph.output
        symbol = flat.type.tensor_type.shape.dim[1].dim_param
        assert symbol and symbol != "N*2"
        assert branches.type.tensor_type.shape == flat.type.tensor_type.shape
        for attribute in inferred.graph.node[1].attribute:
            assert (
                attribute.g.output[0].type.tensor_type.shape
                == flat.type.tensor_type.shape
            )
        reinferred = onnx.shape_inference.infer_shapes(
            ModelProto.FromString(inferred.SerializeToString()), strict_mode=True
        )
        assert reinferred.graph.output == inferred.graph.output

    def _assert_valueinfo_shape(
        self, onnx_model: ModelProto, value_infos: list[ValueInfoProto]
    ) -> None:
        """Assert onnx_model.value_info should be the same as expected value_infos
        Instead of exact symbol, use -1 to represent symbolic shape in expected value_infos
        """
        for expected_vi in value_infos:
            shape = self._get_shape_from_name(onnx_model, expected_vi.name)
            assert shape is not None, f"{onnx_model}"
            if expected_vi.type.HasField("tensor_type"):
                expected_shape = expected_vi.type.tensor_type.shape
            elif expected_vi.type.HasField("sparse_tensor_type"):
                expected_shape = expected_vi.type.sparse_tensor_type.shape
            assert len(shape.dim) == len(expected_shape.dim), f"{onnx_model}"
            for dim_i, dim in enumerate(shape.dim):
                expected_dim = expected_shape.dim[dim_i]
                # -1 means it's a symbolic shape
                if expected_dim.dim_value == -1:
                    # symbolic dimension must exist
                    assert dim.dim_param, f"{onnx_model}"
                else:
                    assert dim.dim_value == expected_dim.dim_value, f"{onnx_model}"

    def _count_unique_dim_param_number(self, onnx_model: ModelProto) -> int:
        """Return the total number of unique symbolic shape"""
        symbol_shape_set = set()
        inputs = list(onnx_model.graph.input)
        outputs = list(onnx_model.graph.output)
        valueinfos = list(onnx_model.graph.value_info)
        for v in inputs + outputs + valueinfos:
            for dim in v.type.tensor_type.shape.dim:
                if dim.dim_param:
                    symbol_shape_set.add(dim.dim_param)
        return len(symbol_shape_set)

    def _get_shape_from_name(
        self, onnx_model: ModelProto, name: str
    ) -> TensorShapeProto | None:
        """Get shape from tensor_type or sparse_tensor_type according to given name"""
        inputs = list(onnx_model.graph.input)
        outputs = list(onnx_model.graph.output)
        valueinfos = list(onnx_model.graph.value_info)
        for v in inputs + outputs + valueinfos:
            if v.name == name:
                if v.type.HasField("tensor_type"):
                    return v.type.tensor_type.shape
                if v.type.HasField("sparse_tensor_type"):
                    return v.type.sparse_tensor_type.shape
        return None

    def test_concat_enable_symbolic(self) -> None:
        concat = helper.make_node(
            "Concat", inputs=["A", "B"], outputs=["C"], name="Concat", axis=1
        )
        cast = onnx.helper.make_node(
            "Cast", inputs=["C"], outputs=["output"], to=TensorProto.FLOAT
        )
        graph_def = helper.make_graph(
            name="test_graph",
            nodes=[concat, cast],
            inputs=[
                helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, "A"]),
                helper.make_tensor_value_info("B", TensorProto.FLOAT, [2, 3]),
            ],
            outputs=[
                helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, None])
            ],
        )

        onnx_model = make_model(graph_def)
        inferred_model = onnx.shape_inference.infer_shapes(onnx_model, strict_mode=True)
        self._assert_valueinfo_shape(
            inferred_model, [make_tensor_value_info("C", TensorProto.FLOAT, (2, -1))]
        )
        # the symbolic shape of C and output should be the same
        assert self._get_shape_from_name(
            inferred_model, "C"
        ) == self._get_shape_from_name(inferred_model, "output")

    def test_two_symbolic_concat(self) -> None:
        concat1 = helper.make_node(
            "Concat", inputs=["A", "B"], outputs=["C"], name="Concat", axis=1
        )
        concat2 = helper.make_node(
            "Concat", inputs=["C", "D"], outputs=["E"], name="Concat", axis=1
        )
        cast = onnx.helper.make_node(
            "Cast", inputs=["E"], outputs=["output"], to=TensorProto.FLOAT
        )
        graph_def = helper.make_graph(
            name="test_graph",
            nodes=[concat1, concat2, cast],
            inputs=[
                helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, "A"]),
                helper.make_tensor_value_info("B", TensorProto.FLOAT, [2, 3]),
                helper.make_tensor_value_info("D", TensorProto.FLOAT, [2, "D"]),
            ],
            outputs=[
                helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, None])
            ],
        )

        onnx_model = make_model(graph_def)
        inferred_model = onnx.shape_inference.infer_shapes(onnx_model, strict_mode=True)
        self._assert_valueinfo_shape(
            inferred_model,
            [
                make_tensor_value_info("C", TensorProto.FLOAT, (2, -1)),
                make_tensor_value_info("E", TensorProto.FLOAT, (2, -1)),
            ],
        )
        # the symbolic shape of E and output should be the same
        assert self._get_shape_from_name(
            inferred_model, "E"
        ) == self._get_shape_from_name(inferred_model, "output")

    def test_duplicate_symbolic_shape(self) -> None:
        concat1 = helper.make_node(
            "Concat", inputs=["A", "B"], outputs=["C"], name="Concat", axis=1
        )
        concat2 = helper.make_node(
            "Concat", inputs=["C", "D"], outputs=["E"], name="Concat", axis=1
        )
        cast = onnx.helper.make_node(
            "Cast", inputs=["E"], outputs=["output"], to=TensorProto.FLOAT
        )
        graph_def = helper.make_graph(
            name="test_graph",
            nodes=[concat1, concat2, cast],
            inputs=[
                helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, "_d0"]),
                helper.make_tensor_value_info("B", TensorProto.FLOAT, [2, 3]),
                helper.make_tensor_value_info("D", TensorProto.FLOAT, [2, "_d1"]),
            ],
            outputs=[
                helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, "_d0"])
            ],
        )

        onnx_model = make_model(graph_def)
        original_count = self._count_unique_dim_param_number(onnx_model)
        inferred_model = onnx.shape_inference.infer_shapes(onnx_model, strict_mode=True)
        inferred_count = self._count_unique_dim_param_number(inferred_model)
        # The two Concat expressions add distinct symbols without replacing
        # the pre-existing '_d0' and '_d1' annotations.
        assert inferred_count == original_count + 2, f"{inferred_model}{onnx_model}"

    def test_unknown_shape(self) -> None:
        concat = helper.make_node(
            "Concat", inputs=["A", "B"], outputs=["C"], name="Concat", axis=1
        )
        cast = onnx.helper.make_node(
            "Cast", inputs=["C"], outputs=["output"], to=TensorProto.FLOAT
        )
        graph_def = helper.make_graph(
            name="test_graph",
            nodes=[concat, cast],
            inputs=[
                helper.make_tensor_value_info(
                    "A", TensorProto.FLOAT, [3, None]
                ),  # unknown shape
                helper.make_tensor_value_info("B", TensorProto.FLOAT, [3, None]),
            ],
            outputs=[
                helper.make_tensor_value_info("output", TensorProto.FLOAT, [3, None])
            ],
        )

        onnx_model = make_model(graph_def)
        inferred_model = onnx.shape_inference.infer_shapes(onnx_model, strict_mode=True)
        self._assert_valueinfo_shape(
            inferred_model, [make_tensor_value_info("C", TensorProto.FLOAT, (3, -1))]
        )
        # C and output share the concrete first dimension and one symbolic dimension.
        assert self._get_shape_from_name(
            inferred_model, "C"
        ) == self._get_shape_from_name(inferred_model, "output")
