# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import automatic_conversion_test_base
import numpy as np
import pytest

import onnx
from onnx import helper

#####################################################################################
# Every test calls _test_op_conversion to downgrade a model from the most recent opset version
# to a early version and runs checker + shape inference on the downgraded model.
####################################################################################


class TestAutomaticDowngrade(automatic_conversion_test_base.TestAutomaticConversion):
    def _test_op_downgrade(self, op: str, *args, **kwargs):
        strict_check = kwargs.pop("strict_check", False)
        mode = "strict_downgrade" if strict_check else "downgrade"
        self._test_op_conversion(op, *args, **kwargs, mode=mode)

    @pytest.mark.parametrize(
        "op",
        [
            "ReduceL1",
            "ReduceL2",
            "ReduceLogSum",
            "ReduceLogSumExp",
            "ReduceMean",
            "ReduceMax",
            "ReduceMin",
            "ReduceProd",
            "ReduceSum",
            "ReduceSumSquare",
        ],
    )
    def test_reduce_ops(self, op) -> None:
        # TODO: need to add test cases for missing axes input which depends on this pr:
        # https://github.com/onnx/onnx/pull/5613
        axes = helper.make_tensor(
            "b", onnx.TensorProto.INT64, dims=[3], vals=np.array([0, 1, 2])
        )
        self._test_op_downgrade(
            op,
            from_opset=13,
            input_shapes=[[3, 4, 5], [3]],
            output_shapes=[[1, 1, 1]],
            input_types=[onnx.TensorProto.FLOAT, onnx.TensorProto.INT64],
            initializer=[axes],
        )

    def test_dft20_no_axis(self) -> None:
        self._test_model_conversion(
            to_opset=19,
            model="""
            <ir_version: 9, opset_import: [ "" : 20]>
            dft_no_axis (float[N, M, 1] x) => (float[N, M, 2] y)
            {
                y = DFT (x)
            }
        """,
        )

    def test_dft20_initializer_axis(self) -> None:
        self._test_model_conversion(
            to_opset=19,
            model="""
            <ir_version: 9, opset_import: [ "" : 20]>
            dft_no_axis (float[N, M, 1] x, int64 dft_length) => (float[N, K, 2] y)
            <int64 axis = {1}>
            {
                y = DFT (x, dft_length, axis)
            }
        """,
        )

    def test_dft20_constant_axis(self) -> None:
        self._test_model_conversion(
            to_opset=19,
            model="""
            <ir_version: 9, opset_import: [ "" : 20]>
            dft_no_axis (float[N, M, 1] x, int64 dft_length) => (float[N, K, 2] y)
            {
                axis = Constant <value = int64{1}>()
                y = DFT (x, dft_length, axis)
            }
        """,
        )

    def test_dft20_unknown_axis(self) -> None:
        self._test_model_conversion_fails(
            to_opset=19,
            model="""
            <ir_version: 9, opset_import: [ "" : 20]>
            dft_no_axis (float[N, M, 1] x, int64 dft_length, int64 axis) => (float[P, K, 2] y)
            {
                y = DFT (x, dft_length, axis)
            }
        """,
        )

    def test_Einsum(self) -> None:
        self._test_op_downgrade(
            "Einsum",
            12,
            [[3, 4, 5], [3, 5, 6]],
            [[3, 4, 6]],
            attrs={"equation": "bij, bjk -> bik"},
        )

    def test_attention_25_to_24_default_window(self) -> None:
        """Attention with disabled window bounds can be downgraded."""
        self._test_op_downgrade(
            "Attention",
            25,
            [[2, 3, 4, 8], [2, 3, 6, 8], [2, 3, 6, 8]],
            [[2, 3, 4, 8]],
            attrs={"left_window_size": -1, "right_window_size": -1},
        )

    @pytest.mark.parametrize(
        "window_attribute", ["left_window_size", "right_window_size"]
    )
    def test_attention_25_to_24_window_fails(self, window_attribute: str) -> None:
        """Attention with an enabled window bound cannot be downgraded."""
        model = onnx.parser.parse_model(
            f"""
            <ir_version: 10, opset_import: [ "" : 25]>
            attn (float[2, 3, 4, 8] Q, float[2, 3, 6, 8] K, float[2, 3, 6, 8] V)
                => (float[2, 3, 4, 8] Y)
            {{
                Y = Attention <{window_attribute} = 3> (Q, K, V)
            }}
            """
        )
        onnx.checker.check_model(model)
        with pytest.raises(
            RuntimeError,
            match=rf"{window_attribute} must be -1 .* got 3.*Windowed attention",
        ):
            onnx.version_converter.convert_version(model, 24)

    def test_LinearAttention_downgrade_fails(self) -> None:
        self._test_model_conversion_fails(
            to_opset=24,
            model="""
            <ir_version: 10, opset_import: [ "" : 27]>
            linear_attention (float[2, 4, 64] Q, float[2, 4, 64] K, float[2, 4, 64] V)
                => (float[2, 4, 64] output, float[2, 4, 16, 16] present_state)
            {
                output, present_state = LinearAttention <q_num_heads = 4, kv_num_heads = 4, update_rule = "linear"> (Q, K, V)
            }
        """,
        )

    def test_CausalConvWithState_downgrade_fails(self) -> None:
        # CausalConvWithState was introduced at opset 27; no decomposition
        # adapter exists for downgrading to opset 24. The version converter
        # must raise.
        self._test_model_conversion_fails(
            to_opset=24,
            model="""
            <ir_version: 10, opset_import: [ "" : 27]>
            causal_conv_with_state (float[2, 4, 8] input, float[4, 1, 4] weight)
                => (float[2, 4, 8] output, float[2, 4, 3] present_state)
            {
                output, present_state = CausalConvWithState (input, weight)
            }
        """,
        )

    def test_optional_downgrade(self) -> None:
        self._test_op_downgrade(
            "Optional",
            15,
            optional_outputs=(0,),
            strict_check=True,
        )

    def test_optional_has_element_downgrade_without_input(self) -> None:
        self._test_op_downgrade(
            "OptionalHasElement",
            18,
            input_shapes=(),
            output_shapes=((),),
            output_types=(onnx.TensorProto.BOOL,),
            strict_check=True,
        )

    def test_optional_get_element_downgrade(self) -> None:
        self._test_op_downgrade(
            "OptionalGetElement",
            18,
            strict_check=True,
        )

    def test_optional28_float6_attribute_downgrade_fails(self) -> None:
        element_type = helper.make_tensor_type_proto(
            onnx.TensorProto.FLOAT6E2M3, (3, 4, 5)
        )
        model = helper.make_model(
            helper.make_graph(
                [helper.make_node("Optional", [], ["output"], type=element_type)],
                "optional_float6",
                [],
                [
                    helper.make_value_info(
                        "output", helper.make_optional_type_proto(element_type)
                    )
                ],
            ),
            ir_version=14,
            opset_imports=[helper.make_opsetid("", 28)],
        )
        self._test_model_conversion_fails(to_opset=18, model=model)

    def test_optional_has_element18_downgrade_fails(self) -> None:
        # non-optional input is not allowed for OptionalHasElement-15
        self._test_model_conversion_fails(
            to_opset=15,
            model="""
                    <ir_version: 8, opset_import: [ "" : 18]>
                    optional_has_element (float[3, 4, 5] input)
                        => (bool output)
                    {
                        output = OptionalHasElement (input)
                    }
                """,
        )

    def test_optional_get_element18_downgrade_fails(self) -> None:
        # non-optional input is not allowed for OptionalGetElement-15
        self._test_model_conversion_fails(
            to_opset=15,
            model="""
                    <ir_version: 8, opset_import: [ "" : 18]>
                    optional_has_element (float[3, 4, 5] input)
                        => (float[3, 4, 5] output)
                    {
                        output = OptionalGetElement (input)
                    }
                """,
        )

    # bfloat16 is not supported for OptionalHasElement-18.
    # Adapter must reject all tensor and its container type:
    # tensor(bfloat16), seq(tensor(bfloat16)),
    # optional(tensor(bfloat16)), optional(seq(tensor(bfloat16)))
    def test_optional_has_element28_downgrade_fails_1(self) -> None:
        self._test_model_conversion_fails(
            to_opset=18,
            model="""
                <ir_version: 13, opset_import: [ "" : 28]>
                optional_has_element (bfloat16[3, 4, 5] input) => (bool output)
                {
                    output = OptionalHasElement (input)
                }
                """,
        )

    def test_optional_has_element28_downgrade_fails_2(self) -> None:
        self._test_model_conversion_fails(
            to_opset=18,
            model="""
                <ir_version: 13, opset_import: [ "" : 28]>
                optional_has_element (optional(bfloat16[3, 4, 5]) input)
                    => (bool output)
                {
                    output = OptionalHasElement (input)
                }
                """,
        )

    def test_optional_has_element28_downgrade_fails_3(self) -> None:
        self._test_model_conversion_fails(
            to_opset=18,
            model="""
                <ir_version: 13, opset_import: [ "" : 28]>
                optional_has_element (seq(bfloat16[3, 4, 5]) input)
                    => (bool output)
                {
                    output = OptionalHasElement (input)
                }
                """,
        )

    def test_optional_has_element28_downgrade_fails_4(self) -> None:
        self._test_model_conversion_fails(
            to_opset=18,
            model="""
                <ir_version: 13, opset_import: [ "" : 28]>
                optional_has_element (optional(seq(bfloat16[3, 4, 5])) input)
                    => (bool output)
                {
                    output = OptionalHasElement (input)
                }
                """,
        )

    # bfloat16 is not supported for OptionalGetElement-18.
    # Adapter must reject all tensor and its container type:
    # tensor(bfloat16), seq(tensor(bfloat16)),
    # optional(tensor(bfloat16)), optional(seq(tensor(bfloat16)))
    def test_optional_get_element28_downgrade_fails_1(self) -> None:
        self._test_model_conversion_fails(
            to_opset=18,
            model="""
                <ir_version: 13, opset_import: [ "" : 28]>
                optional_has_element (bfloat16[3, 4, 5] input)
                    => (bfloat16[3, 4, 5] output)
                {
                    output = OptionalGetElement (input)
                }
                """,
        )

    def test_optional_get_element28_downgrade_fails_2(self) -> None:
        self._test_model_conversion_fails(
            to_opset=18,
            model="""
                <ir_version: 13, opset_import: [ "" : 28]>
                optional_has_element (seq(bfloat16[3, 4, 5]) input)
                    => (seq(bfloat16[3, 4, 5]) output)
                {
                    output = OptionalGetElement (input)
                }
                """,
        )

    def test_optional_get_element28_downgrade_fails_3(self) -> None:
        self._test_model_conversion_fails(
            to_opset=18,
            model="""
                <ir_version: 13, opset_import: [ "" : 28]>
                optional_has_element (optional(bfloat16[3, 4, 5]) input)
                    => (bfloat16[3, 4, 5] output)
                {
                    output = OptionalGetElement (input)
                }
                """,
        )

    def test_optional_get_element28_downgrade_fails_4(self) -> None:
        self._test_model_conversion_fails(
            to_opset=18,
            model="""
                <ir_version: 13, opset_import: [ "" : 28]>
                optional_has_element (optional(seq(bfloat16[3, 4, 5])) input)
                    => (seq(bfloat16[3, 4, 5]) output)
                {
                    output = OptionalGetElement (input)
                }
                """,
        )

    def test_depth_to_space(self) -> None:
        self._test_op_downgrade(
            "DepthToSpace",
            28,
            [[1, 8, 3, 3]],
            [[1, 2, 6, 6]],
            attrs={"blocksize": 2, "mode": "CRD"},
        )

    def test_space_to_depth_dcr(self) -> None:
        self._test_op_downgrade(
            "SpaceToDepth",
            28,
            [[1, 2, 6, 6]],
            [[1, 8, 3, 3]],
            attrs={"blocksize": 2, "mode": "DCR"},
        )

    def test_space_to_depth_crd_downgrade_fails(self) -> None:
        model = onnx.parser.parse_model(
            """
            <ir_version: 10, opset_import: [ "" : 28]>
            space_to_depth_crd (float[1, 2, 6, 6] input) => (float[1, 8, 3, 3] output)
            {
                output = SpaceToDepth <blocksize = 2, mode = "CRD"> (input)
            }
            """
        )
        onnx.checker.check_model(model)
        with pytest.raises(RuntimeError, match="mode must have value DCR"):
            onnx.version_converter.convert_version(model, 27)
