# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Illustrative example of using an Opaque type.

Opaque types allow a model to carry values whose internal representation is
not defined by the ONNX spec, but is instead understood only by the
producer/consumer of a custom domain's ops (much like custom ops themselves).

This test illustrates the idea with a toy "random number generator" (RNG)
type: one custom op creates a stateful RNG (represented via an Opaque type)
from a seed value, and another custom op consumes the RNG (together with a
requested shape) to produce a random tensor, returning both the generated
tensor and an updated RNG (since ONNX ops are functions and cannot mutate
their inputs in place: any state update must be reflected as an explicit
output). The example is deliberately simple -- it does not implement an
actual RNG algorithm nor address the many details (such as the precise
semantics of the RNG state update) that a real-world stateful-RNG design
would need to address -- the goal here is just to show how Opaque types can
be declared, produced, consumed, and type/shape-inferred.
"""

from __future__ import annotations

import onnx
import onnx.checker
import onnx.defs
import onnx.parser
import onnx.shape_inference
from onnx import TensorProto, TypeProto, checker
from onnx.defs import OpSchema

# Use a dedicated custom domain for the two illustrative ops below.
DOMAIN = "test.rng"

# String representation (as accepted by OpSchema type constraints/parameters)
# of the opaque RNG type: a type named "RNG" defined in DOMAIN.
RNG_TYPE_STR = f"opaque({DOMAIN},RNG)"


def _make_rng_type_proto() -> TypeProto:
    """Builds a TypeProto representing the opaque RNG type."""
    type_proto = TypeProto()
    type_proto.opaque_type.domain = DOMAIN
    type_proto.opaque_type.name = "RNG"
    return type_proto


def _create_rng_schema() -> OpSchema:
    """Schema for a custom op that creates an RNG (opaque type) from a seed."""

    def infer_shape(ctx: onnx.shape_inference.InferenceContext) -> None:
        ctx.set_output_type(0, _make_rng_type_proto())

    schema = OpSchema(
        "CreateRNG",
        DOMAIN,
        1,
        doc="Creates a stateful random-number generator (RNG) from a seed value.",
        inputs=[OpSchema.FormalParameter("seed", "tensor(int64)")],
        outputs=[OpSchema.FormalParameter("rng", RNG_TYPE_STR)],
    )
    schema.set_type_and_shape_inference_function(infer_shape)
    return schema


def _random_tensor_schema() -> OpSchema:
    """Schema for a custom op that uses an RNG to produce a random tensor.

    ONNX ops are (mathematical) functions and cannot have side effects. Since
    using an RNG to generate a random value conceptually advances/updates its
    internal state, that update is made explicit by returning a new/updated
    RNG as a second output (alongside the generated tensor), rather than
    mutating the input RNG in place.
    """

    def infer_shape(ctx: onnx.shape_inference.InferenceContext) -> None:
        shape_attr = ctx.get_attribute("shape")
        shape = list(shape_attr.ints) if shape_attr is not None else []
        out = ctx.get_output_type(0)
        out.tensor_type.elem_type = TensorProto.FLOAT
        for dim in shape:
            out.tensor_type.shape.dim.add().dim_value = dim
        ctx.set_output_type(0, out)
        # The second output is the updated RNG, of the same (opaque) type as
        # the input RNG.
        ctx.set_output_type(1, ctx.get_input_type(0))

    schema = OpSchema(
        "RandomTensor",
        DOMAIN,
        1,
        doc=(
            "Generates a tensor of the given shape with values drawn from a "
            "(standard normal) distribution, using the given RNG. Returns "
            "the generated tensor along with the updated RNG, reflecting "
            "the state-update side-effect of generating random values."
        ),
        inputs=[OpSchema.FormalParameter("rng", RNG_TYPE_STR)],
        outputs=[
            OpSchema.FormalParameter("Y", "tensor(float)"),
            OpSchema.FormalParameter("rng_out", RNG_TYPE_STR),
        ],
        attributes=[OpSchema.Attribute("shape", OpSchema.AttrType.INTS)],
    )
    schema.set_type_and_shape_inference_function(infer_shape)
    return schema


class TestOpaqueTypeRNGExample:
    """Illustrates Opaque types via a toy stateful-RNG example.

    Two custom ops (in a custom domain) are used together in a single model:
    * ``CreateRNG(seed) -> rng`` produces an opaque ``RNG`` value.
    * ``RandomTensor(rng) -> Y, rng_out`` consumes the opaque ``RNG`` value
      (along with a ``shape`` attribute) to produce a random tensor,
      returning the updated RNG (``rng_out``) alongside the generated
      tensor (``Y``).

    The model itself is described using ``onnx.parser.parse_model``, ONNX's
    compact text format, as a single string.
    """

    @staticmethod
    def _register_schemas() -> None:
        onnx.defs.register_schema(_create_rng_schema())
        onnx.defs.register_schema(_random_tensor_schema())

    @staticmethod
    def _deregister_schemas() -> None:
        onnx.defs.deregister_schema("CreateRNG", 1, DOMAIN)
        onnx.defs.deregister_schema("RandomTensor", 1, DOMAIN)

    def setup_method(self):
        self._register_schemas()

    def teardown_method(self):
        self._deregister_schemas()

    @staticmethod
    def _make_model() -> onnx.ModelProto:
        # Note: the intermediate value "rng" and the second graph output
        # "rng2" are of the opaque RNG type, which is not (yet) expressible
        # in ONNX's text format. So they are left untyped here; their types
        # are subsequently determined by shape inference (see below).
        return onnx.parser.parse_model(
            f"""
            <
                ir_version: 10,
                opset_import: ["": {onnx.defs.onnx_opset_version()}, "{DOMAIN}": 1]
            >
            agraph (int64 seed) => (float[2,3] Y, rng2)
            {{
                rng = {DOMAIN}.CreateRNG (seed)
                Y, rng2 = {DOMAIN}.RandomTensor <shape = [2, 3]> (rng)
            }}
            """
        )

    def test_opaque_rng_example(self) -> None:
        model = self._make_model()

        # Shape inference should successfully infer the opaque RNG type for
        # the intermediate value "rng" and the second output "rng2", as well
        # as the tensor type/shape for the generated-tensor output "Y".
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)

        # The inferred model should pass checker validation.
        checker.check_model(inferred, full_check=True)

        value_infos = {vi.name: vi for vi in inferred.graph.value_info}
        rng_type = value_infos["rng"].type
        assert rng_type.WhichOneof("value") == "opaque_type"
        assert rng_type.opaque_type.domain == DOMAIN
        assert rng_type.opaque_type.name == "RNG"

        rng_out_type = inferred.graph.output[1].type
        assert rng_out_type.WhichOneof("value") == "opaque_type"
        assert rng_out_type.opaque_type.domain == DOMAIN
        assert rng_out_type.opaque_type.name == "RNG"

        output_type = inferred.graph.output[0].type
        assert output_type.tensor_type.elem_type == TensorProto.FLOAT
        assert [d.dim_value for d in output_type.tensor_type.shape.dim] == [2, 3]
