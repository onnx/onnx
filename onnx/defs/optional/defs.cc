/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"

#include <algorithm>
#include <numeric>

namespace ONNX_NAMESPACE {
static std::vector<std::string> optional_and_tensor_types() {
  auto optional_types = OpSchema::all_optional_types();
  auto tensor_types = OpSchema::all_tensor_types();
  auto sequence_types = OpSchema::all_tensor_sequence_types();
  optional_types.insert(optional_types.end(), tensor_types.begin(), tensor_types.end());
  optional_types.insert(optional_types.end(), sequence_types.begin(), sequence_types.end());
  return optional_types;
}

// getOptionalType: returns the type of a value-attribute, if specified, of an Optional op.
// Returns true if a value-attribute is present and type was filled in.
bool getOptionalType (InferenceContext& ctx, TypeProto_Tensor& tensorTypeProto) {
  auto* value = ctx.getAttribute("value");
  auto* sparse_value = ctx.getAttribute("sparse_value");
  auto* value_int = ctx.getAttribute("value_int");
  auto* value_ints = ctx.getAttribute("value_ints");
  auto* value_float = ctx.getAttribute("value_float");
  auto* value_floats = ctx.getAttribute("value_floats");
  auto* value_string = ctx.getAttribute("value_string");
  auto* value_strings = ctx.getAttribute("value_strings");

  int num_value_attrs =
      (nullptr != value) +
      (nullptr != sparse_value) +
      (nullptr != value_int) +
      (nullptr != value_ints) +
      (nullptr != value_float) +
      (nullptr != value_floats) +
      (nullptr != value_string) +
      (nullptr != value_strings);

  if (num_value_attrs > 1) {
    fail_shape_inference(
        "Only one of the attributes 'value', 'value_*' or 'sparse_value' must be specified for an Optional node.");
  }

  tensorTypeProto.mutable_shape()->clear_dim();

  auto set_scalar_type = [&](int dtype) {
    tensorTypeProto.set_elem_type(dtype);
  };

  auto set_1D_type = [&](int dtype, int64_t size) {
    tensorTypeProto.set_elem_type(dtype);
    tensorTypeProto.mutable_shape()->add_dim()->set_dim_value(size);
  };

  auto set_ND_type = [&](int dtype, const google::protobuf::RepeatedField<int64_t>& dims) {
    tensorTypeProto.set_elem_type(dtype);
    for (auto d : dims) {
      tensorTypeProto.mutable_shape()->add_dim()->set_dim_value(d);
    }
  };

  if (nullptr != value) {
    const TensorProto& tensor_proto = value->t();
    set_ND_type(tensor_proto.data_type(), tensor_proto.dims());
    return true;
  }

  if (nullptr != value_int) {
    set_scalar_type(TensorProto::INT64);
    return true;
  }

  if (nullptr != value_ints) {
    set_1D_type(TensorProto::INT64, value_ints->ints_size());
    return true;
  }

  if (nullptr != value_float) {
    set_scalar_type(TensorProto::FLOAT);
    return true;
  }

  if (nullptr != value_floats) {
    set_1D_type(TensorProto::FLOAT, value_floats->floats_size());
    return true;
  }

  if (nullptr != value_string) {
    set_scalar_type(TensorProto::STRING);
    return true;
  }

  if (nullptr != value_strings) {
    set_1D_type(TensorProto::STRING, value_strings->strings_size());
    return true;
  }

  if (nullptr != sparse_value) {
    const SparseTensorProto& sparse = sparse_value->sparse_tensor();
    set_ND_type(sparse.values().data_type(), sparse.dims());
    return true;
  }

  return false;
}

void OptionalInferenceFunction(InferenceContext& ctx) {
  const size_t numInputs = ctx.getNumInputs();

  // Type specified via "type" attribute, if any.
  const auto* type_attr_proto = ctx.getAttribute("type");
  const TypeProto* attr_type = (type_attr_proto == nullptr) ? nullptr : & type_attr_proto->tp();

  // Type of value specified via some "value" attribute, if any.
  TypeProto val_type;
  bool const_value_specified = getOptionalType(ctx, *val_type.mutable_tensor_type());

  auto& target_type = *ctx.getOutputType(0)->mutable_optional_type()->mutable_elem_type();

  if ((numInputs > 0) && const_value_specified) {
    fail_type_inference("Optional must not specify both an input value and a value attribute.");
  }
  if (numInputs > 0) {
    // Construct an optional containing the input value
    auto input_type = ctx.getInputType(0);
    if (input_type == nullptr) {
      fail_type_inference("Input type is null. Type information is expected for the input.");
    }
    target_type.CopyFrom(*input_type);
    if (attr_type != nullptr)
      UnionTypeInfo (*attr_type, target_type);
  } else if (const_value_specified) {
    // Construct an optional containing the attribute-specified value
    target_type.CopyFrom(val_type);
    if (attr_type != nullptr)
      UnionTypeInfo (*attr_type, target_type);
  } else if (attr_type != nullptr) {
    auto& source_type = type_attr_proto->tp();
    target_type.CopyFrom(*attr_type);
  } else {
    fail_type_inference("Optional must specify type attribute if no value is specified.");
  }
}

static const char* Optional_ver19_doc = R"DOC(
Constructs an optional-type value containing either an empty optional of a certain type specified by the attribute,
or a non-empty value containing the input element or an attribute value, whichever is specified.

This operator is used to create either a `SOME v` value or a `NONE` value.
The value `v` may be specified either as an input argument or via attributes
(exactly as in the `Constant` op).

If no input value is specified (either via input or attribute) a `NONE` value is
constructed. In this case, the `type` attribute must be specified to enable a
(monomorphic) type to be inferred for the output.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Optional,
    19,
    OpSchema()
        .SetDoc(Optional_ver19_doc)
        .Input(0, "input", "The input element.", "V", OpSchema::Optional)
        .Attr("type", "Type of the element in the optional output", AttributeProto::TYPE_PROTO, OPTIONAL_VALUE)
        .Attr("value", "The value for the elements of the output tensor.", AttributeProto::TENSOR, false)
        .Attr(
            "sparse_value",
            "The value for the elements of the output tensor in sparse format.",
            AttributeProto::SPARSE_TENSOR,
            false)
        .Attr(
            "value_int",
            "The value for the sole element for the scalar, int64, output tensor.",
            AttributeProto::INT,
            false)
        .Attr(
            "value_ints",
            "The values for the elements for the 1D, int64, output tensor.",
            AttributeProto::INTS,
            false)
        .Attr(
            "value_float",
            "The value for the sole element for the scalar, float32, output tensor.",
            AttributeProto::FLOAT,
            false)
        .Attr(
            "value_floats",
            "The values for the elements for the 1D, float32, output tensor.",
            AttributeProto::FLOATS,
            false)
        .Attr(
            "value_string",
            "The value for the sole element for the scalar, UTF-8 string, output tensor.",
            AttributeProto::STRING,
            false)
        .Attr(
            "value_strings",
            "The values for the elements for the 1D, UTF-8 string, output tensor.",
            AttributeProto::STRINGS,
            false)
        .Output(0, "output", "The optional output enclosing the input element.", "O")
        .TypeConstraint(
            "V",
            []() {
              auto t = OpSchema::all_tensor_types();
              auto s = OpSchema::all_tensor_sequence_types();
              t.insert(t.end(), s.begin(), s.end());
              return t;
            }(),
            "Constrain input type to all tensor and sequence types.")
        .TypeConstraint(
            "O",
            OpSchema::all_optional_types(),
            "Constrain output type to all optional tensor or optional sequence types.")
        .TypeAndShapeInferenceFunction(OptionalInferenceFunction));

static const char* OptionalHasElement_ver18_doc = R"DOC(
Returns true if (1) the input is an optional-type and contains an element,
or, (2) the input is a tensor or sequence type.
If the input is not provided or is an empty optional-type, this op returns false.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    OptionalHasElement,
    18,
    OpSchema()
        .SetDoc(OptionalHasElement_ver18_doc)
        .Input(0, "input", "The optional input.", "O", OpSchema::Optional)
        .Output(
            0,
            "output",
            "A scalar boolean tensor. If true, it indicates that optional-type input contains an element. Otherwise, it is empty.",
            "B")
        .TypeConstraint(
            "O",
            optional_and_tensor_types(),
            "Constrain input type to optional tensor and optional sequence types.")
        .TypeConstraint("B", {"tensor(bool)"}, "Constrain output to a boolean tensor.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          const size_t numInputs = ctx.getNumInputs();
          if (numInputs != 0 && numInputs != 1) {
            fail_type_inference("OptionalHasElement is expected to have 0 or 1 input.");
          }
          const size_t numOutputs = ctx.getNumOutputs();
          if (numOutputs != 1) {
            fail_type_inference("OptionalHasElement is expected to have 1 output.");
          }
          auto* output_tensor_type = ctx.getOutputType(0)->mutable_tensor_type();
          output_tensor_type->set_elem_type(TensorProto::BOOL);
          output_tensor_type->mutable_shape()->Clear();
        }));

static const char* OptionalGetElement_ver18_doc = R"DOC(
If the input is a tensor or sequence type, it returns the input.
If the input is an optional type, it outputs the element in the input.
It is an error if the input is an empty optional-type (i.e. does not have an element) and the behavior is undefined in this case.
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    OptionalGetElement,
    18,
    OpSchema()
        .SetDoc(OptionalGetElement_ver18_doc)
        .Input(0, "input", "The optional input.", "O")
        .Output(0, "output", "Output element in the optional input.", "V")
        .TypeConstraint(
            "O",
            optional_and_tensor_types(),
            "Constrain input type to optional tensor and optional sequence types.")
        .TypeConstraint(
            "V",
            []() {
              auto t = OpSchema::all_tensor_types();
              auto s = OpSchema::all_tensor_sequence_types();
              t.insert(t.end(), s.begin(), s.end());
              return t;
            }(),
            "Constrain output type to all tensor or sequence types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          const size_t numInputs = ctx.getNumInputs();
          if (numInputs != 1) {
            fail_type_inference("OptionalGetElement must have an input element.");
          }
          auto input_type = ctx.getInputType(0);
          if (input_type == nullptr) {
            fail_type_inference("Input type is null. Input must have Type information.");
          }
          if (input_type->has_optional_type()) {
            if (!input_type->optional_type().has_elem_type()) {
              fail_type_inference("Optional-type input must contain an element with type information.");
            }
            ctx.getOutputType(0)->CopyFrom(input_type->optional_type().elem_type());
          } else {
            propagateShapeAndTypeFromFirstInput(ctx);
          }
        }));

} // namespace ONNX_NAMESPACE
