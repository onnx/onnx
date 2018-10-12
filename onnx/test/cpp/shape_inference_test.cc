#include <iostream>
#include "gtest/gtest.h"
#include "onnx/onnx_pb.h"
#include "onnx/defs/shape_inference.h"

namespace ONNX_NAMESPACE {
namespace Test {

static void CreateDims(TypeProto_Tensor& proto, int num_dims) {
  auto mutable_shape = proto.mutable_shape();
  mutable_shape->clear_dim();

  for (int i = 0; i < num_dims; ++i)
    mutable_shape->add_dim();
}

static void SetDimValues(TypeProto_Tensor& proto, const std::vector<int>& values) {
  auto* mutable_shape = proto.mutable_shape();
  EXPECT_TRUE(mutable_shape->dim_size() == values.size());

  int idx = 0;
  for (auto value : values) {
    auto mutable_dim = mutable_shape->mutable_dim(idx++);
    if (value != -1)
      mutable_dim->set_dim_value(value);
  }
}

static void SetDimParams(TypeProto_Tensor& proto, const std::vector<const std::string*>& values) {
  auto mutable_shape = proto.mutable_shape();
  EXPECT_TRUE(mutable_shape->dim_size() == values.size());

  int idx = 0;
  for (auto value : values) {
    auto mutable_dim = mutable_shape->mutable_dim(idx++);
    if (value)
      mutable_dim->set_dim_param(*value);
  }
}

static void Dump(TypeProto_Tensor& t) {
  auto& s_shape = t.shape();
  auto num_dims = s_shape.dim_size();
  std::cout << num_dims << " dims. ";
  for (int i = 0; i < num_dims; ++i) {
    auto x = s_shape.dim(0);
    auto y = x.has_dim_value();
    auto z = x.has_dim_param();

    std::cout << "Dim " << i
              << " Value:" << (y ? std::to_string(x.dim_value()) : "<unset>")
              << ", Param:" << (z ? x.dim_param() : "<unset>") << "\n";
  }
};

TEST(ShapeInferenceTest, mergeShapeInfo_HasShape) {
  // source has shape, target doesn't
  {
    TypeProto_Tensor source;
    TypeProto_Tensor target;

    CreateDims(source, 1);
    SetDimValues(source, {1});
    mergeInShapeInfo(source, target);

    Dump(target);
    auto& shape = target.shape();
    EXPECT_TRUE(shape.dim_size() == 1 && shape.dim(0).dim_value() == 1);
  }

  // source has no shape, target does
  {
    TypeProto_Tensor source;
    TypeProto_Tensor target;

    CreateDims(target, 1);
    SetDimValues(target, {1});
    mergeInShapeInfo(source, target);

    Dump(target);
    auto& shape = target.shape();
    EXPECT_TRUE(shape.dim_size() == 1 && shape.dim(0).dim_value() == 1);
  }
}
TEST(ShapeInferenceTest, mergeShapeInfo_PreferValueOverParam) {
  std::string param = "A";

  // source has value, target has param. prefer value
  {
    TypeProto_Tensor source;
    TypeProto_Tensor target;

    CreateDims(source, 1);
    SetDimValues(source, {1});

    CreateDims(target, 1);
    SetDimParams(target, {&param});

    mergeInShapeInfo(source, target);

    Dump(target);
    auto& shape = target.shape();
    EXPECT_TRUE(shape.dim_size() == 1 && shape.dim(0).dim_value() == 1);
  }

  // source has param, target has value.
  {
    TypeProto_Tensor source;
    TypeProto_Tensor target;

    CreateDims(source, 1);
    SetDimParams(source, {&param});

    CreateDims(target, 1);
    SetDimValues(target, {1});

    mergeInShapeInfo(source, target);

    Dump(target);
    auto& shape = target.shape();
    EXPECT_TRUE(shape.dim_size() == 1 && shape.dim(0).dim_value() == 1);
  }
}

TEST(ShapeInferenceTest, mergeShapeInfo_CombineShapes) {
  // merge from both sides, preferring real value over -1
  {
    TypeProto_Tensor source;
    TypeProto_Tensor target;

    CreateDims(source, 2);
    SetDimValues(source, {-1, 2});

    CreateDims(target, 2);
    SetDimValues(target, {1, -1});

    mergeInShapeInfo(source, target);

    Dump(target);
    auto& shape = target.shape();
    EXPECT_TRUE(shape.dim(0).dim_value() == 1 &&
                shape.dim(1).dim_value() == 2);
  }

  // prefer value over param,
  {
    TypeProto_Tensor source;
    TypeProto_Tensor target;

    CreateDims(source, 2);
    SetDimValues(source, {-1, 2});

    CreateDims(target, 2);
    SetDimValues(target, {1, 0});
    // replace second dim with a param. the value from the source should be preferred
    const std::string param = "A";
    target.mutable_shape()->mutable_dim(1)->set_dim_param(param);

    mergeInShapeInfo(source, target);

    Dump(target);
    auto& shape = target.shape();
    EXPECT_TRUE(shape.dim(0).dim_value() == 1 &&
                shape.dim(1).dim_value() == 2);
  }
}

TEST(ShapeInferenceTest, mergeShapeInfo_Mismatches) {
  // mismatched num dims
  {
    TypeProto_Tensor source;
    TypeProto_Tensor target;

    CreateDims(source, 2);
    SetDimValues(source, {-1, 2});

    CreateDims(target, 3);
    SetDimValues(target, {1, -1, 1});

    EXPECT_THROW(mergeInShapeInfo(source, target), ONNX_NAMESPACE::InferenceError);
  }

  // mismatched dim values
  {
    TypeProto_Tensor source;
    TypeProto_Tensor target;

    CreateDims(source, 2);
    SetDimValues(source, {2, 2});

    CreateDims(target, 2);
    SetDimValues(target, {2, 1});

    EXPECT_THROW(mergeInShapeInfo(source, target), ONNX_NAMESPACE::InferenceError);
  }

  // mismatched param value. prefer target
  {
    TypeProto_Tensor source;
    TypeProto_Tensor target;
    const std::string param_a = "A";
    const std::string param_b = "B";

    CreateDims(source, 1);
    SetDimParams(source, {&param_a});

    CreateDims(target, 1);
    SetDimParams(target, {&param_b});

    mergeInShapeInfo(source, target);

    auto& shape = target.shape();
    EXPECT_TRUE(shape.dim(0).dim_param() == "B");
  }
}
} // namespace Test
} // namespace ONNX_NAMESPACE