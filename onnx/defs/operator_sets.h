/*
 * SPDX-License-Identifier: Apache-2.0
 */


#pragma once

#include "onnx/defs/schema.h"
#include "onnx/defs/operator_versions.h"

namespace ONNX_NAMESPACE {

// Forward declarations for ai.onnx version 1
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Abs);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Add);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, And);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ArgMax);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ArgMin);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, AveragePool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, BatchNormalization);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Cast);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Ceil);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Clip);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Concat);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Constant);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Conv);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ConvTranspose);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, DepthToSpace);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Div);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Dropout);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Elu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Equal);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Exp);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Flatten);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Floor);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, GRU);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Gather);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Gemm);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, GlobalAveragePool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, GlobalLpPool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, GlobalMaxPool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Greater);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, HardSigmoid);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Hardmax);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Identity);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, If);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, InstanceNormalization);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, LRN);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, LSTM);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, LeakyRelu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Less);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Log);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, LogSoftmax);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Loop);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, LpNormalization);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, LpPool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, MatMul);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Max);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, MaxPool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, MaxRoiPool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Mean);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Min);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Mul);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Neg);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Not);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Or);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, PRelu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Pad);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Pow);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, RNN);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, RandomNormal);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, RandomNormalLike);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, RandomUniform);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, RandomUniformLike);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Reciprocal);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ReduceL1);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ReduceL2);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ReduceLogSum);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ReduceLogSumExp);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ReduceMax);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ReduceMean);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ReduceMin);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ReduceProd);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ReduceSum);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ReduceSumSquare);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Relu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Reshape);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Selu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Shape);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Sigmoid);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Size);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Slice);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Softmax);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Softplus);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Softsign);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, SpaceToDepth);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Split);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Sqrt);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Squeeze);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Sub);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Sum);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Tanh);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Tile);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, TopK);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Transpose);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Unsqueeze);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Upsample);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Xor);

// Iterate over schema from ai.onnx version 1
class OpSet_Onnx_ver1 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Abs)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Add)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, And)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ArgMax)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ArgMin)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 1, AveragePool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 1, BatchNormalization)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Cast)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Ceil)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Clip)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Concat)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Constant)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Conv)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 1, ConvTranspose)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 1, DepthToSpace)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Div)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Dropout)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Elu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Equal)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Exp)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Flatten)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Floor)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, GRU)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Gather)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Gemm)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 1, GlobalAveragePool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 1, GlobalLpPool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 1, GlobalMaxPool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Greater)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 1, HardSigmoid)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Hardmax)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Identity)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, If)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 1, InstanceNormalization)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, LRN)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, LSTM)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, LeakyRelu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Less)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Log)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, LogSoftmax)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Loop)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 1, LpNormalization)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, LpPool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, MatMul)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Max)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, MaxPool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, MaxRoiPool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Mean)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Min)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Mul)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Neg)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Not)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Or)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, PRelu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Pad)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Pow)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, RNN)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 1, RandomNormal)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 1, RandomNormalLike)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 1, RandomUniform)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 1, RandomUniformLike)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Reciprocal)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ReduceL1)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ReduceL2)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 1, ReduceLogSum)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 1, ReduceLogSumExp)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ReduceMax)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ReduceMean)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ReduceMin)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ReduceProd)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, ReduceSum)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 1, ReduceSumSquare)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Relu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Reshape)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Selu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Shape)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Sigmoid)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Size)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Slice)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Softmax)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Softplus)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Softsign)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 1, SpaceToDepth)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Split)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Sqrt)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Squeeze)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Sub)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Sum)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Tanh)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Tile)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, TopK)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Transpose)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Unsqueeze)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Upsample)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 1, Xor)>());
  }
};

// Forward declarations for ai.onnx version 2
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 2, GlobalLpPool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 2, LpPool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 2, Pad);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 2, Split);

// Iterate over schema from ai.onnx version 2
class OpSet_Onnx_ver2 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 2, GlobalLpPool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 2, LpPool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 2, Pad)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 2, Split)>());
  }
};

// Forward declarations for ai.onnx version 3
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 3, GRU);

// Iterate over schema from ai.onnx version 3
class OpSet_Onnx_ver3 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 3, GRU)>());
  }
};

// Forward declarations for ai.onnx version 4
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 4, Concat);

// Iterate over schema from ai.onnx version 4
class OpSet_Onnx_ver4 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 4, Concat)>());
  }
};

// Forward declarations for ai.onnx version 5
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 5, Reshape);

// Iterate over schema from ai.onnx version 5
class OpSet_Onnx_ver5 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 5, Reshape)>());
  }
};

// Forward declarations for ai.onnx version 6
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Abs);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Add);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, BatchNormalization);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Cast);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Ceil);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Clip);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Div);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Dropout);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Elu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Exp);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Floor);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Gemm);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, HardSigmoid);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, InstanceNormalization);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, LeakyRelu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Log);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Max);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Mean);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Min);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Mul);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Neg);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, PRelu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Reciprocal);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Relu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Selu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Sigmoid);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Sqrt);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Sub);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Sum);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Tanh);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Tile);

// Iterate over schema from ai.onnx version 6
class OpSet_Onnx_ver6 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Abs)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Add)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 6, BatchNormalization)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Cast)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Ceil)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Clip)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Div)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Dropout)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Elu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Exp)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Floor)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Gemm)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 6, HardSigmoid)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 6, InstanceNormalization)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, LeakyRelu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Log)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Max)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Mean)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Min)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Mul)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Neg)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, PRelu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Reciprocal)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Relu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Selu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Sigmoid)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Sqrt)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Sub)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Sum)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Tanh)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 6, Tile)>());
  }
};

// Forward declarations for ai.onnx version 7
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Acos);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Add);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, And);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Asin);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Atan);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, AveragePool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, BatchNormalization);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Cos);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Div);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Dropout);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Equal);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Gemm);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Greater);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, GRU);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Less);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, LSTM);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Mul);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Or);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Pow);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, RNN);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Sin);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Sub);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Tan);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Upsample);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Multinomial);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Xor);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, PRelu);

// Iterate over schema from ai.onnx version 7
class OpSet_Onnx_ver7 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Acos)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Add)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, And)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Asin)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Atan)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 7, AveragePool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 7, BatchNormalization)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Cos)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Div)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Dropout)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Equal)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Gemm)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Greater)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, GRU)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Less)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, LSTM)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Mul)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Or)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Pow)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, RNN)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Sin)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Sub)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Tan)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Upsample)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 7, Multinomial)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, Xor)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 7, PRelu)>());
  }
};

// Forward declarations for ai.onnx version 8
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 8, Expand);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 8, Max);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 8, Min);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 8, Sum);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 8, Mean);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 8, MaxPool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 8, Scan);

// Iterate over schema from ai.onnx version 8
class OpSet_Onnx_ver8 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 8, Expand)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 8, Min)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 8, Max)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 8, Sum)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 8, Mean)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 8, MaxPool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 8, Scan)>());
  }
};

// Forward declarations for ai.onnx version 9
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, BatchNormalization);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Compress);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, ConstantOfShape);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, EyeLike);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Greater);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Less);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Upsample);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, MaxUnpool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Constant);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, MatMul);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, OneHot);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, PRelu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Gemm);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Flatten);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Sinh);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Cosh);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Asinh);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Acosh);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Atanh);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Shrink);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, IsNaN);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Sign);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Scan);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Erf);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Scatter);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Where);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Cast);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, NonZero);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, TfIdfVectorizer);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, MeanVarianceNormalization);

// Iterate over schema from ai.onnx version 9
class OpSet_Onnx_ver9 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 9, BatchNormalization)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Compress)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 9, ConstantOfShape)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, EyeLike)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Greater)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Less)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Upsample)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, MaxUnpool)>());
    // Add more types' support to Constant/MatMul/PRelu/Gemm/Flatten op.
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Constant)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, MatMul)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, OneHot)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, PRelu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Gemm)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Flatten)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Scatter)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Sinh)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Cosh)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Asinh)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Acosh)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Atanh)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Shrink)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, IsNaN)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Sign)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Scan)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Erf)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Cast)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, Where)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 9, NonZero)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 9, TfIdfVectorizer)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 9, MeanVarianceNormalization)>());
  }
};

// Forward declarations for ai.onnx version 10
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, StringNormalizer);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, Upsample);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, Resize);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, TopK);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, MaxPool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, Mod);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, AveragePool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, Slice);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, ThresholdedRelu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, Dropout);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, MatMulInteger);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, QLinearMatMul);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, ConvInteger);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, QLinearConv);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, QuantizeLinear);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, DequantizeLinear);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, IsInf);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, NonMaxSuppression);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, ReverseSequence);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, RoiAlign);

// Iterate over schema from ai.onnx version 10
class OpSet_Onnx_ver10 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, Upsample)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, Resize)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 10, StringNormalizer)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, TopK)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, MaxPool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, Mod)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 10, AveragePool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, Slice)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 10, ThresholdedRelu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, Dropout)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 10, MatMulInteger)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 10, QLinearMatMul)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 10, ConvInteger)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 10, QLinearConv)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 10, QuantizeLinear)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 10, DequantizeLinear)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, IsInf)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 10, NonMaxSuppression)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 10, ReverseSequence)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 10, RoiAlign)>());
  }
};

// Forward declarations for ai.onnx version 11
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Loop);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, CumSum);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Round);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, BitShift);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Unique);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, TopK);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, DepthToSpace);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Equal);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Constant);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, DynamicQuantizeLinear);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, GatherElements);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ScatterElements);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Scatter);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Clip);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Resize);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Range);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Det);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ScatterND);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, GatherND);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Gather);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, OneHot);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Slice);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Squeeze);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Unsqueeze);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Flatten);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ArgMax);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ArgMin);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ReduceL1);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ReduceL2);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ReduceLogSum);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ReduceLogSumExp);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ReduceMax);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ReduceMean);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ReduceMin);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ReduceProd);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ReduceSum);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ReduceSumSquare);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Compress);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Concat);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Hardmax);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, LogSoftmax);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Softmax);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Scan);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Split);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, AveragePool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, MaxPool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, MaxUnpool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, LpPool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Conv);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ConvTranspose);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, SequenceEmpty);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, SequenceConstruct);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, SequenceInsert);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, SequenceAt);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, SequenceErase);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, SequenceLength);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, SplitToSequence);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ConcatFromSequence);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Pad);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Gemm);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, If);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, NonMaxSuppression);

// Iterate over schema from ai.onnx version 11
class OpSet_Onnx_ver11 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Loop)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, BitShift)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Unique)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, CumSum)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Round)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, TopK)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 11, DepthToSpace)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Equal)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Constant)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 11, DynamicQuantizeLinear)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 11, GatherElements)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 11, ScatterElements)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Scatter)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Clip)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Resize)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Range)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Det)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ScatterND)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, GatherND)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Gather)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, OneHot)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Slice)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Squeeze)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Unsqueeze)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Flatten)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ArgMin)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ArgMax)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ReduceL1)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ReduceL2)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ReduceLogSum)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ReduceLogSumExp)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ReduceMax)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ReduceMean)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ReduceMin)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ReduceProd)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ReduceSum)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ReduceSumSquare)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Compress)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Concat)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Hardmax)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, LogSoftmax)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Softmax)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Scan)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Split)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, AveragePool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, MaxPool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, MaxUnpool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, LpPool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Conv)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ConvTranspose)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, SequenceEmpty)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, SequenceConstruct)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, SequenceInsert)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, SequenceAt)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, SequenceErase)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, SequenceLength)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, SplitToSequence)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, ConcatFromSequence)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Pad)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Gemm)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, If)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, NonMaxSuppression)>());
  }
};

// Forward declarations for ai.onnx version 12
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, ArgMax);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, ArgMin);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, Clip);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, Einsum);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, MaxPool);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, ReduceMax);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, ReduceMin);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, GatherND);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, NegativeLogLikelihoodLoss);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, Dropout);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, Constant);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, Celu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, Max);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, Min);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, LessOrEqual);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, GreaterOrEqual);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, SoftmaxCrossEntropyLoss);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, Pow);

// Iterate over schema from ai.onnx version 12
class OpSet_Onnx_ver12 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, ArgMax)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, ArgMin)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, Clip)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, Einsum)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, MaxPool)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, ReduceMax)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, ReduceMin)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, GatherND)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 12, NegativeLogLikelihoodLoss)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, Dropout)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, Constant)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, Celu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, Max)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, Min)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 12, LessOrEqual)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 12, GreaterOrEqual)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 12, SoftmaxCrossEntropyLoss)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 12, Pow)>());
  }
};
// Forward declarations for ai.onnx version 13
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Constant);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Greater);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Less);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Equal);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Add);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Sub);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Mul);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Div);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Softmax);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, LogSoftmax);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Hardmax);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Mod);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Neg);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Abs);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Reciprocal);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Floor);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Ceil);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Sqrt);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Relu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Exp);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Log);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Tanh);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Pow);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Sigmoid);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Max);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Min);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Sum);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Mean);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Clip);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Gemm);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, MatMul);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Expand);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Sign);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Erf);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, SoftmaxCrossEntropyLoss);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, NegativeLogLikelihoodLoss);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Dropout);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Flatten);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, LRN);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, MeanVarianceNormalization);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ReduceMax);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ReduceMin);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ReduceSum);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ReduceSumSquare);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ReduceMean);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ReduceProd);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ReduceLogSum);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ReduceLogSumExp);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ReduceL1);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ReduceL2);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ArgMax);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ArgMin);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Cast);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Reshape);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Shape);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Size);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Concat);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Split);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Slice);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Transpose);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ScatterND);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ScatterElements);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Gather);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, GatherElements);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Squeeze);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Unsqueeze);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, SpaceToDepth);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, DepthToSpace);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Tile);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Resize);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Identity);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, IsNaN);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, NonZero);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, GatherND);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Pad);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, QuantizeLinear);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, DequantizeLinear);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Loop);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, If);

// Iterate over schema from ai.onnx version 13
class OpSet_Onnx_ver13 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Constant)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Greater)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Less)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Equal)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Add)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Sub)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Mul)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Div)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Softmax)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, LogSoftmax)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Hardmax)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Mod)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Neg)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Abs)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Reciprocal)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Floor)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Ceil)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Sqrt)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Relu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Exp)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Log)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Tanh)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Pow)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Sigmoid)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Max)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Min)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Sum)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Mean)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Clip)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Gemm)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, MatMul)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Expand)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Sign)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Erf)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, SoftmaxCrossEntropyLoss)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, NegativeLogLikelihoodLoss)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Dropout)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Flatten)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, LRN)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, MeanVarianceNormalization)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ReduceMax)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ReduceMin)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ReduceSum)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ReduceSumSquare)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ReduceMean)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ReduceProd)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ReduceLogSum)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ReduceLogSumExp)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ReduceL1)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ReduceL2)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ArgMax)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ArgMin)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Cast)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Reshape)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Shape)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Size)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Concat)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Split)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Slice)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Transpose)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ScatterND)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, ScatterElements)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Gather)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, GatherElements)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Squeeze)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Unsqueeze)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, SpaceToDepth)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, DepthToSpace)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Tile)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Resize)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Identity)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, IsNaN)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, NonZero)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, GatherND)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Pad)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, QuantizeLinear)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, DequantizeLinear)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, Loop)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 13, If)>());
  }
};

// Forward declarations for ai.onnx version 14
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 14, CumSum);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 14, Relu);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 14, Reshape);

// Iterate over schema from ai.onnx version 14
class OpSet_Onnx_ver14 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 14, CumSum)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 14, Relu)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 14, Reshape)>());
  }
};

inline void RegisterOnnxOperatorSetSchema() {
  RegisterOpSetSchema<OpSet_Onnx_ver1>();
  RegisterOpSetSchema<OpSet_Onnx_ver2>();
  RegisterOpSetSchema<OpSet_Onnx_ver3>();
  RegisterOpSetSchema<OpSet_Onnx_ver4>();
  RegisterOpSetSchema<OpSet_Onnx_ver5>();
  RegisterOpSetSchema<OpSet_Onnx_ver6>();
  RegisterOpSetSchema<OpSet_Onnx_ver7>();
  RegisterOpSetSchema<OpSet_Onnx_ver8>();
  RegisterOpSetSchema<OpSet_Onnx_ver9>();
  RegisterOpSetSchema<OpSet_Onnx_ver10>();
  RegisterOpSetSchema<OpSet_Onnx_ver11>();
  RegisterOpSetSchema<OpSet_Onnx_ver12>();
  RegisterOpSetSchema<OpSet_Onnx_ver13>();
  RegisterOpSetSchema<OpSet_Onnx_ver14>();
}

OpSet_Versions ops = OpSet_Versions();

class OpSet_Onnx_ver_latest {
 public:
  static void ForEachSchemaByVersion(std::function<void(OpSchema)> fn, int target_version) { 
    std::vector<OpSchema> class_names = ops.getAllLatestVersion("Onnx", target_version);
    for (auto class_name : class_names) {
      fn(class_name);
    }
  }
};

class OpSet_Onnx_ver_latest_1: public OpSet_Onnx_ver_latest{
 public:
  static void ForEachSchema(std::function<void(OpSchema)> fn) {
    ForEachSchemaByVersion(fn, 1);
  }
};

class OpSet_Onnx_ver_latest_2: public OpSet_Onnx_ver_latest{
 public:
  static void ForEachSchema(std::function<void(OpSchema)> fn) {
    ForEachSchemaByVersion(fn, 2);
  }
};

class OpSet_Onnx_ver_latest_3: public OpSet_Onnx_ver_latest{
 public:
  static void ForEachSchema(std::function<void(OpSchema)> fn) {
    ForEachSchemaByVersion(fn, 3);
  }
};

class OpSet_Onnx_ver_latest_4: public OpSet_Onnx_ver_latest{
 public:
  static void ForEachSchema(std::function<void(OpSchema)> fn) {
    ForEachSchemaByVersion(fn, 4);
  }
};

class OpSet_Onnx_ver_latest_5: public OpSet_Onnx_ver_latest{
 public:
  static void ForEachSchema(std::function<void(OpSchema)> fn) {
    ForEachSchemaByVersion(fn, 5);
  }
};

class OpSet_Onnx_ver_latest_6: public OpSet_Onnx_ver_latest{
 public:
  static void ForEachSchema(std::function<void(OpSchema)> fn) {
    ForEachSchemaByVersion(fn, 6);
  }
};

class OpSet_Onnx_ver_latest_7: public OpSet_Onnx_ver_latest{
 public:
  static void ForEachSchema(std::function<void(OpSchema)> fn) {
    ForEachSchemaByVersion(fn, 7);
  }
};

class OpSet_Onnx_ver_latest_8: public OpSet_Onnx_ver_latest{
 public:
  static void ForEachSchema(std::function<void(OpSchema)> fn) {
    ForEachSchemaByVersion(fn, 8);
  }
};

class OpSet_Onnx_ver_latest_9: public OpSet_Onnx_ver_latest{
 public:
  static void ForEachSchema(std::function<void(OpSchema)> fn) {
    ForEachSchemaByVersion(fn, 9);
  }
};

class OpSet_Onnx_ver_latest_10: public OpSet_Onnx_ver_latest{
 public:
  static void ForEachSchema(std::function<void(OpSchema)> fn) {
    ForEachSchemaByVersion(fn, 10);
  }
};

class OpSet_Onnx_ver_latest_11: public OpSet_Onnx_ver_latest{
 public:
  static void ForEachSchema(std::function<void(OpSchema)> fn) {
    ForEachSchemaByVersion(fn, 11);
  }
};

class OpSet_Onnx_ver_latest_12: public OpSet_Onnx_ver_latest{
 public:
  static void ForEachSchema(std::function<void(OpSchema)> fn) {
    ForEachSchemaByVersion(fn, 12);
  }
};

class OpSet_Onnx_ver_latest_13: public OpSet_Onnx_ver_latest{
 public:
  static void ForEachSchema(std::function<void(OpSchema)> fn) {
    ForEachSchemaByVersion(fn, 13);
  }
};

class OpSet_Onnx_ver_latest_14: public OpSet_Onnx_ver_latest{
 public:
  static void ForEachSchema(std::function<void(OpSchema)> fn) {
    ForEachSchemaByVersion(fn, 14);
  }
};

inline void RegisterOnnxOperatorSetSchema(int target_version) {
  
  ops.addOpset("Abs", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Abs)>());
  ops.addOpset("Add", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Add)>());
  ops.addOpset("And", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, And)>());
  ops.addOpset("ArgMax", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, ArgMax)>());
  ops.addOpset("ArgMin", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, ArgMin)>());
  ops.addOpset("AveragePool", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, AveragePool)>());
  ops.addOpset("BatchNormalization", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, BatchNormalization)>());
  ops.addOpset("Cast", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Cast)>());
  ops.addOpset("Ceil", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Ceil)>());
  ops.addOpset("Clip", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Clip)>());
  ops.addOpset("Concat", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Concat)>());
  ops.addOpset("Constant", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Constant)>());
  ops.addOpset("Conv", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Conv)>());
  ops.addOpset("ConvTranspose", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, ConvTranspose)>());
  ops.addOpset("DepthToSpace", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, DepthToSpace)>());
  ops.addOpset("Div", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Div)>());
  ops.addOpset("Dropout", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Dropout)>());
  ops.addOpset("Elu", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Elu)>());
  ops.addOpset("Equal", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Equal)>());
  ops.addOpset("Exp", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Exp)>());
  ops.addOpset("Flatten", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Flatten)>());
  ops.addOpset("Floor", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Floor)>());
  ops.addOpset("GRU", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, GRU)>());
  ops.addOpset("Gather", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Gather)>());
  ops.addOpset("Gemm", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Gemm)>());
  ops.addOpset("GlobalAveragePool", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, GlobalAveragePool)>());
  ops.addOpset("GlobalLpPool", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, GlobalLpPool)>());
  ops.addOpset("GlobalMaxPool", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, GlobalMaxPool)>());
  ops.addOpset("Greater", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Greater)>());
  ops.addOpset("HardSigmoid", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, HardSigmoid)>());
  ops.addOpset("Hardmax", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Hardmax)>());
  ops.addOpset("Identity", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Identity)>());
  ops.addOpset("If", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, If)>());
  ops.addOpset("InstanceNormalization", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, InstanceNormalization)>());
  ops.addOpset("LRN", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, LRN)>());
  ops.addOpset("LSTM", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, LSTM)>());
  ops.addOpset("LeakyRelu", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, LeakyRelu)>());
  ops.addOpset("Less", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Less)>());
  ops.addOpset("Log", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Log)>());
  ops.addOpset("LogSoftmax", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, LogSoftmax)>());
  ops.addOpset("Loop", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Loop)>());
  ops.addOpset("LpNormalization", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, LpNormalization)>());
  ops.addOpset("LpPool", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, LpPool)>());
  ops.addOpset("MatMul", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, MatMul)>());
  ops.addOpset("Max", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Max)>());
  ops.addOpset("MaxPool", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, MaxPool)>());
  ops.addOpset("MaxRoiPool", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, MaxRoiPool)>());
  ops.addOpset("Mean", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Mean)>());
  ops.addOpset("Min", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Min)>());
  ops.addOpset("Mul", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Mul)>());
  ops.addOpset("Neg", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Neg)>());
  ops.addOpset("Not", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Not)>());
  ops.addOpset("Or", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Or)>());
  ops.addOpset("PRelu", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, PRelu)>());
  ops.addOpset("Pad", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Pad)>());
  ops.addOpset("Pow", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Pow)>());
  ops.addOpset("RNN", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, RNN)>());
  ops.addOpset("RandomNormal", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, RandomNormal)>());
  ops.addOpset("RandomNormalLike", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, RandomNormalLike)>());
  ops.addOpset("RandomUniform", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, RandomUniform)>());
  ops.addOpset("RandomUniformLike", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, RandomUniformLike)>());
  ops.addOpset("Reciprocal", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Reciprocal)>());
  ops.addOpset("ReduceL1", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, ReduceL1)>());
  ops.addOpset("ReduceL2", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, ReduceL2)>());
  ops.addOpset("ReduceLogSum", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, ReduceLogSum)>());
  ops.addOpset("ReduceLogSumExp", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, ReduceLogSumExp)>());
  ops.addOpset("ReduceMax", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, ReduceMax)>());
  ops.addOpset("ReduceMean", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, ReduceMean)>());
  ops.addOpset("ReduceMin", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, ReduceMin)>());
  ops.addOpset("ReduceProd", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, ReduceProd)>());
  ops.addOpset("ReduceSum", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, ReduceSum)>());
  ops.addOpset("ReduceSumSquare", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, ReduceSumSquare)>());
  ops.addOpset("Relu", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Relu)>());
  ops.addOpset("Reshape", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Reshape)>());
  ops.addOpset("Selu", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Selu)>());
  ops.addOpset("Shape", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Shape)>());
  ops.addOpset("Sigmoid", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Sigmoid)>());
  ops.addOpset("Size", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Size)>());
  ops.addOpset("Slice", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Slice)>());
  ops.addOpset("Softmax", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Softmax)>());
  ops.addOpset("Softplus", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Softplus)>());
  ops.addOpset("Softsign", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Softsign)>());
  ops.addOpset("SpaceToDepth", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, SpaceToDepth)>());
  ops.addOpset("Split", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Split)>());
  ops.addOpset("Sqrt", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Sqrt)>());
  ops.addOpset("Squeeze", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Squeeze)>());
  ops.addOpset("Sub", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Sub)>());
  ops.addOpset("Sum", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Sum)>());
  ops.addOpset("Tanh", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Tanh)>());
  ops.addOpset("Tile", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Tile)>());
  ops.addOpset("TopK", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, TopK)>());
  ops.addOpset("Transpose", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Transpose)>());
  ops.addOpset("Unsqueeze", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Unsqueeze)>());
  ops.addOpset("Upsample", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Upsample)>());
  ops.addOpset("Xor", "Onnx",  1, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  1, Xor)>());
  ops.addOpset("GlobalLpPool", "Onnx",  2, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  2, GlobalLpPool)>());
  ops.addOpset("LpPool", "Onnx",  2, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  2, LpPool)>());
  ops.addOpset("Pad", "Onnx",  2, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  2, Pad)>());
  ops.addOpset("Split", "Onnx",  2, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  2, Split)>());
  ops.addOpset("GRU", "Onnx",  3, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  3, GRU)>());
  ops.addOpset("Concat", "Onnx",  4, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  4, Concat)>());
  ops.addOpset("Reshape", "Onnx",  5, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  5, Reshape)>());
  ops.addOpset("Abs", "Onnx",  6, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  6, Abs)>());
  ops.addOpset("Add", "Onnx",  6, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  6, Add)>());
  ops.addOpset("BatchNormalization", "Onnx",  6, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  6, BatchNormalization)>());
  ops.addOpset("Cast", "Onnx",  6, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  6, Cast)>());
  ops.addOpset("Ceil", "Onnx",  6, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  6, Ceil)>());
  ops.addOpset("Clip", "Onnx",  6, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  6, Clip)>());
  ops.addOpset("Div", "Onnx",  6, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  6, Div)>());
  ops.addOpset("Dropout", "Onnx",  6, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  6, Dropout)>());
  ops.addOpset("Elu", "Onnx",  6, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  6, Elu)>());
  ops.addOpset("Exp", "Onnx",  6, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  6, Exp)>());
  ops.addOpset("Floor", "Onnx",  6, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  6, Floor)>());
  ops.addOpset("Gemm", "Onnx",  6, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  6, Gemm)>());
  ops.addOpset("HardSigmoid", "Onnx",  6, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  6, HardSigmoid)>());
  ops.addOpset("InstanceNormalization", "Onnx",  6, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  6, InstanceNormalization)>());
  ops.addOpset("LeakyRelu", "Onnx",  6, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  6, LeakyRelu)>());
  ops.addOpset("Log", "Onnx",  6, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  6, Log)>());
  ops.addOpset("Max", "Onnx",  6, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  6, Max)>());
  ops.addOpset("Mean", "Onnx",  6, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  6, Mean)>());
  ops.addOpset("Min", "Onnx",  6, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  6, Min)>());
  ops.addOpset("Mul", "Onnx",  6, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  6, Mul)>());
  ops.addOpset("Neg", "Onnx",  6, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  6, Neg)>());
  ops.addOpset("PRelu", "Onnx",  6, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  6, PRelu)>());
  ops.addOpset("Reciprocal", "Onnx",  6, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  6, Reciprocal)>());
  ops.addOpset("Relu", "Onnx",  6, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  6, Relu)>());
  ops.addOpset("Selu", "Onnx",  6, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  6, Selu)>());
  ops.addOpset("Sigmoid", "Onnx",  6, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  6, Sigmoid)>());
  ops.addOpset("Sqrt", "Onnx",  6, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  6, Sqrt)>());
  ops.addOpset("Sub", "Onnx",  6, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  6, Sub)>());
  ops.addOpset("Sum", "Onnx",  6, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  6, Sum)>());
  ops.addOpset("Tanh", "Onnx",  6, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  6, Tanh)>());
  ops.addOpset("Tile", "Onnx",  6, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  6, Tile)>());
  ops.addOpset("Acos", "Onnx",  7, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  7, Acos)>());
  ops.addOpset("Add", "Onnx",  7, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  7, Add)>());
  ops.addOpset("And", "Onnx",  7, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  7, And)>());
  ops.addOpset("Asin", "Onnx",  7, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  7, Asin)>());
  ops.addOpset("Atan", "Onnx",  7, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  7, Atan)>());
  ops.addOpset("AveragePool", "Onnx",  7, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  7, AveragePool)>());
  ops.addOpset("BatchNormalization", "Onnx",  7, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  7, BatchNormalization)>());
  ops.addOpset("Cos", "Onnx",  7, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  7, Cos)>());
  ops.addOpset("Div", "Onnx",  7, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  7, Div)>());
  ops.addOpset("Dropout", "Onnx",  7, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  7, Dropout)>());
  ops.addOpset("Equal", "Onnx",  7, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  7, Equal)>());
  ops.addOpset("Gemm", "Onnx",  7, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  7, Gemm)>());
  ops.addOpset("Greater", "Onnx",  7, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  7, Greater)>());
  ops.addOpset("GRU", "Onnx",  7, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  7, GRU)>());
  ops.addOpset("Less", "Onnx",  7, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  7, Less)>());
  ops.addOpset("LSTM", "Onnx",  7, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  7, LSTM)>());
  ops.addOpset("Mul", "Onnx",  7, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  7, Mul)>());
  ops.addOpset("Or", "Onnx",  7, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  7, Or)>());
  ops.addOpset("Pow", "Onnx",  7, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  7, Pow)>());
  ops.addOpset("RNN", "Onnx",  7, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  7, RNN)>());
  ops.addOpset("Sin", "Onnx",  7, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  7, Sin)>());
  ops.addOpset("Sub", "Onnx",  7, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  7, Sub)>());
  ops.addOpset("Tan", "Onnx",  7, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  7, Tan)>());
  ops.addOpset("Upsample", "Onnx",  7, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  7, Upsample)>());
  ops.addOpset("Multinomial", "Onnx",  7, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  7, Multinomial)>());
  ops.addOpset("Xor", "Onnx",  7, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  7, Xor)>());
  ops.addOpset("PRelu", "Onnx",  7, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  7, PRelu)>());
  ops.addOpset("Expand", "Onnx",  8, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  8, Expand)>());
  ops.addOpset("Max", "Onnx",  8, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  8, Max)>());
  ops.addOpset("Min", "Onnx",  8, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  8, Min)>());
  ops.addOpset("Sum", "Onnx",  8, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  8, Sum)>());
  ops.addOpset("Mean", "Onnx",  8, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  8, Mean)>());
  ops.addOpset("MaxPool", "Onnx",  8, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  8, MaxPool)>());
  ops.addOpset("Scan", "Onnx",  8, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  8, Scan)>());
  ops.addOpset("BatchNormalization", "Onnx",  9, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  9, BatchNormalization)>());
  ops.addOpset("Compress", "Onnx",  9, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  9, Compress)>());
  ops.addOpset("ConstantOfShape", "Onnx",  9, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  9, ConstantOfShape)>());
  ops.addOpset("EyeLike", "Onnx",  9, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  9, EyeLike)>());
  ops.addOpset("Greater", "Onnx",  9, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  9, Greater)>());
  ops.addOpset("Less", "Onnx",  9, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  9, Less)>());
  ops.addOpset("Upsample", "Onnx",  9, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  9, Upsample)>());
  ops.addOpset("MaxUnpool", "Onnx",  9, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  9, MaxUnpool)>());
  ops.addOpset("Constant", "Onnx",  9, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  9, Constant)>());
  ops.addOpset("MatMul", "Onnx",  9, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  9, MatMul)>());
  ops.addOpset("OneHot", "Onnx",  9, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  9, OneHot)>());
  ops.addOpset("PRelu", "Onnx",  9, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  9, PRelu)>());
  ops.addOpset("Gemm", "Onnx",  9, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  9, Gemm)>());
  ops.addOpset("Flatten", "Onnx",  9, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  9, Flatten)>());
  ops.addOpset("Sinh", "Onnx",  9, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  9, Sinh)>());
  ops.addOpset("Cosh", "Onnx",  9, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  9, Cosh)>());
  ops.addOpset("Asinh", "Onnx",  9, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  9, Asinh)>());
  ops.addOpset("Acosh", "Onnx",  9, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  9, Acosh)>());
  ops.addOpset("Atanh", "Onnx",  9, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  9, Atanh)>());
  ops.addOpset("Shrink", "Onnx",  9, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  9, Shrink)>());
  ops.addOpset("IsNaN", "Onnx",  9, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  9, IsNaN)>());
  ops.addOpset("Sign", "Onnx",  9, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  9, Sign)>());
  ops.addOpset("Scan", "Onnx",  9, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  9, Scan)>());
  ops.addOpset("Erf", "Onnx",  9, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  9, Erf)>());
  ops.addOpset("Scatter", "Onnx",  9, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  9, Scatter)>());
  ops.addOpset("Where", "Onnx",  9, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  9, Where)>());
  ops.addOpset("Cast", "Onnx",  9, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  9, Cast)>());
  ops.addOpset("NonZero", "Onnx",  9, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  9, NonZero)>());
  ops.addOpset("TfIdfVectorizer", "Onnx",  9, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  9, TfIdfVectorizer)>());
  ops.addOpset("MeanVarianceNormalization", "Onnx",  9, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  9, MeanVarianceNormalization)>());
  ops.addOpset("StringNormalizer", "Onnx",  10, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  10, StringNormalizer)>());
  ops.addOpset("Upsample", "Onnx",  10, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  10, Upsample)>());
  ops.addOpset("Resize", "Onnx",  10, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  10, Resize)>());
  ops.addOpset("TopK", "Onnx",  10, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  10, TopK)>());
  ops.addOpset("MaxPool", "Onnx",  10, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  10, MaxPool)>());
  ops.addOpset("Mod", "Onnx",  10, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  10, Mod)>());
  ops.addOpset("AveragePool", "Onnx",  10, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  10, AveragePool)>());
  ops.addOpset("Slice", "Onnx",  10, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  10, Slice)>());
  ops.addOpset("ThresholdedRelu", "Onnx",  10, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  10, ThresholdedRelu)>());
  ops.addOpset("Dropout", "Onnx",  10, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  10, Dropout)>());
  ops.addOpset("MatMulInteger", "Onnx",  10, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  10, MatMulInteger)>());
  ops.addOpset("QLinearMatMul", "Onnx",  10, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  10, QLinearMatMul)>());
  ops.addOpset("ConvInteger", "Onnx",  10, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  10, ConvInteger)>());
  ops.addOpset("QLinearConv", "Onnx",  10, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  10, QLinearConv)>());
  ops.addOpset("QuantizeLinear", "Onnx",  10, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  10, QuantizeLinear)>());
  ops.addOpset("DequantizeLinear", "Onnx",  10, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  10, DequantizeLinear)>());
  ops.addOpset("IsInf", "Onnx",  10, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  10, IsInf)>());
  ops.addOpset("NonMaxSuppression", "Onnx",  10, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  10, NonMaxSuppression)>());
  ops.addOpset("ReverseSequence", "Onnx",  10, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  10, ReverseSequence)>());
  ops.addOpset("RoiAlign", "Onnx",  10, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  10, RoiAlign)>());
  ops.addOpset("Loop", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, Loop)>());
  ops.addOpset("CumSum", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, CumSum)>());
  ops.addOpset("Round", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, Round)>());
  ops.addOpset("BitShift", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, BitShift)>());
  ops.addOpset("Unique", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, Unique)>());
  ops.addOpset("TopK", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, TopK)>());
  ops.addOpset("DepthToSpace", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, DepthToSpace)>());
  ops.addOpset("Equal", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, Equal)>());
  ops.addOpset("Constant", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, Constant)>());
  ops.addOpset("DynamicQuantizeLinear", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, DynamicQuantizeLinear)>());
  ops.addOpset("GatherElements", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, GatherElements)>());
  ops.addOpset("ScatterElements", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, ScatterElements)>());
  ops.addOpset("Scatter", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, Scatter)>());
  ops.addOpset("Clip", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, Clip)>());
  ops.addOpset("Resize", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, Resize)>());
  ops.addOpset("Range", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, Range)>());
  ops.addOpset("Det", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, Det)>());
  ops.addOpset("ScatterND", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, ScatterND)>());
  ops.addOpset("GatherND", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, GatherND)>());
  ops.addOpset("Gather", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, Gather)>());
  ops.addOpset("OneHot", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, OneHot)>());
  ops.addOpset("Slice", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, Slice)>());
  ops.addOpset("Squeeze", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, Squeeze)>());
  ops.addOpset("Unsqueeze", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, Unsqueeze)>());
  ops.addOpset("Flatten", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, Flatten)>());
  ops.addOpset("ArgMax", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, ArgMax)>());
  ops.addOpset("ArgMin", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, ArgMin)>());
  ops.addOpset("ReduceL1", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, ReduceL1)>());
  ops.addOpset("ReduceL2", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, ReduceL2)>());
  ops.addOpset("ReduceLogSum", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, ReduceLogSum)>());
  ops.addOpset("ReduceLogSumExp", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, ReduceLogSumExp)>());
  ops.addOpset("ReduceMax", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, ReduceMax)>());
  ops.addOpset("ReduceMean", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, ReduceMean)>());
  ops.addOpset("ReduceMin", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, ReduceMin)>());
  ops.addOpset("ReduceProd", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, ReduceProd)>());
  ops.addOpset("ReduceSum", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, ReduceSum)>());
  ops.addOpset("ReduceSumSquare", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, ReduceSumSquare)>());
  ops.addOpset("Compress", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, Compress)>());
  ops.addOpset("Concat", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, Concat)>());
  ops.addOpset("Hardmax", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, Hardmax)>());
  ops.addOpset("LogSoftmax", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, LogSoftmax)>());
  ops.addOpset("Softmax", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, Softmax)>());
  ops.addOpset("Scan", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, Scan)>());
  ops.addOpset("Split", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, Split)>());
  ops.addOpset("AveragePool", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, AveragePool)>());
  ops.addOpset("MaxPool", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, MaxPool)>());
  ops.addOpset("MaxUnpool", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, MaxUnpool)>());
  ops.addOpset("LpPool", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, LpPool)>());
  ops.addOpset("Conv", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, Conv)>());
  ops.addOpset("ConvTranspose", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, ConvTranspose)>());
  ops.addOpset("SequenceEmpty", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, SequenceEmpty)>());
  ops.addOpset("SequenceConstruct", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, SequenceConstruct)>());
  ops.addOpset("SequenceInsert", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, SequenceInsert)>());
  ops.addOpset("SequenceAt", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, SequenceAt)>());
  ops.addOpset("SequenceErase", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, SequenceErase)>());
  ops.addOpset("SequenceLength", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, SequenceLength)>());
  ops.addOpset("SplitToSequence", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, SplitToSequence)>());
  ops.addOpset("ConcatFromSequence", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, ConcatFromSequence)>());
  ops.addOpset("Pad", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, Pad)>());
  ops.addOpset("Gemm", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, Gemm)>());
  ops.addOpset("If", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, If)>());
  ops.addOpset("NonMaxSuppression", "Onnx",  11, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  11, NonMaxSuppression)>());
  ops.addOpset("ArgMax", "Onnx",  12, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  12, ArgMax)>());
  ops.addOpset("ArgMin", "Onnx",  12, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  12, ArgMin)>());
  ops.addOpset("Clip", "Onnx",  12, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  12, Clip)>());
  ops.addOpset("Einsum", "Onnx",  12, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  12, Einsum)>());
  ops.addOpset("MaxPool", "Onnx",  12, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  12, MaxPool)>());
  ops.addOpset("ReduceMax", "Onnx",  12, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  12, ReduceMax)>());
  ops.addOpset("ReduceMin", "Onnx",  12, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  12, ReduceMin)>());
  ops.addOpset("GatherND", "Onnx",  12, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  12, GatherND)>());
  ops.addOpset("NegativeLogLikelihoodLoss", "Onnx",  12, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  12, NegativeLogLikelihoodLoss)>());
  ops.addOpset("Dropout", "Onnx",  12, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  12, Dropout)>());
  ops.addOpset("Constant", "Onnx",  12, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  12, Constant)>());
  ops.addOpset("Celu", "Onnx",  12, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  12, Celu)>());
  ops.addOpset("Max", "Onnx",  12, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  12, Max)>());
  ops.addOpset("Min", "Onnx",  12, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  12, Min)>());
  ops.addOpset("LessOrEqual", "Onnx",  12, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  12, LessOrEqual)>());
  ops.addOpset("GreaterOrEqual", "Onnx",  12, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  12, GreaterOrEqual)>());
  ops.addOpset("SoftmaxCrossEntropyLoss", "Onnx",  12, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  12, SoftmaxCrossEntropyLoss)>());
  ops.addOpset("Pow", "Onnx",  12, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  12, Pow)>());
  ops.addOpset("Constant", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Constant)>());
  ops.addOpset("Greater", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Greater)>());
  ops.addOpset("Less", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Less)>());
  ops.addOpset("Equal", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Equal)>());
  ops.addOpset("Add", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Add)>());
  ops.addOpset("Sub", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Sub)>());
  ops.addOpset("Mul", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Mul)>());
  ops.addOpset("Div", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Div)>());
  ops.addOpset("Softmax", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Softmax)>());
  ops.addOpset("LogSoftmax", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, LogSoftmax)>());
  ops.addOpset("Hardmax", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Hardmax)>());
  ops.addOpset("Mod", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Mod)>());
  ops.addOpset("Neg", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Neg)>());
  ops.addOpset("Abs", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Abs)>());
  ops.addOpset("Reciprocal", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Reciprocal)>());
  ops.addOpset("Floor", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Floor)>());
  ops.addOpset("Ceil", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Ceil)>());
  ops.addOpset("Sqrt", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Sqrt)>());
  ops.addOpset("Relu", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Relu)>());
  ops.addOpset("Exp", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Exp)>());
  ops.addOpset("Log", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Log)>());
  ops.addOpset("Tanh", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Tanh)>());
  ops.addOpset("Pow", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Pow)>());
  ops.addOpset("Sigmoid", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Sigmoid)>());
  ops.addOpset("Max", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Max)>());
  ops.addOpset("Min", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Min)>());
  ops.addOpset("Sum", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Sum)>());
  ops.addOpset("Mean", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Mean)>());
  ops.addOpset("Clip", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Clip)>());
  ops.addOpset("Gemm", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Gemm)>());
  ops.addOpset("MatMul", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, MatMul)>());
  ops.addOpset("Expand", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Expand)>());
  ops.addOpset("Sign", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Sign)>());
  ops.addOpset("Erf", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Erf)>());
  ops.addOpset("SoftmaxCrossEntropyLoss", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, SoftmaxCrossEntropyLoss)>());
  ops.addOpset("NegativeLogLikelihoodLoss", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, NegativeLogLikelihoodLoss)>());
  ops.addOpset("Dropout", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Dropout)>());
  ops.addOpset("Flatten", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Flatten)>());
  ops.addOpset("LRN", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, LRN)>());
  ops.addOpset("MeanVarianceNormalization", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, MeanVarianceNormalization)>());
  ops.addOpset("ReduceMax", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, ReduceMax)>());
  ops.addOpset("ReduceMin", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, ReduceMin)>());
  ops.addOpset("ReduceSum", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, ReduceSum)>());
  ops.addOpset("ReduceSumSquare", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, ReduceSumSquare)>());
  ops.addOpset("ReduceMean", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, ReduceMean)>());
  ops.addOpset("ReduceProd", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, ReduceProd)>());
  ops.addOpset("ReduceLogSum", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, ReduceLogSum)>());
  ops.addOpset("ReduceLogSumExp", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, ReduceLogSumExp)>());
  ops.addOpset("ReduceL1", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, ReduceL1)>());
  ops.addOpset("ReduceL2", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, ReduceL2)>());
  ops.addOpset("ArgMax", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, ArgMax)>());
  ops.addOpset("ArgMin", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, ArgMin)>());
  ops.addOpset("Cast", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Cast)>());
  ops.addOpset("Reshape", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Reshape)>());
  ops.addOpset("Shape", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Shape)>());
  ops.addOpset("Size", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Size)>());
  ops.addOpset("Concat", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Concat)>());
  ops.addOpset("Split", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Split)>());
  ops.addOpset("Slice", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Slice)>());
  ops.addOpset("Transpose", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Transpose)>());
  ops.addOpset("ScatterND", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, ScatterND)>());
  ops.addOpset("ScatterElements", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, ScatterElements)>());
  ops.addOpset("Gather", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Gather)>());
  ops.addOpset("GatherElements", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, GatherElements)>());
  ops.addOpset("Squeeze", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Squeeze)>());
  ops.addOpset("Unsqueeze", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Unsqueeze)>());
  ops.addOpset("SpaceToDepth", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, SpaceToDepth)>());
  ops.addOpset("DepthToSpace", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, DepthToSpace)>());
  ops.addOpset("Tile", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Tile)>());
  ops.addOpset("Resize", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Resize)>());
  ops.addOpset("Identity", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Identity)>());
  ops.addOpset("IsNaN", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, IsNaN)>());
  ops.addOpset("NonZero", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, NonZero)>());
  ops.addOpset("GatherND", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, GatherND)>());
  ops.addOpset("Pad", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Pad)>());
  ops.addOpset("QuantizeLinear", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, QuantizeLinear)>());
  ops.addOpset("DequantizeLinear", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, DequantizeLinear)>());
  ops.addOpset("Loop", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, Loop)>());
  ops.addOpset("If", "Onnx",  13, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  13, If)>());
  ops.addOpset("CumSum", "Onnx",  14, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  14, CumSum)>());
  ops.addOpset("Relu", "Onnx",  14, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  14, Relu)>());
  ops.addOpset("Reshape", "Onnx",  14, GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx,  14, Reshape)>());
  
  // according to the target_version, register opset schema 
  switch (target_version) {
    case 1:
      RegisterOpSetSchema<OpSet_Onnx_ver_latest_1>();
      break;
    case 2:
      RegisterOpSetSchema<OpSet_Onnx_ver_latest_2>();
      break;
    case 3:
      RegisterOpSetSchema<OpSet_Onnx_ver_latest_3>();
      break;
    case 4:
      RegisterOpSetSchema<OpSet_Onnx_ver_latest_4>();
      break;
    case 5:
      RegisterOpSetSchema<OpSet_Onnx_ver_latest_5>();
      break;
    case 6:
      RegisterOpSetSchema<OpSet_Onnx_ver_latest_6>();
      break;
    case 7:
      RegisterOpSetSchema<OpSet_Onnx_ver_latest_7>();
      break;
    case 8:
      RegisterOpSetSchema<OpSet_Onnx_ver_latest_8>();
      break;
    case 9:
      RegisterOpSetSchema<OpSet_Onnx_ver_latest_9>();
      break;
    case 10:
      RegisterOpSetSchema<OpSet_Onnx_ver_latest_10>();
      break;
    case 11:
      RegisterOpSetSchema<OpSet_Onnx_ver_latest_11>();
      break;
    case 12:
      RegisterOpSetSchema<OpSet_Onnx_ver_latest_12>();
      break;
    case 13:
      RegisterOpSetSchema<OpSet_Onnx_ver_latest_13>();
      break;
    case 14:
      RegisterOpSetSchema<OpSet_Onnx_ver_latest_14>();
      break;
    default:
      break;
  }

}

} // namespace ONNX_NAMESPACE
