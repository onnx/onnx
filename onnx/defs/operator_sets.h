// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#pragma once

#include "onnx/defs/schema.h"

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
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 10, RoiAlign)>());
  }
};

class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Loop);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, CumSum);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Round);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, BitShift);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Equal);

// Iterate over schema from ai.onnx version 10
class OpSet_Onnx_ver11 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Loop)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, CumSum)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Round)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(
           Onnx, 11, BitShift)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(Onnx, 11, Equal)>());
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
}

} // namespace ONNX_NAMESPACE
