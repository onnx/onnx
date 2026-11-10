```toml
name = "HammingWindow"
domain = "ai.onnx"
shape_inference = "defs::math::utils::CosineSumWindowOpShapeInference"

[attributes.output_datatype]
type = "INT"
desc = "The data type of the output tensor. Strictly must be one of the values from DataType enum in TensorProto whose values correspond to T2. The default value is 1 = FLOAT. "
default = 1

[attributes.periodic]
type = "INT"
desc = "If 1, returns a window to be used as periodic function. If 0, return a symmetric window. When 'periodic' is specified, hann computes a window of length size + 1 and returns the first size points. The default value is 1. "
default = 1

[[inputs]]
name = "size"
type = "T1"
desc = "A scalar value indicating the length of the window."
differentiation = "non_differentiable"

[[outputs]]
name = "output"
type = "T2"
desc = "A Hamming window with length: size. The output has the shape: [size]."
differentiation = "non_differentiable"

[type_constraints.T1]
types = ["tensor(int32)", "tensor(int64)"]
desc = "Constrain the input size to int32_t or int64_t."

[type_constraints.T2]
types = "all_numeric_types_ir4"
desc = "Constrain output types to numeric tensors."

[function_bodies]
17 = """
{
   A0 = Constant <value: tensor = float {0.54347825}> ()
   A1 = Constant <value: tensor = float {0.45652175}> ()
   A2 = Constant <value: tensor = float {0}> ()
   Zero = Constant <value: tensor = float {0}> ()
   One = Constant <value: tensor = float {1}> ()
   Two = Constant <value: tensor = float {2}> ()
   Tau = Constant <value: tensor = float {6.2831855}> ()
   Periodic_Size_FP = Cast <to: int = 1> (size)
   Symmetric_Size_FP = Sub (Periodic_Size_FP, One)
   IsPeriodic = Constant <value_int: int = @periodic> ()
   IsPeriodic_FP = Cast <to: int = 1> (IsPeriodic)
   IsSymmetric_FP = Sub (One, IsPeriodic_FP)
   Periodic_Component = Mul (Periodic_Size_FP, IsPeriodic_FP)
   Symmetric_Component = Mul (Symmetric_Size_FP, IsSymmetric_FP)
   Size_FP = Add (Periodic_Component, Symmetric_Component)
   AngularIncrement = Div (Tau, Size_FP)
   Range = Range (Zero, Periodic_Size_FP, One)
   RangeAngular = Mul (Range, AngularIncrement)
   TwoRangeAngular = Mul (RangeAngular, Two)
   CosTwoRangeAngular = Cos (TwoRangeAngular)
   A2_Component = Mul (A2, CosTwoRangeAngular)
   CosRangeAngular = Cos (RangeAngular)
   A1_Component = Mul (A1, CosRangeAngular)
   Temp0 = Sub (A0, A1_Component)
   Temp1 = Add (Temp0, A2_Component)
   output = Cast <to: int = @output_datatype> (Temp1)
}
"""
```

Generates a Hamming window as described in the paper https://ieeexplore.ieee.org/document/1455106.
