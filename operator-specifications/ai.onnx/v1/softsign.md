```toml
name = "Softsign"
domain = "ai.onnx"
shape_inference = "propagateShapeAndTypeFromFirstInput"

[[inputs]]
name = "input"
type = "T"
desc = "Input tensor"
differentiation = "differentiable"

[[outputs]]
name = "output"
type = "T"
desc = "The softsign (x/(1+|x|)) values of the input tensor computed element-wise"
differentiation = "differentiable"

[type_constraints.T]
types = ["tensor(float16)", "tensor(float)", "tensor(double)"]
desc = "Constrain input and output types to float tensors."

[function_bodies]
18 = """
{
   One = Constant <value: tensor = float {1}> ()
   OneCast = CastLike (One, input)
   AbsInput = Abs (input)
   OneAddAbsInput = Add (OneCast, AbsInput)
   output = Div (input, OneAddAbsInput)
}
"""
```

Calculates the softsign (x/(1+|x|)) of the given input tensor element-wise.
