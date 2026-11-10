```toml
name = "GreaterOrEqual"
domain = "ai.onnx"

[type_constraints.T]
types = "all_numeric_types_ir4"
desc = "Constrain input types to all numeric tensors."

[type_constraints.T1]
types = ["tensor(bool)"]
desc = "Constrain output to boolean tensor."

[function_bodies]
16 = """
{
   O1 = Greater (A, B)
   O2 = Equal (A, B)
   C = Or (O1, O2)
}
"""
```
