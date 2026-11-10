```toml
name = "Mod"
domain = "ai.onnx"

[attributes.fmod]
type = "INT"
desc = "Whether the operator should behave like fmod (default=0 meaning it will do integer mods); Set this to 1 to force fmod treatment"
default = 0

[[inputs]]
name = "A"
type = "T"
desc = "Dividend tensor"

[[inputs]]
name = "B"
type = "T"
desc = "Divisor tensor"

[[outputs]]
name = "C"
type = "T"
desc = "Remainder tensor"

[type_constraints.T]
types = "all_numeric_types"
desc = "Constrain input and output types to high-precision numeric tensors."
```

Performs element-wise binary modulus (with Numpy-style broadcasting support).
  The sign of the remainder is the same as that of the Divisor.

  Mod operator can also behave like C fmod() or numpy.fmod. In this case, the sign of the remainder however, will be the same as the Dividend
  (in contrast to integer mod). To force a behavior like numpy.fmod() an 'fmod' Attribute is provided.
  This attribute is set to 0 by default causing the behavior to be like integer mod.
  Setting this attribute to 1 causes the remainder to be calculated similar to that of numpy.fmod().

  If the input type is floating point, then `fmod` attribute must be set to 1.

  In case of dividend being zero, the results will be platform dependent.

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
