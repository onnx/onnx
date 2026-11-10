```toml
name = "DictVectorizer"
domain = "ai.onnx.ml"
shape_inference = "defs::traditionalml::utils::DictVectorizerShapeInference_opset1"

[attributes.int64_vocabulary]
type = "INTS"
desc = "An integer vocabulary array.<br>One and only one of the vocabularies must be defined."

[attributes.string_vocabulary]
type = "STRINGS"
desc = "A string vocabulary array.<br>One and only one of the vocabularies must be defined."

[[inputs]]
name = "X"
type = "T1"
desc = "A dictionary."

[[outputs]]
name = "Y"
type = "T2"
desc = "A 1-D tensor holding values from the input dictionary."

[type_constraints.T1]
types = [
    "map(string, int64)",
    "map(int64, string)",
    "map(int64, float)",
    "map(int64, double)",
    "map(string, float)",
    "map(string, double)",
]
desc = "The input must be a map from strings or integers to either strings or a numeric type. The key and value types cannot be the same."

[type_constraints.T2]
types = ["tensor(int64)", "tensor(float)", "tensor(double)", "tensor(string)"]
desc = "The output will be a tensor of the value type of the input map. It's shape will be [1,C], where C is the length of the input dictionary."
```

Uses an index mapping to convert a dictionary to an array.<br>
Given a dictionary, each key is looked up in the vocabulary attribute corresponding to
the key type. The index into the vocabulary array at which the key is found is then
used to index the output 1-D tensor 'Y' and insert into it the value found in the dictionary 'X'.<br>
The key type of the input map must correspond to the element type of the defined vocabulary attribute.
Therefore, the output array will be equal in length to the index mapping vector parameter.
All keys in the input dictionary must be present in the index mapping vector.
For each item in the input dictionary, insert its value in the output array.
Any keys not present in the input dictionary, will be zero in the output array.<br>
For example: if the ``string_vocabulary`` parameter is set to ``["a", "c", "b", "z"]``,
then an input of ``{"a": 4, "c": 8}`` will produce an output of ``[4, 8, 0, 0]``.
