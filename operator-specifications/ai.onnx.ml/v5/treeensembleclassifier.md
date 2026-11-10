```toml
name = "TreeEnsembleClassifier"
domain = "ai.onnx.ml"
deprecated = true
```

This operator is DEPRECATED. Please use TreeEnsemble with provides similar functionality.
In order to determine the top class, the ArgMax node can be applied to the output of TreeEnsemble.
To encode class labels, use a LabelEncoder operator.
Tree Ensemble classifier. Returns the top class for each of N inputs.<br>
The attributes named 'nodes_X' form a sequence of tuples, associated by
index into the sequences, which must all be of equal length. These tuples
define the nodes.<br>
Similarly, all fields prefixed with 'class_' are tuples of votes at the leaves.
A leaf may have multiple votes, where each vote is weighted by
the associated class_weights index.<br>
One and only one of classlabels_strings or classlabels_int64s
will be defined. The class_ids are indices into this list.
All fields ending with <i>_as_tensor</i> can be used instead of the
same parameter without the suffix if the element type is double and not float.
