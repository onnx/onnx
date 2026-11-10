```toml
name = "TreeEnsembleClassifier"
domain = "ai.onnx.ml"
shape_inference = "defs::traditionalml::utils::TreeEnsembleClassifierShapeInference_opset1"

[attributes.base_values]
type = "FLOATS"
desc = "Base values for classification, added to final class score; the size must be the same as the classes or can be left unassigned (assumed 0)"

[attributes.class_ids]
type = "INTS"
desc = "The index of the class list that each weight is for."

[attributes.class_nodeids]
type = "INTS"
desc = "node id that this weight is for."

[attributes.class_treeids]
type = "INTS"
desc = "The id of the tree that this node is in."

[attributes.class_weights]
type = "FLOATS"
desc = "The weight for the class in class_id."

[attributes.classlabels_int64s]
type = "INTS"
desc = "Class labels if using integer labels.<br>One and only one of the 'classlabels_*' attributes must be defined."

[attributes.classlabels_strings]
type = "STRINGS"
desc = "Class labels if using string labels.<br>One and only one of the 'classlabels_*' attributes must be defined."

[attributes.nodes_falsenodeids]
type = "INTS"
desc = "Child node if expression is false."

[attributes.nodes_featureids]
type = "INTS"
desc = "Feature id for each node."

[attributes.nodes_hitrates]
type = "FLOATS"
desc = "Popularity of each node, used for performance and may be omitted."

[attributes.nodes_missing_value_tracks_true]
type = "INTS"
desc = "For each node, define what to do in the presence of a missing value: if a value is missing (NaN), use the 'true' or 'false' branch based on the value in this array.<br>This attribute may be left undefined, and the default value is false (0) for all nodes."

[attributes.nodes_modes]
type = "STRINGS"
desc = "The node kind, that is, the comparison to make at the node. There is no comparison to make at a leaf node.<br>One of 'BRANCH_LEQ', 'BRANCH_LT', 'BRANCH_GTE', 'BRANCH_GT', 'BRANCH_EQ', 'BRANCH_NEQ', 'LEAF'"

[attributes.nodes_nodeids]
type = "INTS"
desc = "Node id for each node. Ids may restart at zero for each tree, but it not required to."

[attributes.nodes_treeids]
type = "INTS"
desc = "Tree id for each node."

[attributes.nodes_truenodeids]
type = "INTS"
desc = "Child node if expression is true."

[attributes.nodes_values]
type = "FLOATS"
desc = "Thresholds to do the splitting on for each node."

[attributes.post_transform]
type = "STRING"
desc = "Indicates the transform to apply to the score. <br> One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT.'"
default = "NONE"

[[inputs]]
name = "X"
type = "T1"
desc = "Input of shape [N,F]"

[[outputs]]
name = "Y"
type = "T2"
desc = "N, Top class for each point"

[[outputs]]
name = "Z"
type = "tensor(float)"
desc = "The class score for each class, for each point, a tensor of shape [N,E]."

[type_constraints.T1]
types = ["tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)"]
desc = "The input type must be a tensor of a numeric type."

[type_constraints.T2]
types = ["tensor(string)", "tensor(int64)"]
desc = "The output type will be a tensor of strings or integers, depending on which of the classlabels_* attributes is used."
```

Tree Ensemble classifier.  Returns the top class for each of N inputs.<br>
The attributes named 'nodes_X' form a sequence of tuples, associated by
index into the sequences, which must all be of equal length. These tuples
define the nodes.<br>
Similarly, all fields prefixed with 'class_' are tuples of votes at the leaves.
A leaf may have multiple votes, where each vote is weighted by
the associated class_weights index.<br>
One and only one of classlabels_strings or classlabels_int64s
will be defined. The class_ids are indices into this list.
