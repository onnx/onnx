```toml
name = "TreeEnsemble"
domain = "ai.onnx.ml"

[attributes.aggregate_function]
type = "INT"
desc = "Defines how to aggregate leaf values within a target. <br>One of 'AVERAGE' (0) 'SUM' (1) 'MIN' (2) 'MAX (3) defaults to 'SUM' (1)"
default = 1

[attributes.leaf_targetids]
type = "INTS"
desc = "The index of the target that this leaf contributes to (this must be in range `[0, n_targets)`)."
required = true

[attributes.leaf_weights]
type = "TENSOR"
desc = "The weight for each leaf."
required = true

[attributes.membership_values]
type = "TENSOR"
desc = "Members to test membership of for each set membership node. List all of the members to test again in the order that the 'BRANCH_MEMBER' mode appears in `node_modes`, delimited by `NaN`s. Will have the same number of sets of values as nodes with mode 'BRANCH_MEMBER'. This may be omitted if the node doesn't contain any 'BRANCH_MEMBER' nodes."

[attributes.n_targets]
type = "INT"
desc = "The total number of targets."

[attributes.nodes_falseleafs]
type = "INTS"
desc = "1 if false branch is leaf for each node and 0 if an interior node. To represent a tree that is a leaf (only has one node), one can do so by having a single `nodes_*` entry with true and false branches referencing the same `leaf_*` entry"
required = true

[attributes.nodes_falsenodeids]
type = "INTS"
desc = "If `nodes_falseleafs` is false at an entry, this represents the position of the false branch node. This position can be used to index into a `nodes_*` entry. If `nodes_falseleafs` is false, it is an index into the leaf_* attributes."
required = true

[attributes.nodes_featureids]
type = "INTS"
desc = "Feature id for each node."
required = true

[attributes.nodes_hitrates]
type = "TENSOR"
desc = "Popularity of each node, used for performance and may be omitted."

[attributes.nodes_missing_value_tracks_true]
type = "INTS"
desc = "For each node, define whether to follow the true branch (if attribute value is 1) or false branch (if attribute value is 0) in the presence of a NaN input feature. This attribute may be left undefined and the default value is false (0) for all nodes."

[attributes.nodes_modes]
type = "TENSOR"
desc = "The comparison operation performed by the node. This is encoded as an enumeration of 0 ('BRANCH_LEQ'), 1 ('BRANCH_LT'), 2 ('BRANCH_GTE'), 3 ('BRANCH_GT'), 4 ('BRANCH_EQ'), 5 ('BRANCH_NEQ'), and 6 ('BRANCH_MEMBER'). Note this is a tensor of type uint8."
required = true

[attributes.nodes_splits]
type = "TENSOR"
desc = "Thresholds to do the splitting on for each node with mode that is not 'BRANCH_MEMBER'."
required = true

[attributes.nodes_trueleafs]
type = "INTS"
desc = "1 if true branch is leaf for each node and 0 an interior node. To represent a tree that is a leaf (only has one node), one can do so by having a single `nodes_*` entry with true and false branches referencing the same `leaf_*` entry"
required = true

[attributes.nodes_truenodeids]
type = "INTS"
desc = "If `nodes_trueleafs` is false at an entry, this represents the position of the true branch node. This position can be used to index into a `nodes_*` entry. If `nodes_trueleafs` is false, it is an index into the leaf_* attributes."
required = true

[attributes.post_transform]
type = "INT"
desc = "Indicates the transform to apply to the score. <br>One of 'NONE' (0), 'SOFTMAX' (1), 'LOGISTIC' (2), 'SOFTMAX_ZERO' (3) or 'PROBIT' (4), defaults to 'NONE' (0)"
default = 0

[attributes.tree_roots]
type = "INTS"
desc = "Index into `nodes_*` for the root of each tree. The tree structure is derived from the branching of each node."
required = true

[[inputs]]
name = "X"
type = "T"
desc = "Input of shape [Batch Size, Number of Features]"

[[outputs]]
name = "Y"
type = "T"
desc = "Output of shape [Batch Size, Number of targets]"

[type_constraints.T]
types = ["tensor(float)", "tensor(double)", "tensor(float16)"]
desc = "The input type must be a tensor of a numeric type."
```

Tree Ensemble operator.  Returns the regressed values for each input in a batch.
Inputs have dimensions `[N, F]` where `N` is the input batch size and `F` is the number of input features.
Outputs have dimensions `[N, num_targets]` where `N` is the batch size and `num_targets` is the number of targets, which is a configurable attribute.

The encoding of this attribute is split along interior nodes and the leaves of the trees. Notably, attributes with the prefix `nodes_*` are associated with interior nodes, and attributes with the prefix `leaf_*` are associated with leaves.
The attributes `nodes_*` must all have the same length and encode a sequence of tuples, as defined by taking all the `nodes_*` fields at a given position.

All fields prefixed with `leaf_*` represent tree leaves, and similarly define tuples of leaves and must have identical length.

This operator can be used to implement both the previous `TreeEnsembleRegressor` and `TreeEnsembleClassifier` nodes.
The `TreeEnsembleRegressor` node maps directly to this node and requires changing how the nodes are represented.
The `TreeEnsembleClassifier` node can be implemented by adding a `ArgMax` node after this node to determine the top class.
To encode class labels, a `LabelEncoder` or `GatherND` operator may be used.
