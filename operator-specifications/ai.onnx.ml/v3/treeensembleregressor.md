```toml
name = "TreeEnsembleRegressor"
domain = "ai.onnx.ml"

[attributes.aggregate_function]
type = "STRING"
desc = "Defines how to aggregate leaf values within a target. <br>One of 'AVERAGE,' 'SUM,' 'MIN,' 'MAX.'"
default = "SUM"

[attributes.base_values]
type = "FLOATS"
desc = "Base values for regression, added to final prediction after applying aggregate_function; the size must be the same as the classes or can be left unassigned (assumed 0)"

[attributes.base_values_as_tensor]
type = "TENSOR"
desc = "Base values for regression, added to final prediction after applying aggregate_function; the size must be the same as the classes or can be left unassigned (assumed 0)"

[attributes.n_targets]
type = "INT"
desc = "The total number of targets."

[attributes.nodes_falsenodeids]
type = "INTS"
desc = "Child node if expression is false"

[attributes.nodes_featureids]
type = "INTS"
desc = "Feature id for each node."

[attributes.nodes_hitrates]
type = "FLOATS"
desc = "Popularity of each node, used for performance and may be omitted."

[attributes.nodes_hitrates_as_tensor]
type = "TENSOR"
desc = "Popularity of each node, used for performance and may be omitted."

[attributes.nodes_missing_value_tracks_true]
type = "INTS"
desc = "For each node, define what to do in the presence of a NaN: use the 'true' (if the attribute value is 1) or 'false' (if the attribute value is 0) branch based on the value in this array.<br>This attribute may be left undefined and the default value is false (0) for all nodes."

[attributes.nodes_modes]
type = "STRINGS"
desc = "The node kind, that is, the comparison to make at the node. There is no comparison to make at a leaf node.<br>One of 'BRANCH_LEQ', 'BRANCH_LT', 'BRANCH_GTE', 'BRANCH_GT', 'BRANCH_EQ', 'BRANCH_NEQ', 'LEAF'"

[attributes.nodes_nodeids]
type = "INTS"
desc = "Node id for each node. Node ids must restart at zero for each tree and increase sequentially."

[attributes.nodes_treeids]
type = "INTS"
desc = "Tree id for each node."

[attributes.nodes_truenodeids]
type = "INTS"
desc = "Child node if expression is true"

[attributes.nodes_values]
type = "FLOATS"
desc = "Thresholds to do the splitting on for each node."

[attributes.nodes_values_as_tensor]
type = "TENSOR"
desc = "Thresholds to do the splitting on for each node."

[attributes.post_transform]
type = "STRING"
desc = "Indicates the transform to apply to the score. <br>One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT'"
default = "NONE"

[attributes.target_ids]
type = "INTS"
desc = "The index of the target that each weight is for"

[attributes.target_nodeids]
type = "INTS"
desc = "The node id of each weight"

[attributes.target_treeids]
type = "INTS"
desc = "The id of the tree that each node is in."

[attributes.target_weights]
type = "FLOATS"
desc = "The weight for each target"

[attributes.target_weights_as_tensor]
type = "TENSOR"
desc = "The weight for each target"
```

Tree Ensemble regressor.  Returns the regressed values for each input in N.<br>
All args with nodes_ are fields of a tuple of tree nodes, and
it is assumed they are the same length, and an index i will decode the
tuple across these inputs.  Each node id can appear only once
for each tree id.<br>
All fields prefixed with target_ are tuples of votes at the leaves.<br>
A leaf may have multiple votes, where each vote is weighted by
the associated target_weights index.<br>
All fields ending with <i>_as_tensor</i> can be used instead of the
same parameter without the suffix if the element type is double and not float.
All trees must have their node ids start at 0 and increment by 1.<br>
Mode enum is BRANCH_LEQ, BRANCH_LT, BRANCH_GTE, BRANCH_GT, BRANCH_EQ, BRANCH_NEQ, LEAF
