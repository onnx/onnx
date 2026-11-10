```toml
name = "TreeEnsembleRegressor"
domain = "ai.onnx.ml"
deprecated = true
```

This operator is DEPRECATED. Please use TreeEnsemble instead which provides the same
functionality.<br>
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
