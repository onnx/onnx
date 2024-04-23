/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/defs/schema.h"
#include "onnx/defs/traditionalml/utils.h"

#ifdef ONNX_ML
namespace ONNX_NAMESPACE {

static const char* TreeEnsemble_ver5_doc = R"DOC(
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
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    TreeEnsemble,
    5,
    OpSchema()
        .SetDoc(TreeEnsemble_ver5_doc)
        .Input(0, "X", "Input of shape [Batch Size, Number of Features]", "T")
        .Output(0, "Y", "Output of shape [Batch Size, Number of targets]", "T")
        .TypeConstraint(
            "T",
            {"tensor(float)", "tensor(double)", "tensor(float16)"},
            "The input type must be a tensor of a numeric type.")
        .Attr("nodes_featureids", "Feature id for each node.", AttributeProto::INTS, true)
        .Attr(
            "nodes_splits",
            "Thresholds to do the splitting on for each node with mode that is not 'BRANCH_MEMBER'.",
            AttributeProto::TENSOR,
            true)
        .Attr(
            "nodes_hitrates",
            "Popularity of each node, used for performance and may be omitted.",
            AttributeProto::TENSOR,
            OPTIONAL_VALUE)
        .Attr(
            "nodes_modes",
            "The comparison operation performed by the node. This is encoded as an enumeration of 0 ('BRANCH_LEQ'), 1 "
            "('BRANCH_LT'), 2 ('BRANCH_GTE'), 3 ('BRANCH_GT'), 4 ('BRANCH_EQ'), 5 ('BRANCH_NEQ'), and 6 "
            "('BRANCH_MEMBER'). Note this is a tensor of type uint8.",
            AttributeProto::TENSOR,
            true)
        .Attr(
            "nodes_truenodeids",
            "If `nodes_trueleafs` is false at an entry, this represents the position of the true branch node. This "
            "position can be used to index into a `nodes_*` entry. If `nodes_trueleafs` is false, it is an index into "
            "the leaf_* attributes.",
            AttributeProto::INTS,
            true)
        .Attr(
            "nodes_falsenodeids",
            "If `nodes_falseleafs` is false at an entry, this represents the position of the false branch node. This "
            "position can be used to index into a `nodes_*` entry. If `nodes_falseleafs` is false, it is an index into "
            "the leaf_* attributes.",
            AttributeProto::INTS,
            true)
        .Attr(
            "nodes_trueleafs",
            "1 if true branch is leaf for each node and 0 an interior node. To represent a tree that is a leaf (only "
            "has one node), one can do so by having a single `nodes_*` entry with true and false branches referencing "
            "the same `leaf_*` entry",
            AttributeProto::INTS,
            true)
        .Attr(
            "nodes_falseleafs",
            "1 if false branch is leaf for each node and 0 if an interior node. To represent a tree that is a leaf "
            "(only has one node), one can do so by having a single `nodes_*` entry with true and false branches "
            "referencing the same `leaf_*` entry",
            AttributeProto::INTS,
            true)
        .Attr(
            "nodes_missing_value_tracks_true",
            "For each node, define whether to follow the true branch (if attribute value is 1) or false branch (if "
            "attribute value is 0) in the presence of a NaN input feature. This attribute may be left undefined and "
            "the default value is false (0) for all nodes.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "tree_roots",
            "Index into `nodes_*` for the root of each tree. The tree structure is derived from the branching of each "
            "node.",
            AttributeProto::INTS,
            true)
        .Attr(
            "membership_values",
            "Members to test membership of for each set membership node. List all of the members to test again in the "
            "order that the 'BRANCH_MEMBER' mode appears in `node_modes`, delimited by `NaN`s. Will have the same "
            "number "
            "of sets of values as nodes with mode 'BRANCH_MEMBER'. This may be omitted if the node doesn't contain any "
            "'BRANCH_MEMBER' nodes.",
            AttributeProto::TENSOR,
            OPTIONAL_VALUE)
        .Attr(
            "leaf_targetids",
            "The index of the target that this leaf contributes to (this must be in range `[0, n_targets)`).",
            AttributeProto::INTS,
            true)
        .Attr("leaf_weights", "The weight for each leaf.", AttributeProto::TENSOR, true)
        .Attr("n_targets", "The total number of targets.", AttributeProto::INT, OPTIONAL_VALUE)
        .Attr(
            "post_transform",
            "Indicates the transform to apply to the score. <br>One of 'NONE' (0), 'SOFTMAX' (1), 'LOGISTIC' (2), "
            "'SOFTMAX_ZERO' (3) or 'PROBIT' (4), defaults to 'NONE' (0)",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "aggregate_function",
            "Defines how to aggregate leaf values within a target. <br>One of 'AVERAGE' (0) 'SUM' (1) 'MIN' (2) 'MAX "
            "(3) defaults to 'SUM' (1)",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          checkInputRank(ctx, 0, 2);
          auto* nodes_splits = ctx.getAttribute("nodes_splits");
          if (nullptr == nodes_splits) {
            fail_shape_inference("Attribute 'nodes_splits' is required.");
          }
          if (nodes_splits->t().dims_size() != 1) {
            fail_shape_inference("Attribute 'nodes_splits' must be 1D.");
          }
          auto input_type = ctx.getInputType(0)->tensor_type().elem_type();
          // Check that input type is same as split type
          if (input_type != nodes_splits->t().data_type()) {
            fail_shape_inference(
                "Attribute 'nodes_splits' must have same type as input. Input type is ",
                input_type,
                " and attribute type is ",
                nodes_splits->t().data_type());
          }

          // Expected nodes_* length
          auto expected_length = nodes_splits->t().dims(0);
          // Validate all nodes_* attributes that are set have the same length and are 1D.
          AssertAttributeProtoTypeAndLength(
              ctx.getAttribute("nodes_featureids"), expected_length, TensorProto_DataType_INT64, true);
          AssertAttributeProtoTypeAndLength(
              ctx.getAttribute("nodes_hitrates"), expected_length, TensorProto_DataType_FLOAT, false);
          AssertAttributeProtoTypeAndLength(
              ctx.getAttribute("nodes_modes"), expected_length, TensorProto_DataType_UINT8, true);
          AssertAttributeProtoTypeAndLength(
              ctx.getAttribute("nodes_truenodeids"), expected_length, TensorProto_DataType_INT64, true);
          AssertAttributeProtoTypeAndLength(
              ctx.getAttribute("nodes_falsenodeids"), expected_length, TensorProto_DataType_INT64, true);
          AssertAttributeProtoTypeAndLength(
              ctx.getAttribute("nodes_trueleafs"), expected_length, TensorProto_DataType_INT64, true);
          AssertAttributeProtoTypeAndLength(
              ctx.getAttribute("nodes_falseleafs"), expected_length, TensorProto_DataType_INT64, true);
          AssertAttributeProtoTypeAndLength(
              ctx.getAttribute("nodes_missing_value_tracks_true"), expected_length, TensorProto_DataType_INT64, false);

          // The set membership values and the splits must have the same type as the input.
          auto* membership_values = ctx.getAttribute("membership_values");
          if (nullptr != membership_values && membership_values->t().data_type() != input_type) {
            fail_shape_inference(
                "Attribute 'membership_values' must have same type as input. Input type is ",
                input_type,
                " and attribute type is ",
                membership_values->t().data_type());
          }
          AssertAttributeProtoTypeAndLength(
              ctx.getAttribute("nodes_splits"), expected_length, static_cast<TensorProto_DataType>(input_type), true);

          // Validate all leaf_* attributes that are set have the same length and are 1D.
          auto* leaf_targetids = ctx.getAttribute("leaf_targetids");
          auto* leaf_weights = ctx.getAttribute("leaf_weights");
          if (nullptr != leaf_targetids && nullptr != leaf_weights) {
            if (leaf_targetids->ints_size() != leaf_weights->t().dims(0)) {
              fail_shape_inference(
                  "Attribute 'leaf_targetids' must have same length as attribute 'leaf_weights'. 'leaf_targetids' "
                  "length is ",
                  leaf_targetids->ints_size(),
                  " and 'leaf_weights' length is ",
                  leaf_weights->t().dims(0));
            }
          } else {
            fail_shape_inference("Attributes 'leaf_targetids' and 'leaf_weights' must both be set.");
          }

          // Validate weights have same type as input.
          if (leaf_weights->t().data_type() != input_type) {
            fail_shape_inference(
                "Attribute 'leaf_weights' must have same type as input. Input type is ",
                input_type,
                " and attribute type is ",
                leaf_weights->t().data_type());
          }

          checkInputRank(ctx, 0, 2);

          Dim N, E;
          unifyInputDim(ctx, 0, 0, N);
          if (nullptr != ctx.getAttribute("n_targets")) {
            unifyDim(E, ctx.getAttribute("n_targets")->i());
          }
          updateOutputElemType(ctx, 0, input_type);
          updateOutputShape(ctx, 0, {N, E});
        }));

static const char* LabelEncoder_ver1_doc = R"DOC(
    Converts strings to integers and vice versa.<br>
    If the string default value is set, it will convert integers to strings.
    If the int default value is set, it will convert strings to integers.<br>
    Each operator converts either integers to strings or strings to integers, depending
    on which default value attribute is provided. Only one default value attribute
    should be defined.<br>
    When converting from integers to strings, the string is fetched from the
    'classes_strings' list, by simple indexing.<br>
    When converting from strings to integers, the string is looked up in the list
    and the index at which it is found is used as the converted value.
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    LabelEncoder,
    1,
    OpSchema()
        .SetDoc(LabelEncoder_ver1_doc)
        .Input(0, "X", "Input data.", "T1")
        .Output(0, "Y", "Output data. If strings are input, the output values are integers, and vice versa.", "T2")
        .TypeConstraint(
            "T1",
            {"tensor(string)", "tensor(int64)"},
            "The input type must be a tensor of integers or strings, of any shape.")
        .TypeConstraint(
            "T2",
            {"tensor(string)", "tensor(int64)"},
            "The output type will be a tensor of strings or integers, and will have the same shape as the input.")
        .Attr("classes_strings", "A list of labels.", AttributeProto::STRINGS, OPTIONAL_VALUE)
        .Attr(
            "default_int64",
            "An integer to use when an input string value is not found in the map.<br>One and only one of the "
            "'default_*' attributes must be defined.",
            AttributeProto::INT,
            static_cast<int64_t>(-1))
        .Attr(
            "default_string",
            "A string to use when an input integer value is not found in the map.<br>One and only one of the "
            "'default_*' attributes must be defined.",
            AttributeProto::STRING,
            std::string("_Unused"))
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto input_elem_type = ctx.getInputType(0)->tensor_type().elem_type();
          auto output_elem_type = ctx.getOutputType(0)->mutable_tensor_type();
          if (TensorProto::STRING == input_elem_type) {
            output_elem_type->set_elem_type(TensorProto::INT64);
          } else if (TensorProto::INT64 == input_elem_type) {
            output_elem_type->set_elem_type(TensorProto::STRING);
          }
        }));

static const char* TreeEnsembleClassifier_ver1_doc = R"DOC(
    Tree Ensemble classifier.  Returns the top class for each of N inputs.<br>
    The attributes named 'nodes_X' form a sequence of tuples, associated by
    index into the sequences, which must all be of equal length. These tuples
    define the nodes.<br>
    Similarly, all fields prefixed with 'class_' are tuples of votes at the leaves.
    A leaf may have multiple votes, where each vote is weighted by
    the associated class_weights index.<br>
    One and only one of classlabels_strings or classlabels_int64s
    will be defined. The class_ids are indices into this list.
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    TreeEnsembleClassifier,
    1,
    OpSchema()
        .SetDoc(TreeEnsembleClassifier_ver1_doc)
        .Input(0, "X", "Input of shape [N,F]", "T1")
        .Output(0, "Y", "N, Top class for each point", "T2")
        .Output(1, "Z", "The class score for each class, for each point, a tensor of shape [N,E].", "tensor(float)")
        .TypeConstraint(
            "T1",
            {"tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)"},
            "The input type must be a tensor of a numeric type.")
        .TypeConstraint(
            "T2",
            {"tensor(string)", "tensor(int64)"},
            "The output type will be a tensor of strings or integers, depending on which of the classlabels_* "
            "attributes is used.")
        .Attr("nodes_treeids", "Tree id for each node.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr(
            "nodes_nodeids",
            "Node id for each node. Ids may restart at zero for each tree, but it not required to.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr("nodes_featureids", "Feature id for each node.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr(
            "nodes_values",
            "Thresholds to do the splitting on for each node.",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE)
        .Attr(
            "nodes_hitrates",
            "Popularity of each node, used for performance and may be omitted.",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE)
        .Attr(
            "nodes_modes",
            "The node kind, that is, the comparison to make at the node. There is no comparison to make at a leaf "
            "node.<br>One of 'BRANCH_LEQ', 'BRANCH_LT', 'BRANCH_GTE', 'BRANCH_GT', 'BRANCH_EQ', 'BRANCH_NEQ', 'LEAF'",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr("nodes_truenodeids", "Child node if expression is true.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("nodes_falsenodeids", "Child node if expression is false.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr(
            "nodes_missing_value_tracks_true",
            "For each node, define what to do in the presence of a missing value: if a value is missing (NaN), use the "
            "'true' or 'false' branch based on the value in this array.<br>This attribute may be left undefined, and "
            "the default value is false (0) for all nodes.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr("class_treeids", "The id of the tree that this node is in.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("class_nodeids", "node id that this weight is for.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("class_ids", "The index of the class list that each weight is for.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("class_weights", "The weight for the class in class_id.", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr(
            "classlabels_strings",
            "Class labels if using string labels.<br>One and only one of the 'classlabels_*' attributes must be "
            "defined.",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr(
            "classlabels_int64s",
            "Class labels if using integer labels.<br>One and only one of the 'classlabels_*' attributes must be "
            "defined.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "post_transform",
            "Indicates the transform to apply to the score. <br> One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' "
            "or 'PROBIT.'",
            AttributeProto::STRING,
            std::string("NONE"))
        .Attr(
            "base_values",
            "Base values for classification, added to final class score; the size must be the same as the classes or "
            "can be left unassigned (assumed 0)",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          std::vector<std::string> label_strs;
          auto result = getRepeatedAttribute(ctx, "classlabels_strings", label_strs);
          bool using_strings = (result && !label_strs.empty());
          auto output_elem_type = ctx.getOutputType(0)->mutable_tensor_type();
          if (using_strings) {
            output_elem_type->set_elem_type(TensorProto::STRING);
          } else {
            output_elem_type->set_elem_type(TensorProto::INT64);
          }
        }));

static const char* TreeEnsembleClassifier_ver3_doc = R"DOC(
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
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    TreeEnsembleClassifier,
    3,
    OpSchema()
        .SetDoc(TreeEnsembleClassifier_ver3_doc)
        .Input(0, "X", "Input of shape [N,F]", "T1")
        .Output(0, "Y", "N, Top class for each point", "T2")
        .Output(1, "Z", "The class score for each class, for each point, a tensor of shape [N,E].", "tensor(float)")
        .TypeConstraint(
            "T1",
            {"tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)"},
            "The input type must be a tensor of a numeric type.")
        .TypeConstraint(
            "T2",
            {"tensor(string)", "tensor(int64)"},
            "The output type will be a tensor of strings or integers, depending on which of the classlabels_* "
            "attributes is used.")
        .Attr("nodes_treeids", "Tree id for each node.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr(
            "nodes_nodeids",
            "Node id for each node. Ids may restart at zero for each tree, but it not required to.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr("nodes_featureids", "Feature id for each node.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr(
            "nodes_values",
            "Thresholds to do the splitting on for each node.",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE)
        .Attr(
            "nodes_values_as_tensor",
            "Thresholds to do the splitting on for each node.",
            AttributeProto::TENSOR,
            OPTIONAL_VALUE)
        .Attr(
            "nodes_hitrates",
            "Popularity of each node, used for performance and may be omitted.",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE)
        .Attr(
            "nodes_hitrates_as_tensor",
            "Popularity of each node, used for performance and may be omitted.",
            AttributeProto::TENSOR,
            OPTIONAL_VALUE)
        .Attr(
            "nodes_modes",
            "The node kind, that is, the comparison to make at the node. There is no comparison to make at a leaf "
            "node.<br>One of 'BRANCH_LEQ', 'BRANCH_LT', 'BRANCH_GTE', 'BRANCH_GT', 'BRANCH_EQ', 'BRANCH_NEQ', 'LEAF'",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr("nodes_truenodeids", "Child node if expression is true.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("nodes_falsenodeids", "Child node if expression is false.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr(
            "nodes_missing_value_tracks_true",
            "For each node, define what to do in the presence of a missing value: if a value is missing (NaN), use the "
            "'true' or 'false' branch based on the value in this array.<br>This attribute may be left undefined, and "
            "the default value is false (0) for all nodes.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr("class_treeids", "The id of the tree that this node is in.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("class_nodeids", "node id that this weight is for.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("class_ids", "The index of the class list that each weight is for.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("class_weights", "The weight for the class in class_id.", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr(
            "class_weights_as_tensor",
            "The weight for the class in class_id.",
            AttributeProto::TENSOR,
            OPTIONAL_VALUE)
        .Attr(
            "classlabels_strings",
            "Class labels if using string labels.<br>One and only one of the 'classlabels_*' attributes must be "
            "defined.",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr(
            "classlabels_int64s",
            "Class labels if using integer labels.<br>One and only one of the 'classlabels_*' attributes must be "
            "defined.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "post_transform",
            "Indicates the transform to apply to the score. <br> One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' "
            "or 'PROBIT.'",
            AttributeProto::STRING,
            std::string("NONE"))
        .Attr(
            "base_values",
            "Base values for classification, added to final class score; the size must be the same as the classes or "
            "can be left unassigned (assumed 0)",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE)
        .Attr(
            "base_values_as_tensor",
            "Base values for classification, added to final class score; the size must be the same as the classes or "
            "can be left unassigned (assumed 0)",
            AttributeProto::TENSOR,
            OPTIONAL_VALUE)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto* nodes_values = ctx.getAttribute("nodes_values");
          auto* nodes_values_as_tensor = ctx.getAttribute("nodes_values_as_tensor");
          auto* nodes_hitrates = ctx.getAttribute("nodes_hitrates");
          auto* nodes_hitrates_as_tensor = ctx.getAttribute("nodes_hitrates_as_tensor");
          auto* class_weights = ctx.getAttribute("class_weights");
          auto* class_weights_as_tensor = ctx.getAttribute("class_weights_as_tensor");
          auto* base_values = ctx.getAttribute("base_values");
          auto* base_values_as_tensor = ctx.getAttribute("base_values_as_tensor");

          if (nullptr != nodes_values && nullptr != nodes_values_as_tensor) {
            fail_shape_inference(
                "Only one of the attributes 'nodes_values', 'nodes_values_as_tensor' should be specified.");
          }
          if (nullptr != nodes_hitrates && nullptr != nodes_hitrates_as_tensor) {
            fail_shape_inference(
                "Only one of the attributes 'nodes_hitrates', 'nodes_hitrates_as_tensor' should be specified.");
          }
          if (nullptr != class_weights && nullptr != class_weights_as_tensor) {
            fail_shape_inference(
                "Only one of the attributes 'class_weights', 'class_weights_as_tensor' should be specified.");
          }
          if (nullptr != base_values && nullptr != base_values_as_tensor) {
            fail_shape_inference(
                "Only one of the attributes 'base_values', 'base_values_as_tensor' should be specified.");
          }

          std::vector<std::string> classlabels_strings;
          auto result = getRepeatedAttribute(ctx, "classlabels_strings", classlabels_strings);
          bool using_strings = (result && !classlabels_strings.empty());
          if (using_strings) {
            updateOutputElemType(ctx, 0, TensorProto::STRING);
          } else {
            updateOutputElemType(ctx, 0, TensorProto::INT64);
          }
          updateOutputElemType(ctx, 1, TensorProto::FLOAT);

          checkInputRank(ctx, 0, 2);
          Dim N, E;
          unifyInputDim(ctx, 0, 0, N);

          if (using_strings) {
            unifyDim(E, classlabels_strings.size());
          } else {
            std::vector<int64_t> classlabels_int64s;
            result = getRepeatedAttribute(ctx, "classlabels_int64s", classlabels_int64s);
            if (!result || classlabels_int64s.empty()) {
              fail_shape_inference("Non of classlabels_int64s or classlabels_strings is set.");
            }
            unifyDim(E, classlabels_int64s.size());
          }
          updateOutputShape(ctx, 0, {N});
          updateOutputShape(ctx, 1, {N, E});
        }));

static const char* TreeEnsembleRegressor_ver1_doc = R"DOC(
    Tree Ensemble regressor.  Returns the regressed values for each input in N.<br>
    All args with nodes_ are fields of a tuple of tree nodes, and
    it is assumed they are the same length, and an index i will decode the
    tuple across these inputs.  Each node id can appear only once
    for each tree id.<br>
    All fields prefixed with target_ are tuples of votes at the leaves.<br>
    A leaf may have multiple votes, where each vote is weighted by
    the associated target_weights index.<br>
    All trees must have their node ids start at 0 and increment by 1.<br>
    Mode enum is BRANCH_LEQ, BRANCH_LT, BRANCH_GTE, BRANCH_GT, BRANCH_EQ, BRANCH_NEQ, LEAF
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    TreeEnsembleRegressor,
    1,
    OpSchema()
        .SetDoc(TreeEnsembleRegressor_ver1_doc)
        .Input(0, "X", "Input of shape [N,F]", "T")
        .Output(0, "Y", "N classes", "tensor(float)")
        .TypeConstraint(
            "T",
            {"tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)"},
            "The input type must be a tensor of a numeric type.")
        .Attr("nodes_treeids", "Tree id for each node.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr(
            "nodes_nodeids",
            "Node id for each node. Node ids must restart at zero for each tree and increase sequentially.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr("nodes_featureids", "Feature id for each node.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr(
            "nodes_values",
            "Thresholds to do the splitting on for each node.",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE)
        .Attr(
            "nodes_hitrates",
            "Popularity of each node, used for performance and may be omitted.",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE)
        .Attr(
            "nodes_modes",
            "The node kind, that is, the comparison to make at the node. There is no comparison to make at a leaf "
            "node.<br>One of 'BRANCH_LEQ', 'BRANCH_LT', 'BRANCH_GTE', 'BRANCH_GT', 'BRANCH_EQ', 'BRANCH_NEQ', 'LEAF'",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr("nodes_truenodeids", "Child node if expression is true", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("nodes_falsenodeids", "Child node if expression is false", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr(
            "nodes_missing_value_tracks_true",
            "For each node, define what to do in the presence of a NaN: use the 'true' (if the attribute value is 1) "
            "or 'false' (if the attribute value is 0) branch based on the value in this array.<br>This attribute may "
            "be left undefined and the default value is false (0) for all nodes.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr("target_treeids", "The id of the tree that each node is in.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("target_nodeids", "The node id of each weight", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("target_ids", "The index of the target that each weight is for", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("target_weights", "The weight for each target", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr("n_targets", "The total number of targets.", AttributeProto::INT, OPTIONAL_VALUE)
        .Attr(
            "post_transform",
            "Indicates the transform to apply to the score. <br>One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' "
            "or 'PROBIT'",
            AttributeProto::STRING,
            std::string("NONE"))
        .Attr(
            "aggregate_function",
            "Defines how to aggregate leaf values within a target. <br>One of 'AVERAGE,' 'SUM,' 'MIN,' 'MAX.'",
            AttributeProto::STRING,
            std::string("SUM"))
        .Attr(
            "base_values",
            "Base values for classification, added to final class score; the size must be the same as the classes or "
            "can be left unassigned (assumed 0)",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE));

static const char* TreeEnsembleRegressor_ver3_doc = R"DOC(
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
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    TreeEnsembleRegressor,
    3,
    OpSchema()
        .SetDoc(TreeEnsembleRegressor_ver3_doc)
        .Input(0, "X", "Input of shape [N,F]", "T")
        .Output(0, "Y", "N classes", "tensor(float)")
        .TypeConstraint(
            "T",
            {"tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)"},
            "The input type must be a tensor of a numeric type.")
        .Attr("nodes_treeids", "Tree id for each node.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr(
            "nodes_nodeids",
            "Node id for each node. Node ids must restart at zero for each tree and increase sequentially.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr("nodes_featureids", "Feature id for each node.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr(
            "nodes_values",
            "Thresholds to do the splitting on for each node.",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE)
        .Attr(
            "nodes_values_as_tensor",
            "Thresholds to do the splitting on for each node.",
            AttributeProto::TENSOR,
            OPTIONAL_VALUE)
        .Attr(
            "nodes_hitrates",
            "Popularity of each node, used for performance and may be omitted.",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE)
        .Attr(
            "nodes_hitrates_as_tensor",
            "Popularity of each node, used for performance and may be omitted.",
            AttributeProto::TENSOR,
            OPTIONAL_VALUE)
        .Attr(
            "nodes_modes",
            "The node kind, that is, the comparison to make at the node. There is no comparison to make at a leaf "
            "node.<br>One of 'BRANCH_LEQ', 'BRANCH_LT', 'BRANCH_GTE', 'BRANCH_GT', 'BRANCH_EQ', 'BRANCH_NEQ', 'LEAF'",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr("nodes_truenodeids", "Child node if expression is true", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("nodes_falsenodeids", "Child node if expression is false", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr(
            "nodes_missing_value_tracks_true",
            "For each node, define what to do in the presence of a NaN: use the 'true' (if the attribute value is 1) "
            "or 'false' (if the attribute value is 0) branch based on the value in this array.<br>This attribute may "
            "be left undefined and the default value is false (0) for all nodes.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr("target_treeids", "The id of the tree that each node is in.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("target_nodeids", "The node id of each weight", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("target_ids", "The index of the target that each weight is for", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("target_weights", "The weight for each target", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr("target_weights_as_tensor", "The weight for each target", AttributeProto::TENSOR, OPTIONAL_VALUE)
        .Attr("n_targets", "The total number of targets.", AttributeProto::INT, OPTIONAL_VALUE)
        .Attr(
            "post_transform",
            "Indicates the transform to apply to the score. <br>One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' "
            "or 'PROBIT'",
            AttributeProto::STRING,
            std::string("NONE"))
        .Attr(
            "aggregate_function",
            "Defines how to aggregate leaf values within a target. <br>One of 'AVERAGE,' 'SUM,' 'MIN,' 'MAX.'",
            AttributeProto::STRING,
            std::string("SUM"))
        .Attr(
            "base_values",
            "Base values for regression, added to final prediction after applying aggregate_function; the size must be "
            "the same as the classes or can be left unassigned (assumed 0)",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE)
        .Attr(
            "base_values_as_tensor",
            "Base values for regression, added to final prediction after applying aggregate_function; the size must be "
            "the same as the classes or can be left unassigned (assumed 0)",
            AttributeProto::TENSOR,
            OPTIONAL_VALUE)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto* nodes_values = ctx.getAttribute("nodes_values");
          auto* nodes_values_as_tensor = ctx.getAttribute("nodes_values_as_tensor");
          auto* nodes_hitrates = ctx.getAttribute("nodes_hitrates");
          auto* nodes_hitrates_as_tensor = ctx.getAttribute("nodes_hitrates_as_tensor");
          auto* target_weights = ctx.getAttribute("target_weights");
          auto* target_weights_as_tensor = ctx.getAttribute("target_weights_as_tensor");
          auto* base_values = ctx.getAttribute("base_values");
          auto* base_values_as_tensor = ctx.getAttribute("base_values_as_tensor");

          if (nullptr != nodes_values && nullptr != nodes_values_as_tensor) {
            fail_shape_inference(
                "Only one of the attributes 'nodes_values', 'nodes_values_as_tensor' should be specified.");
          }
          if (nullptr != nodes_hitrates && nullptr != nodes_hitrates_as_tensor) {
            fail_shape_inference(
                "Only one of the attributes 'nodes_hitrates', 'nodes_hitrates_as_tensor' should be specified.");
          }
          if (nullptr != target_weights && nullptr != target_weights_as_tensor) {
            fail_shape_inference(
                "Only one of the attributes 'target_weights', 'target_weights_as_tensor' should be specified.");
          }
          if (nullptr != base_values && nullptr != base_values_as_tensor) {
            fail_shape_inference(
                "Only one of the attributes 'base_values', 'base_values_as_tensor' should be specified.");
          }

          checkInputRank(ctx, 0, 2);
          Dim N, E;
          unifyInputDim(ctx, 0, 0, N);
          if (nullptr != ctx.getAttribute("n_targets")) {
            unifyDim(E, ctx.getAttribute("n_targets")->i());
          }
          updateOutputElemType(ctx, 0, TensorProto::FLOAT);
          updateOutputShape(ctx, 0, {N, E});
        }));

static const char* LabelEncoder_ver2_doc = R"DOC(
    Maps each element in the input tensor to another value.<br>
    The mapping is determined by the two parallel attributes, 'keys_*' and
    'values_*' attribute. The i-th value in the specified 'keys_*' attribute
    would be mapped to the i-th value in the specified 'values_*' attribute. It
    implies that input's element type and the element type of the specified
    'keys_*' should be identical while the output type is identical to the
    specified 'values_*' attribute. If an input element can not be found in the
    specified 'keys_*' attribute, the 'default_*' that matches the specified
    'values_*' attribute may be used as its output value.<br>
    Let's consider an example which maps a string tensor to an integer tensor.
    Assume and 'keys_strings' is ["Amy", "Sally"], 'values_int64s' is [5, 6],
    and 'default_int64' is '-1'.  The input ["Dori", "Amy", "Amy", "Sally",
    "Sally"] would be mapped to [-1, 5, 5, 6, 6].<br>
    Since this operator is an one-to-one mapping, its input and output shapes
    are the same. Notice that only one of 'keys_*'/'values_*' can be set.<br>
    For key look-up, bit-wise comparison is used so even a float NaN can be
    mapped to a value in 'values_*' attribute.<br>
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    LabelEncoder,
    2,
    OpSchema()
        .SetDoc(LabelEncoder_ver2_doc)
        .Input(0, "X", "Input data. It can be either tensor or scalar.", "T1")
        .Output(0, "Y", "Output data.", "T2")
        .TypeConstraint(
            "T1",
            {"tensor(string)", "tensor(int64)", "tensor(float)"},
            "The input type is a tensor of any shape.")
        .TypeConstraint(
            "T2",
            {"tensor(string)", "tensor(int64)", "tensor(float)"},
            "Output type is determined by the specified 'values_*' attribute.")
        .Attr(
            "keys_strings",
            "A list of strings. One and only one of 'keys_*'s should be set.",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr("keys_int64s", "A list of ints.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("keys_floats", "A list of floats.", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr(
            "values_strings",
            "A list of strings. One and only one of 'value_*'s should be set.",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr("values_int64s", "A list of ints.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("values_floats", "A list of floats.", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr("default_string", "A string.", AttributeProto::STRING, std::string("_Unused"))
        .Attr("default_int64", "An integer.", AttributeProto::INT, static_cast<int64_t>(-1))
        .Attr("default_float", "A float.", AttributeProto::FLOAT, -0.f)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Label encoder is one-to-one mapping.
          if (ctx.getNumInputs() != 1) {
            fail_shape_inference("Label encoder has only one input.");
          }
          if (ctx.getNumOutputs() != 1) {
            fail_shape_inference("Label encoder has only one output.");
          }

          // Load all key_* attributes.
          std::vector<std::string> keys_strings;
          bool keys_strings_result = getRepeatedAttribute(ctx, "keys_strings", keys_strings);
          std::vector<int64_t> keys_int64s;
          bool keys_int64s_result = getRepeatedAttribute(ctx, "keys_int64s", keys_int64s);
          std::vector<float> keys_floats;
          bool keys_floats_result = getRepeatedAttribute(ctx, "keys_floats", keys_floats);

          // Check if only one keys_* attribute is set.
          if (static_cast<int>(keys_strings_result) + static_cast<int>(keys_int64s_result) +
                  static_cast<int>(keys_floats_result) !=
              1) {
            fail_shape_inference("Only one of keys_*'s can be set in label encoder.");
          }

          // Check if the specified keys_* matches input type.
          auto input_elem_type = ctx.getInputType(0)->tensor_type().elem_type();
          if (keys_strings_result && input_elem_type != TensorProto::STRING) {
            fail_shape_inference("Input type is not string tensor but key_strings is set");
          }
          if (keys_int64s_result && input_elem_type != TensorProto::INT64) {
            fail_shape_inference("Input type is not int64 tensor but keys_int64s is set");
          }
          if (keys_floats_result && input_elem_type != TensorProto::FLOAT) {
            fail_shape_inference("Input type is not float tensor but keys_floats is set");
          }

          // Load all values_* attributes.
          std::vector<std::string> values_strings;
          bool values_strings_result = getRepeatedAttribute(ctx, "values_strings", values_strings);
          std::vector<int64_t> values_int64s;
          bool values_int64s_result = getRepeatedAttribute(ctx, "values_int64s", values_int64s);
          std::vector<float> values_floats;
          bool values_floats_result = getRepeatedAttribute(ctx, "values_floats", values_floats);

          // Check if only one values_* attribute is set.
          if (static_cast<int>(values_strings_result) + static_cast<int>(values_int64s_result) +
                  static_cast<int>(values_floats_result) !=
              1) {
            fail_shape_inference("Only one of values_*'s can be set in label encoder.");
          }

          // Assign output type based on the specified values_*.
          auto output_elem_type = ctx.getOutputType(0)->mutable_tensor_type();
          if (values_strings_result)
            output_elem_type->set_elem_type(TensorProto::STRING);
          if (values_int64s_result)
            output_elem_type->set_elem_type(TensorProto::INT64);
          if (values_floats_result)
            output_elem_type->set_elem_type(TensorProto::FLOAT);

          // Input and output shapes are the same.
          propagateShapeFromInputToOutput(ctx, 0, 0);
        }));
} // namespace ONNX_NAMESPACE
#endif
