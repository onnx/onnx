// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"

#ifdef ONNX_ML
namespace ONNX_NAMESPACE {
static const char* ArrayFeatureExtractor_ver1_doc = R"DOC(
    Select a subset of the data X based on the indices provided Y.
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    ArrayFeatureExtractor,
    1,
    OpSchema()
        .SetDoc(ArrayFeatureExtractor_ver1_doc)
        .Input(0, "X", "Data to be selected", "T")
        .Input(
            1,
            "Y",
            "The index values to select as a int64 tensor",
            "tensor(int64)")
        .Output(0, "Z", "Selected output data as an array", "T")
        .TypeConstraint(
            "T",
            {"tensor(float)",
             "tensor(double)",
             "tensor(int64)",
             "tensor(int32)",
             "tensor(string)"},
            "allowed types."));

static const char* Binarizer_ver1_doc = R"DOC(
    Makes values 1 or 0 based on a single threshold.
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    Binarizer,
    1,
    OpSchema()
        .SetDoc(Binarizer_ver1_doc)
        .Input(0, "X", "Data to be binarized", "T")
        .Output(0, "Y", "Binarized output data", "T")
        .TypeConstraint(
            "T",
            {"tensor(float)",
             "tensor(double)",
             "tensor(int64)",
             "tensor(int32)"},
            "allowed types.")
        .Attr(
            "threshold",
            "Values greater than this are set to 1, else set to 0",
            AttributeProto::FLOAT,
            0.f));

static const char* CastMap_ver1_doc = R"DOC(
    Converts a map to a tensor.  Map key must be int64 and the values will be ordered
    in ascending order based on this key.  Supports dense packing or sparse packing.
    If using sparse packing, the key cannot exceed the max_map-1 value.
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    CastMap,
    1,
    OpSchema()
        .SetDoc(CastMap_ver1_doc)
        .Input(0, "X", "Data to be encoded", "T1")
        .Output(0, "Y", "encoded output data", "T2")
        .TypeConstraint(
            "T1",
            {"map(int64, string)", "map(int64, float)"},
            " allowed input types.")
        .TypeConstraint(
            "T2",
            {"tensor(string)", "tensor(float)", "tensor(int64)"},
            " allowed output types.")
        .Attr(
            "cast_to",
            "what type of tensor to cast the input to, enum 'TO_FLOAT','TO_STRING','TO_INT64', default is 'TO_FLOAT'",
            AttributeProto::STRING,
            std::string("TO_FLOAT"))
        .Attr(
            "map_form",
            "whether to only output as many values as are in the input, or position the input based on using the key of the map as the index of the output (sparse), enum 'DENSE', 'SPARSE', default is 'DENSE'",
            AttributeProto::STRING,
            std::string("DENSE"))
        .Attr(
            "max_map",
            "if map_form packing is SPARSE, what is the total length of each output in N (max index value)",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto cast_to_attr = ctx.getAttribute("cast_to");
          auto output_type = ctx.getOutputType(0)->mutable_tensor_type();
          if (nullptr == cast_to_attr) {
            output_type->set_elem_type(TensorProto::FLOAT);
            return;
          }
          auto& cast_to = cast_to_attr->s();
          if (0 == cast_to.compare("TO_FLOAT")) {
            output_type->set_elem_type(TensorProto::FLOAT);
          } else if (0 == cast_to.compare("TO_INT64")) {
            output_type->set_elem_type(TensorProto::INT64);
          } else if (0 == cast_to.compare("TO_STRING")) {
            output_type->set_elem_type(TensorProto::STRING);
          }
        }));

static const char* CategoryMapper_ver1_doc = R"DOC(
    Convert strings to int64s and vice versa.
    Takes in a map to use for the conversion.
    The index position in the strings and ints repeated inputs
     is used to do the mapping.
    Each instantiated operator converts either ints to strings or strings to ints.
    This behavior is triggered based on which default value is set.
    If the string default value is set, it will convert ints to strings.
    If the int default value is set, it will convert strings to ints.
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    CategoryMapper,
    1,
    OpSchema()
        .SetDoc(CategoryMapper_ver1_doc)
        .Input(0, "X", "Input data", "T1")
        .Output(
            0,
            "Y",
            "Output data, if strings are input, then output is int64s, and vice versa.",
            "T2")
        .TypeConstraint(
            "T1",
            {"tensor(string)", "tensor(int64)"},
            " allowed types.")
        .TypeConstraint(
            "T2",
            {"tensor(string)", "tensor(int64)"},
            " allowed types.")
        .Attr(
            "cats_strings",
            "strings part of the input map, must be same size and the ints",
            AttributeProto::STRINGS,
            OPTIONAL)
        .Attr(
            "cats_int64s",
            "ints part of the input map, must be same size and the strings",
            AttributeProto::INTS,
            OPTIONAL)
        .Attr(
            "default_string",
            "string value to use if the int is not in the map",
            AttributeProto::STRING,
            std::string("_Unused"))
        .Attr(
            "default_int64",
            "int value to use if the string is not in the map",
            AttributeProto::INT,
            static_cast<int64_t>(-1))
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto input_elem_type = ctx.getInputType(0)->tensor_type().elem_type();
          auto output_elem_type = ctx.getOutputType(0)->mutable_tensor_type();
          if (TensorProto::STRING == input_elem_type) {
            output_elem_type->set_elem_type(TensorProto::INT64);
          } else if (TensorProto::INT64 == input_elem_type) {
            output_elem_type->set_elem_type(TensorProto::STRING);
          }
        }));

static const char* DictVectorizer_ver1_doc = R"DOC(
    Uses an index mapping to convert a dictionary to an array.
    The output array will be equal in length to the index mapping vector parameter.
    All keys in the input dictionary must be present in the index mapping vector.
    For each item in the input dictionary, insert its value in the output array.
    The position of the insertion is determined by the position of the item's key
    in the index mapping. Any keys not present in the input dictionary, will be
    zero in the output array.  Use either string_vocabulary or int64_vocabulary, not both.
    For example: if the ``string_vocabulary`` parameter is set to ``["a", "c", "b", "z"]``,
    then an input of ``{"a": 4, "c": 8}`` will produce an output of ``[4, 8, 0, 0]``.
    )DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    DictVectorizer,
    1,
    OpSchema()
        .SetDoc(DictVectorizer_ver1_doc)
        .Input(0, "X", "The input dictionary", "T1")
        .Output(0, "Y", "The tensor", "T2")
        .TypeConstraint(
            "T1",
            {"map(string, int64)",
             "map(int64, string)",
             "map(int64, float)",
             "map(int64, double)",
             "map(string, float)",
             "map(string, double)"},
            " allowed types.")
        .TypeConstraint(
            "T2",
            {"tensor(int64)",
             "tensor(float)",
             "tensor(double)",
             "tensor(string)"},
            " allowed types.")
        .Attr(
            "string_vocabulary",
            "The vocabulary vector",
            AttributeProto::STRINGS,
            OPTIONAL)
        .Attr(
            "int64_vocabulary",
            "The vocabulary vector",
            AttributeProto::INTS,
            OPTIONAL)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto input_elem_type = ctx.getInputType(0)
                                     ->map_type()
                                     .value_type()
                                     .tensor_type()
                                     .elem_type();
          auto output_elem_type = ctx.getOutputType(0)->mutable_tensor_type();
          output_elem_type->set_elem_type(input_elem_type);
        }));

static const char* Imputer_ver1_doc = R"DOC(
    Replace imputs that equal replaceValue/s  with  imputeValue/s.
    All other inputs are copied to the output unchanged.
    This op is used to replace missing values where we know what a missing value looks like.
    Only one of imputed_value_floats or imputed_value_int64s should be used.
    The size can be 1 element, which will be reused, or the size of the feature set F in input N,F
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    Imputer,
    1,
    OpSchema()
        .SetDoc(Imputer_ver1_doc)
        .Input(0, "X", "Data to be imputed", "T")
        .Output(0, "Y", "Imputed output data", "T")
        .TypeConstraint(
            "T",
            {"tensor(float)",
             "tensor(double)",
             "tensor(int64)",
             "tensor(int32)"},
            " allowed types.")
        .Attr(
            "imputed_value_floats",
            "value to change to",
            AttributeProto::FLOATS,
            OPTIONAL)
        .Attr(
            "replaced_value_float",
            "value that needs replacing",
            AttributeProto::FLOAT,
            0.f)
        .Attr(
            "imputed_value_int64s",
            "value to change to",
            AttributeProto::INTS,
            OPTIONAL)
        .Attr(
            "replaced_value_int64",
            "value that needs replacing",
            AttributeProto::INT,
            static_cast<int64_t>(0)));

static const char* LabelEncoder_ver1_doc = R"DOC(
    Convert class label to their integral type and vice versa.
    In both cases the operator is instantiated with the list of class strings.
    The integral value of the string is the index position in the list.
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    LabelEncoder,
    1,
    OpSchema()
        .SetDoc(LabelEncoder_ver1_doc)
        .Input(0, "X", "Data to be encoded", "T1")
        .Output(0, "Y", "Encoded output data", "T2")
        .TypeConstraint(
            "T1",
            {"tensor(string)", "tensor(int64)"},
            " allowed types.")
        .TypeConstraint(
            "T2",
            {"tensor(string)", "tensor(int64)"},
            " allowed types.")
        .Attr(
            "classes_strings",
            "List of class label strings to be encoded as int64s",
            AttributeProto::STRINGS,
            OPTIONAL)
        .Attr(
            "default_int64",
            "Default value if not in class list as int64",
            AttributeProto::INT,
            static_cast<int64_t>(-1))
        .Attr(
            "default_string",
            "Default value if not in class list as string",
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

static const char* LinearClassifier_ver1_doc = R"DOC(
    Linear classifier prediction (choose class)
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    LinearClassifier,
    1,
    OpSchema()
        .SetDoc(LinearClassifier_ver1_doc)
        .Input(0, "X", "Data to be classified", "T1")
        .Output(0, "Y", "Classification outputs (one class per example", "T2")
        .Output(
            1,
            "Z",
            "Classification scores (N,E - one score for each class, for each example",
            "tensor(float)")
        .TypeConstraint(
            "T1",
            {"tensor(float)",
             "tensor(double)",
             "tensor(int64)",
             "tensor(int32)"},
            " allowed types.")
        .TypeConstraint(
            "T2",
            {"tensor(string)", "tensor(int64)"},
            " allowed types.")
        .Attr("coefficients", "weights of the model(s)", AttributeProto::FLOATS)
        .Attr(
            "intercepts",
            "weights of the intercepts (if used)",
            AttributeProto::FLOATS,
            OPTIONAL)
        .Attr(
            "multi_class",
            "whether to do OvR or multinomial (0=OvR and is default)",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "classlabels_strings",
            "class labels if using string labels",
            AttributeProto::STRINGS,
            OPTIONAL)
        .Attr(
            "classlabels_ints",
            "class labels if using int labels",
            AttributeProto::INTS,
            OPTIONAL)
        .Attr(
            "post_transform",
            "enum NONE, SOFTMAX, LOGISTIC, SOFTMAX_ZERO, PROBIT",
            AttributeProto::STRING,
            std::string("NONE"))
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          std::vector<std::string> label_strs;
          auto result =
              getRepeatedAttribute(ctx, "classlabels_strings", label_strs);
          bool using_strings = (result && !label_strs.empty());
          auto output_elem_type = ctx.getOutputType(0)->mutable_tensor_type();
          if (using_strings) {
            output_elem_type->set_elem_type(TensorProto::STRING);
          } else {
            output_elem_type->set_elem_type(TensorProto::INT64);
          }
        }));

static const char* LinearRegressor_ver1_doc = R"DOC(
    Generalized linear regression evaluation.
    If targets is set to 1 (default) then univariate regression is performed.
    If targets is set to M then M sets of coefficients must be passed in as a sequence
    and M results will be output for each input n in N.
    Coefficients are of the same length as an n, and coefficients for each target are contiguous.
    Intercepts are optional but if provided must match the number of targets.
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    LinearRegressor,
    1,
    OpSchema()
        .SetDoc(LinearRegressor_ver1_doc)
        .Input(0, "X", "Data to be regressed", "T")
        .Output(
            0,
            "Y",
            "Regression outputs (one per target, per example",
            "tensor(float)")
        .TypeConstraint(
            "T",
            {"tensor(float)",
             "tensor(double)",
             "tensor(int64)",
             "tensor(int32)"},
            " allowed types.")
        .Attr(
            "post_transform",
            "enum NONE, SOFTMAX, LOGISTIC, SOFTMAX_ZERO, PROBIT",
            AttributeProto::STRING,
            std::string("NONE"))
        .Attr(
            "coefficients",
            "weights of the model(s)",
            AttributeProto::FLOATS,
            OPTIONAL)
        .Attr(
            "intercepts",
            "weights of the intercepts (if used)",
            AttributeProto::FLOATS,
            OPTIONAL)
        .Attr(
            "targets",
            "total number of regression targets (default is 1)",
            AttributeProto::INT,
            static_cast<int64_t>(1)));

static const char* Normalizer_ver1_doc = R"DOC(
    Normalize the input.  There are three normalization modes,
    which have the corresponding formulas:
    Max .. math::     max(x_i)
    L1  .. math::  z = ||x||_1 = \sum_{i=1}^{n} |x_i|
    L2  .. math::  z = ||x||_2 = \sqrt{\sum_{i=1}^{n} x_i^2}
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    Normalizer,
    1,
    OpSchema()
        .SetDoc(Normalizer_ver1_doc)
        .Input(0, "X", "Data to be encoded", "T")
        .Output(0, "Y", "encoded output data", "tensor(float)")
        .TypeConstraint(
            "T",
            {"tensor(float)",
             "tensor(double)",
             "tensor(int64)",
             "tensor(int32)"},
            " allowed types.")
        .Attr(
            "norm",
            "enum 'MAX', 'L1', 'L2'",
            AttributeProto::STRING,
            std::string("MAX")));

static const char* OneHotEncoder_ver1_doc = R"DOC(
    Replace the inputs with an array of ones and zeros, where the only
    one is the zero-based category that was passed in.  The total category count
    will determine the length of the vector. For example if we pass a
    tensor with a single value of 4, and a category count of 8, the
    output will be a tensor with 0,0,0,0,1,0,0,0 .

    This operator assumes every input in X is of the same category set
    (meaning there is only one category count).

    If the input is a tensor of float, int32, or double, the data will be cast
    to int64s and the cats_int64s category list will be used for the lookups.
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    OneHotEncoder,
    1,
    OpSchema()
        .SetDoc(OneHotEncoder_ver1_doc)
        .Input(0, "X", "Data to be encoded", "T")
        .Output(0, "Y", "encoded output data", "tensor(float)")
        .TypeConstraint(
            "T",
            {"tensor(string)",
             "tensor(int64)",
             "tensor(int32)",
             "tensor(float)",
             "tensor(double)"},
            " allowed types.")
        .Attr(
            "cats_int64s",
            "list of categories, ints",
            AttributeProto::INTS,
            OPTIONAL)
        .Attr(
            "cats_strings",
            "list of categories, strings",
            AttributeProto::STRINGS,
            OPTIONAL)
        .Attr(
            "zeros",
            "if true and category is not present, will return all zeros, if false and missing category, operator will return false. Default is true (1).",
            AttributeProto::INT,
            static_cast<int64_t>(1)));

static const char* Scaler_ver1_doc = R"DOC(
    Rescale input data, for example to standardize features by removing the mean and scaling to unit variance.
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    Scaler,
    1,
    OpSchema()
        .SetDoc(Scaler_ver1_doc)
        .Input(0, "X", "Data to be scaled", "T")
        .Output(0, "Y", "Scaled output data", "tensor(float)")
        .TypeConstraint(
            "T",
            {"tensor(float)",
             "tensor(double)",
             "tensor(int64)",
             "tensor(int32)"},
            " allowed types.")
        .Attr(
            "scale",
            "second, multiply by this, can be length of features or length 1",
            AttributeProto::FLOATS,
            OPTIONAL)
        .Attr(
            "offset",
            "first, offset by this, must be same length as scale",
            AttributeProto::FLOATS,
            OPTIONAL));

static const char* SVMClassifier_ver1_doc = R"DOC(
    SVM classifier prediction
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    SVMClassifier,
    1,
    OpSchema()
        .SetDoc(SVMClassifier_ver1_doc)
        .Input(0, "X", "Data to be classified", "T1")
        .Output(0, "Y", "Classification outputs (one class per example)", "T2")
        .Output(
            1,
            "Z",
            "Class scores (one per class per example), if prob_a and prob_b are provided they are probabilities for each class otherwise they are raw scores.",
            "tensor(float)")
        .TypeConstraint(
            "T1",
            {"tensor(float)",
             "tensor(double)",
             "tensor(int64)",
             "tensor(int32)"},
            " allowed types.")
        .TypeConstraint(
            "T2",
            {"tensor(string)", "tensor(int64)"},
            " allowed types.")
        .Attr(
            "kernel_type",
            "enum LINEAR, POLY, RBF, SIGMOID, defaults to linear",
            AttributeProto::STRING,
            std::string("LINEAR"))
        .Attr(
            "kernel_params",
            "Tensor of 3 elements containing gamma, coef0, degree in that order.  Zero if unused for the kernel.",
            AttributeProto::FLOATS,
            OPTIONAL)
        .Attr("vectors_per_class", "", AttributeProto::INTS, OPTIONAL)
        .Attr("support_vectors", "", AttributeProto::FLOATS, OPTIONAL)
        .Attr("coefficients", "", AttributeProto::FLOATS, OPTIONAL)
        .Attr(
            "prob_a",
            "First set of probability coefficients",
            AttributeProto::FLOATS,
            OPTIONAL)
        .Attr(
            "prob_b",
            "Second set of probability coefficients, must be same size as prob_a, if these are provided then output Z are probability estimates.",
            AttributeProto::FLOATS,
            OPTIONAL)
        .Attr("rho", "", AttributeProto::FLOATS, OPTIONAL)
        .Attr(
            "post_transform",
            "post eval transform for score, enum NONE, SOFTMAX, LOGISTIC, SOFTMAX_ZERO, PROBIT",
            AttributeProto::STRING,
            std::string("NONE"))
        .Attr(
            "classlabels_strings",
            "class labels if using string labels",
            AttributeProto::STRINGS,
            OPTIONAL)
        .Attr(
            "classlabels_ints",
            "class labels if using int labels",
            AttributeProto::INTS,
            OPTIONAL)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          std::vector<std::string> label_strs;
          auto result =
              getRepeatedAttribute(ctx, "classlabels_strings", label_strs);
          bool using_strings = (result && !label_strs.empty());
          auto output_elem_type = ctx.getOutputType(0)->mutable_tensor_type();
          if (using_strings) {
            output_elem_type->set_elem_type(TensorProto::STRING);
          } else {
            output_elem_type->set_elem_type(TensorProto::INT64);
          }
        }));

static const char* SVMRegressor_ver1_doc = R"DOC(
    SVM regression prediction and one-class svm anomaly detection
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    SVMRegressor,
    1,
    OpSchema()
        .SetDoc(SVMRegressor_ver1_doc)
        .Input(0, "X", "Data to be regressed", "T")
        .Output(
            0,
            "Y",
            "Regression outputs (one score per target per example)",
            "tensor(float)")
        .TypeConstraint(
            "T",
            {"tensor(float)",
             "tensor(double)",
             "tensor(int64)",
             "tensor(int32)"},
            " allowed types.")
        .Attr(
            "kernel_type",
            "enum LINEAR, POLY, RBF, SIGMOID, defaults to linear",
            AttributeProto::STRING,
            std::string("LINEAR"))
        .Attr(
            "kernel_params",
            "Tensor of 3 elements containing gamma, coef0, degree in that order.  Zero if unused for the kernel.",
            AttributeProto::FLOATS,
            OPTIONAL)
        .Attr(
            "support_vectors",
            "chosen support vectors",
            AttributeProto::FLOATS,
            OPTIONAL)
        .Attr(
            "one_class",
            "bool whether the regression is a one class svm or not, defaults to false",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "coefficients",
            "support vector coefficients",
            AttributeProto::FLOATS,
            OPTIONAL)
        .Attr(
            "n_supports",
            "number of support vectors",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "post_transform",
            "post eval transform for score, enum NONE, SOFTMAX, LOGISTIC, SOFTMAX_ZERO, PROBIT",
            AttributeProto::STRING,
            std::string("NONE"))
        .Attr("rho", "", AttributeProto::FLOATS, OPTIONAL));

static const char* TreeEnsembleClassifier_ver1_doc = R"DOC(
    Tree Ensemble classifier.  Returns the top class for each input in N.
    All args with nodes_ are fields of a tuple of tree nodes, and
    it is assumed they are the same length, and an index i will decode the
    tuple across these inputs.  Each node id can appear only once
    for each tree id.
    All fields prefixed with class_ are tuples of votes at the leaves.
    A leaf may have multiple votes, where each vote is weighted by
    the associated class_weights index.
    It is expected that either classlabels_strings or classlabels_int64s
    will be passed and the class_ids are an index into this list.
    Mode enum is BRANCH_LEQ, BRANCH_LT, BRANCH_GTE, BRANCH_GT, BRANCH_EQ, BRANCH_NEQ, LEAF
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    TreeEnsembleClassifier,
    1,
    OpSchema()
        .SetDoc(TreeEnsembleClassifier_ver1_doc)
        .Input(0, "X", "Input N,F", "T1")
        .Output(0, "Y", "N, Top class for each point", "T2")
        .Output(
            1,
            "Z",
            "N,E the class score for each class, for each point",
            "tensor(float)")
        .TypeConstraint(
            "T1",
            {"tensor(float)",
             "tensor(double)",
             "tensor(int64)",
             "tensor(int32)"},
            " allowed types.")
        .TypeConstraint(
            "T2",
            {"tensor(string)", "tensor(int64)"},
            " allowed types.")
        .Attr(
            "nodes_treeids",
            "tree id for this node",
            AttributeProto::INTS,
            OPTIONAL)
        .Attr(
            "nodes_nodeids",
            "node id for this node, node ids may restart at zero for each tree (but not required).",
            AttributeProto::INTS,
            OPTIONAL)
        .Attr(
            "nodes_featureids",
            "feature id for this node",
            AttributeProto::INTS,
            OPTIONAL)
        .Attr(
            "nodes_values",
            "thresholds to do the splitting on for this node.",
            AttributeProto::FLOATS,
            OPTIONAL)
        .Attr("nodes_hitrates", "", AttributeProto::FLOATS, OPTIONAL)
        .Attr(
            "nodes_modes",
            "enum of behavior for this node 'BRANCH_LEQ', 'BRANCH_LT', 'BRANCH_GTE', 'BRANCH_GT', 'BRANCH_EQ', 'BRANCH_NEQ', 'LEAF'",
            AttributeProto::STRINGS,
            OPTIONAL)
        .Attr(
            "nodes_truenodeids",
            "child node if expression is true",
            AttributeProto::INTS,
            OPTIONAL)
        .Attr(
            "nodes_falsenodeids",
            "child node if expression is false",
            AttributeProto::INTS,
            OPTIONAL)
        .Attr(
            "nodes_missing_value_tracks_true",
            "for each node, decide if the value is missing (nan) then use true branch, this field can be left unset and will assume false for all nodes",
            AttributeProto::INTS,
            OPTIONAL)
        .Attr(
            "class_treeids",
            "tree that this node is in",
            AttributeProto::INTS,
            OPTIONAL)
        .Attr(
            "class_nodeids",
            "node id that this weight is for",
            AttributeProto::INTS,
            OPTIONAL)
        .Attr(
            "class_ids",
            "index of the class list that this weight is for",
            AttributeProto::INTS,
            OPTIONAL)
        .Attr(
            "class_weights",
            "the weight for the class in class_id",
            AttributeProto::FLOATS,
            OPTIONAL)
        .Attr(
            "classlabels_strings",
            "class labels if using string labels",
            AttributeProto::STRINGS,
            OPTIONAL)
        .Attr(
            "classlabels_int64s",
            "class labels if using int labels",
            AttributeProto::INTS,
            OPTIONAL)
        .Attr(
            "post_transform",
            "post eval transform for score, enum NONE, SOFTMAX, LOGISTIC, SOFTMAX_ZERO, PROBIT",
            AttributeProto::STRING,
            std::string("NONE"))
        .Attr(
            "base_values",
            "base values for classification, added to final class score, size must be the same as classes or can be left unassigned (assumed 0)",
            AttributeProto::FLOATS,
            OPTIONAL)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          std::vector<std::string> label_strs;
          auto result =
              getRepeatedAttribute(ctx, "classlabels_strings", label_strs);
          bool using_strings = (result && !label_strs.empty());
          auto output_elem_type = ctx.getOutputType(0)->mutable_tensor_type();
          if (using_strings) {
            output_elem_type->set_elem_type(TensorProto::STRING);
          } else {
            output_elem_type->set_elem_type(TensorProto::INT64);
          }
        }));

static const char* TreeEnsembleRegressor_ver1_doc = R"DOC(
    Tree Ensemble regressor.  Returns the regressed values for each input in N.
    All args with nodes_ are fields of a tuple of tree nodes, and
    it is assumed they are the same length, and an index i will decode the
    tuple across these inputs.  Each node id can appear only once
    for each tree id.
    All fields prefixed with target_ are tuples of votes at the leaves.
    A leaf may have multiple votes, where each vote is weighted by
    the associated target_weights index.
    All trees must have their node ids start at 0 and increment by 1.
    Mode enum is BRANCH_LEQ, BRANCH_LT, BRANCH_GTE, BRANCH_GT, BRANCH_EQ, BRANCH_NEQ, LEAF
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    TreeEnsembleRegressor,
    1,
    OpSchema()
        .SetDoc(TreeEnsembleRegressor_ver1_doc)
        .Input(0, "X", "Input N,F", "T")
        .Output(0, "Y", "N classes", "tensor(float)")
        .TypeConstraint(
            "T",
            {"tensor(float)",
             "tensor(double)",
             "tensor(int64)",
             "tensor(int32)"},
            " allowed types.")
        .Attr(
            "nodes_treeids",
            "tree id for this node",
            AttributeProto::INTS,
            OPTIONAL)
        .Attr(
            "nodes_nodeids",
            "node id for this node, node ids must restart at zero for each tree and increase sequentially.",
            AttributeProto::INTS,
            OPTIONAL)
        .Attr(
            "nodes_featureids",
            "feature id for this node",
            AttributeProto::INTS,
            OPTIONAL)
        .Attr(
            "nodes_values",
            "thresholds to do the splitting on for this node.",
            AttributeProto::FLOATS,
            OPTIONAL)
        .Attr(
            "nodes_hitrates",
            "popularity of the node, used for performance and may be omitted",
            AttributeProto::FLOATS,
            OPTIONAL)
        .Attr(
            "nodes_modes",
            "enum of behavior for this node as enum of BRANCH_LEQ, BRANCH_LT, BRANCH_GTE, BRANCH_GT, BRANCH_EQ, BRANCH_NEQ, LEAF",
            AttributeProto::STRINGS,
            OPTIONAL)
        .Attr(
            "nodes_truenodeids",
            "child node if expression is true",
            AttributeProto::INTS,
            OPTIONAL)
        .Attr(
            "nodes_falsenodeids",
            "child node if expression is false",
            AttributeProto::INTS,
            OPTIONAL)
        .Attr(
            "nodes_missing_value_tracks_true",
            "for each node, decide if the value is missing (nan) then use true branch, this field can be left unset and will assume false for all nodes",
            AttributeProto::INTS,
            OPTIONAL)
        .Attr(
            "target_treeids",
            "tree that this node is in",
            AttributeProto::INTS,
            OPTIONAL)
        .Attr(
            "target_nodeids",
            "node id that this weight is for",
            AttributeProto::INTS,
            OPTIONAL)
        .Attr(
            "target_ids",
            "index of the class list that this weight is for",
            AttributeProto::INTS,
            OPTIONAL)
        .Attr(
            "target_weights",
            "the weight for the class in target_id",
            AttributeProto::FLOATS,
            OPTIONAL)
        .Attr(
            "n_targets",
            "total number of targets",
            AttributeProto::INT,
            OPTIONAL)
        .Attr(
            "post_transform",
            "post eval transform for score, enum NONE, SOFTMAX, LOGISTIC, SOFTMAX_ZERO, PROBIT",
            AttributeProto::STRING,
            std::string("NONE"))
        .Attr(
            "aggregate_function",
            " enum, how to aggregate leaf values within a target, AVERAGE,SUM,MIN,MAX",
            AttributeProto::STRING,
            std::string("SUM"))
        .Attr(
            "base_values",
            "base values for regression, added to final score, size must be the same as n_outputs or can be left unassigned (assumed 0)",
            AttributeProto::FLOATS,
            OPTIONAL));

static const char* ZipMap_ver1_doc = R"DOC(
    Makes a map from the input and the attributes.
    Assumes input 0 are the values, and the keys are specified by the attributes.
    Must provide keys in either classlabels_strings or classlabels_int64s (but not both).
    Input 0 may have a batch size larger than 1,
    but each input in the batch must be the size of the keys specified by the attributes.
    The order of the input and attributes determines the key-value mapping.
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    ZipMap,
    1,
    OpSchema()
        .SetDoc(ZipMap_ver1_doc)
        .Input(0, "X", "The input values", "tensor(float)")
        .Output(0, "Z", "The output map", "T")
        .TypeConstraint(
            "T",
            {"seq(map(string, float))", "seq(map(int64, float))"},
            " allowed types.")
        .Attr(
            "classlabels_strings",
            "keys if using string keys",
            AttributeProto::STRINGS,
            OPTIONAL)
        .Attr(
            "classlabels_int64s",
            "keys if using int keys",
            AttributeProto::INTS,
            OPTIONAL)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          std::vector<std::string> classlabels_strings;
          bool result = getRepeatedAttribute(
              ctx, "classlabels_strings", classlabels_strings);
          auto output_map_type = ctx.getOutputType(0)
                                     ->mutable_sequence_type()
                                     ->mutable_elem_type()
                                     ->mutable_map_type();
          output_map_type->mutable_value_type()
              ->mutable_tensor_type()
              ->set_elem_type(TensorProto::FLOAT);
          if (result && !classlabels_strings.empty()) {
            output_map_type->set_key_type(TensorProto::STRING);
          }
          std::vector<int64_t> classlabels_int64s;
          result = getRepeatedAttribute(
              ctx, "classlabels_int64s", classlabels_int64s);
          if (result && !classlabels_int64s.empty()) {
            output_map_type->set_key_type(TensorProto::INT64);
          }
        }));

static const char* FeatureVectorizer_ver1_doc = R"DOC(
    Concatenates input features into one continuous output of floats.
    inputdimensions is the size of each input feature.
    Inputs will be written to the output in the order of the input arguments.
    If an input tensor is shorter than its matching input dimension the output will be padded with zeros.
    If an input tensor is longer than its matching input dimension, the additional input will be ignored.
    Input tensors must all be of the same type. Use Cast as needed.  
    Input tensors must all be of the same batch size.
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    FeatureVectorizer,
    1,
    OpSchema()
        .SetDoc(FeatureVectorizer_ver1_doc)
        .Input(0, "X", "ordered input tensors", "T1", OpSchema::Variadic)
        .Output(
            0,
            "Y",
            "Output array, in same order as Input, as floats",
            "tensor(float)")
        .TypeConstraint(
            "T1",
            {"tensor(int32)",
             "tensor(int64)",
             "tensor(float)",
             "tensor(double)"},
            " Allowed input types")
        .Attr(
            "inputdimensions",
            "the size of each input in the input list",
            AttributeProto::INTS,
            OPTIONAL));

} // namespace ONNX_NAMESPACE
#endif
