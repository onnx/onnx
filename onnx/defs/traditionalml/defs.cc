// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"

using AttrType = onnx::OpSchema::AttrType;
using namespace onnx;
#ifdef ONNX_ML
OPERATOR_SCHEMA(ArrayFeatureExtractor)
.SetDomain("ai.onnx.ml")
.NumInputs(2)
.NumOutputs(1)
.SetDoc(R"DOC(
    Select a subset of the data based on the indices passed.
)DOC")
.Input(0, "X", "Data to be selected", "T1")
.Input(1, "Y", "The index values to select as a tensor of int64s", "T2")
.Output(0, "Z", "Selected output data as an array", "T1")
.TypeConstraint(
    "T1",
    { "tensor(float)",
    "tensor(double)",
    "tensor(int64)",
    "tensor(int32)",
    "tensor(string)" },
    " allowed types.")
.TypeConstraint("T2", { "tensor(int64)" }, " Index value types .");

OPERATOR_SCHEMA(Binarizer)
.SetDomain("ai.onnx.ml")
.NumInputs(1)
.NumOutputs(1)
.SetDoc(R"DOC(
    Makes values 1 or 0 based on a single threshold.
)DOC")
.Input(0, "X", "Data to be binarized", "T")
.Output(0, "Y", "Binarized output data", "T")
.TypeConstraint(
    "T",
    { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)" },
    " allowed types.")
.Attr(
    "threshold",
    "Values greater than this are set to 1, else set to 0",
    AttrType::FLOAT);

OPERATOR_SCHEMA(CastMap)
.SetDomain("ai.onnx.ml")
.NumInputs(1)
.NumOutputs(1)
.SetDoc(R"DOC(
    Converts a map to a tensor.  Map key must be int64 and the values will be ordered
    in ascending order based on this key.  Supports dense packing or sparse packing.
    If using sparse packing, the key cannot exceed the max_map-1 value.
)DOC")
.Input(0, "X", "Data to be encoded", "T1")
.Output(0, "Y", "encoded output data", "T2")
.TypeConstraint(
    "T1",
    { "map(int64, string)", "map(int64, float)" },
    " allowed input types.")
.TypeConstraint(
    "T2",
    { "tensor(string)", "tensor(float)", "tensor(int64)" },
    " allowed output types.")
.Attr(
    "cast_to",
    "what to cast output to, enum 'TO_FLOAT', 'TO_STRING', 'TO_INT64', default is 'TO_FLOAT'",
    AttrType::STRING)
.Attr(
    "map_form",
    "whether to only output as many values as are in the input, or position the input based on using the key of the map as the index of the output (sparse), enum 'DENSE', 'SPARSE', default is 'DENSE'",
    AttrType::STRING)
.Attr(
    "max_map",
    "if map_form packing is SPARSE, what is the total length of each output in N (max index value)",
    AttrType::INT);

OPERATOR_SCHEMA(CategoryMapper)
.SetDomain("ai.onnx.ml")
.NumInputs(1)
.NumOutputs(1)
.SetDoc(R"DOC(
    Convert strings to int64s and vice versa.
    Takes in a map to use for the conversion.
    The index position in the strings and ints repeated inputs
     is used to do the mapping.
    Each instantiated operator converts either ints to strings or strings to ints.
    This behavior is triggered based on which default value is set.
    If the string default value is set, it will convert ints to strings.
    If the int default value is set, it will convert strings to ints.
)DOC")
.Input(0, "X", "Input data", "T1")
.Output(
    0,
    "Y",
    "Output data, if strings are input, then output is int64s, and vice versa.",
    "T2")
.TypeConstraint(
    "T1",
    { "tensor(string)", "tensor(int64)" },
    " allowed types.")
.TypeConstraint(
    "T2",
    { "tensor(string)", "tensor(int64)" },
    " allowed types.")
.Attr(
    "cats_strings",
    "strings part of the input map, must be same size and the ints",
    AttrType::STRINGS)
.Attr(
    "cats_int64s",
    "ints part of the input map, must be same size and the strings",
    AttrType::INTS)
.Attr(
    "default_string",
    "string value to use if the int is not in the map",
    AttrType::STRING)
.Attr(
    "default_int64",
    "int value to use if the string is not in the map",
    AttrType::INT);

OPERATOR_SCHEMA(DictVectorizer)
.SetDomain("ai.onnx.ml")
.NumInputs(1)
.NumOutputs(1)
.SetDoc(R"DOC(
    Uses an index mapping to convert a dictionary to an array.
    The output array will be equal in length to the index mapping vector parameter.
    All keys in the input dictionary must be present in the index mapping vector.
    For each item in the input dictionary, insert its value in the ouput array.
    The position of the insertion is determined by the position of the item's key
    in the index mapping. Any keys not present in the input dictionary, will be
    zero in the output array.  Use either string_vocabulary or int64_vocabulary, not both.
    For example: if the ``string_vocabulary`` parameter is set to ``["a", "c", "b", "z"]``,
    then an input of ``{"a": 4, "c": 8}`` will produce an output of ``[4, 8, 0, 0]``.
    )DOC")
.Input(0, "X", "The input dictionary", "T")
.Output(0, "Y", "The tensor", "tensor(int64)")
.TypeConstraint(
    "T",
    { "map(string, int64)", "map(int64, string)" },
    " allowed types.")
.Attr("string_vocabulary", "The vocabulary vector", AttrType::STRINGS)
.Attr("int64_vocabulary", "The vocabulary vector", AttrType::INTS);

OPERATOR_SCHEMA(Imputer)
.SetDomain("ai.onnx.ml")
.NumInputs(1)
.NumOutputs(1)
.SetDoc(R"DOC(
    Replace imputs that equal replaceValue/s  with  imputeValue/s.
    All other inputs are copied to the output unchanged.
    This op is used to replace missing values where we know what a missing value looks like.
    Only one of imputed_value_floats or imputed_value_int64s should be used.
    The size can be 1 element, which will be reused, or the size of the feature set F in input N,F
)DOC")
.Input(0, "X", "Data to be imputed", "T")
.Output(0, "Y", "Imputed output data", "T")
.TypeConstraint(
    "T",
    { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)" },
    " allowed types.")
.Attr("imputed_value_floats", "value to change to", AttrType::FLOATS)
.Attr("replaced_value_float", "value that needs replacing", AttrType::FLOAT)
.Attr("imputed_value_int64s", "value to change to", AttrType::INTS)
.Attr("replaced_value_int64", "value that needs replacing", AttrType::INT);

OPERATOR_SCHEMA(LabelEncoder)
.SetDomain("ai.onnx.ml")
.NumInputs(1)
.NumOutputs(1)
.SetDoc(R"DOC(
    Convert class label to their integral type and vice versa.
    In both cases the operator is instantiated with the list of class strings.
    The integral value of the string is the index position in the list.
)DOC")
.Input(0, "X", "Data to be encoded", "T1")
.Output(0, "Y", "Encoded output data", "T2")
.TypeConstraint(
    "T1",
    { "tensor(string)", "tensor(int64)" },
    " allowed types.")
.TypeConstraint(
    "T2",
    { "tensor(string)", "tensor(int64)" },
    " allowed types.")
.Attr(
    "classes_strings",
    "List of class label strings to be encoded as int64s",
    AttrType::STRINGS)
.Attr(
    "default_int64",
    "Default value if not in class list as int64",
    AttrType::INT)
.Attr(
    "default_string",
    "Default value if not in class list as string",
    AttrType::STRING);

OPERATOR_SCHEMA(LinearClassifier)
.SetDomain("ai.onnx.ml")
.NumInputs(1)
.NumOutputs(2)
.SetDoc(R"DOC(
    Linear classifier prediction (choose class)
)DOC")
.Input(0, "X", "Data to be classified", "T1")
.Output(0, "Y", "Classification outputs (one class per example", "T2")
.Output(
    1,
    "Z",
    "Classification scores (N,E - one score for each class, for each example",
    "tensor(float)")
.TypeConstraint(
    "T1",
    { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)" },
    " allowed types.")
.TypeConstraint(
    "T2",
    { "tensor(string)", "tensor(int64)" },
    " allowed types.")
.Attr("coefficients", "weights of the model(s)", AttrType::FLOATS)
.Attr("intercepts", "weights of the intercepts (if used)", AttrType::FLOATS)
.Attr(
    "multi_class",
    "whether to do OvR or multinomial (0=OvR and is default)",
    AttrType::INT)
.Attr(
    "classlabels_strings",
    "class labels if using string labels",
    AttrType::STRINGS)
.Attr(
    "classlabels_ints",
    "class labels if using int labels",
    AttrType::INTS)
.Attr(
    "post_transform",
    "enum NONE, SOFTMAX, LOGISTIC, SOFTMAX_ZERO, PROBIT",
    AttrType::STRING);

OPERATOR_SCHEMA(LinearRegressor)
.SetDomain("ai.onnx.ml")
.NumInputs(1)
.NumOutputs(1)
.SetDoc(R"DOC(
    Generalized linear regression evaluation.
    If targets is set to 1 (default) then univariate regression is performed.
    If targets is set to M then M sets of coefficients must be passed in as a sequence
    and M results will be output for each input n in N.
    Coefficients are of the same length as an n, and coefficents for each target are contiguous.
    Intercepts are optional but if provided must match the number of targets.
)DOC")
.Input(0, "X", "Data to be regressed", "T")
.Output(
    0,
    "Y",
    "Regression outputs (one per target, per example",
    "tensor(float)")
.TypeConstraint(
    "T",
    { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)" },
    " allowed types.")
.Attr(
    "post_transform",
    "enum NONE, SOFTMAX, LOGISTIC, SOFTMAX_ZERO, PROBIT",
    AttrType::STRING)
.Attr("coefficients", "weights of the model(s)", AttrType::FLOATS)
.Attr("intercepts", "weights of the intercepts (if used)", AttrType::FLOATS)
.Attr(
    "targets",
    "total number of regression targets (default is 1)",
    AttrType::INT);

OPERATOR_SCHEMA(Normalizer)
.SetDomain("ai.onnx.ml")
.NumInputs(1)
.NumOutputs(1)
.SetDoc(R"DOC(
    Normalize the input.  There are three normalization modes,
    which have the corresponding formulas:
    Max .. math::     max(x_i)
    L1  .. math::  z = ||x||_1 = \sum_{i=1}^{n} |x_i|
    L2  .. math::  z = ||x||_2 = \sqrt{\sum_{i=1}^{n} x_i^2}
)DOC")
.Input(0, "X", "Data to be encoded", "T")
.Output(0, "Y", "encoded output data", "tensor(float)")
.TypeConstraint(
    "T",
    { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)" },
    " allowed types.")
.Attr("norm", "0=Lmax, 1=L1, 2=L2", AttrType::STRING);

OPERATOR_SCHEMA(OneHotEncoder)
.SetDomain("ai.onnx.ml")
.NumInputs(1)
.NumOutputs(1)
.SetDoc(R"DOC(
    Replace the inputs with an array of ones and zeros, where the only
    one is the zero-based category that was passed in.  The total category count 
    will determine the length of the vector. For example if we pass a 
    tensor with a single value of 4, and a category count of 8, the 
    output will be a tensor with 0,0,0,0,1,0,0,0 .

    This operator assumes every input in X is of the same category set 
    (meaning there is only one category count).
)DOC")
.Input(0, "X", "Data to be encoded", "T")
.Output(0, "Y", "encoded output data", "tensor(float)")
.TypeConstraint("T", { "tensor(string)", "tensor(int64)" }, " allowed types.")
.Attr("cats_int64s", "list of cateogries, ints", AttrType::INT)
.Attr("cats_strings", "list of cateogries, strings", AttrType::STRINGS)
.Attr(
    "zeros",
    "if true and category is not present, will return all zeros, if false and missing category, operator will return false",
    AttrType::INT);

OPERATOR_SCHEMA(Scaler)
.SetDomain("ai.onnx.ml")
.NumInputs(1)
.NumOutputs(1)
.SetDoc(R"DOC(
    Rescale input data, for example to standardize features by removing the mean and scaling to unit variance.
)DOC")
.Input(0, "X", "Data to be scaled", "T")
.Output(0, "Y", "Scaled output data", "tensor(float)")
.TypeConstraint(
    "T",
    { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)" },
    " allowed types.")
.Attr(
    "scale",
    "second, multiply by this, can be length of features or length 1",
    AttrType::FLOATS)
.Attr(
    "offset",
    "first, offset by this, must be same length as scale",
    AttrType::FLOATS);

OPERATOR_SCHEMA(SVMClassifier)
.SetDomain("ai.onnx.ml")
.NumInputs(1)
.NumOutputs(2)
.SetDoc(R"DOC(
    SVM classifier prediction 
)DOC")
.Input(0, "X", "Data to be classified", "T1")
.Output(0, "Y", "Classification outputs (one class per example)", "T2")
.Output(
    1,
    "Z",
    "Class scores (one per class per example), if prob_a and prob_b are provided they are probabilities for each class otherwise they are raw scores.",
    "tensor(float)")
.TypeConstraint(
    "T1",
    { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)" },
    " allowed types.")
.TypeConstraint(
    "T2",
    { "tensor(string)", "tensor(int64)" },
    " allowed types.")
.Attr(
    "kernel_type",
    "enum LINEAR, POLY, RBF, SIGMOID, defaults to linear",
    AttrType::STRING)
.Attr(
    "kernel_params",
    "Tensor of 3 elements containing gamma, coef0, degree in that order.  Zero if unused for the kernel.",
    AttrType::FLOATS)
.Attr("vectors_per_class", "", AttrType::INTS)
.Attr("support_vectors", "", AttrType::FLOATS)
.Attr("coefficients", "", AttrType::FLOATS)
.Attr("prob_a", "First set of probability coefficients", AttrType::FLOATS)
.Attr(
    "prob_b",
    "Second set of probability coefficients, must be same size as prob_a, if these are provided then output Z are probability estimates.",
    AttrType::FLOATS)
.Attr("rho", "", AttrType::FLOATS)
.Attr(
    "post_transform",
    "post eval transform for score, enum NONE, SOFTMAX, LOGISTIC, SOFTMAX_ZERO, PROBIT",
    AttrType::STRING)
.Attr(
    "classlabels_strings",
    "class labels if using string labels",
    AttrType::STRINGS)
.Attr(
    "classlabels_ints",
    "class labels if using int labels",
    AttrType::INTS);

OPERATOR_SCHEMA(SVMRegressor)
.SetDomain("ai.onnx.ml")
.NumInputs(1)
.NumOutputs(1)
.SetDoc(R"DOC(
    SVM regression prediction and one-class svm anomaly detection
)DOC")
.Input(0, "X", "Data to be regressed", "T")
.Output(
    0,
    "Y",
    "Regression outputs (one score per target per example)",
    "tensor(float)")
.TypeConstraint(
    "T",
    { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)" },
    " allowed types.")
.Attr(
    "kernel_type",
    "enum LINEAR, POLY, RBF, SIGMOID, defaults to linear",
    AttrType::STRING)
.Attr(
    "kernel_params",
    "Tensor of 3 elements containing gamma, coef0, degree in that order.  Zero if unused for the kernel.",
    AttrType::FLOATS)
.Attr("support_vectors", "chosen support vectors", AttrType::FLOATS)
.Attr(
    "one_class",
    "bool whether the regression is a one class svm or not, defaults to false",
    AttrType::INT)
.Attr("coefficients", "support vector coefficients", AttrType::FLOATS)
.Attr("n_supports", "number of support vectors", AttrType::INT)
.Attr(
    "post_transform",
    "post eval transform for score, enum NONE, SOFTMAX, LOGISTIC, SOFTMAX_ZERO, PROBIT",
    AttrType::STRING)
.Attr("rho", "", AttrType::FLOATS);

OPERATOR_SCHEMA(TreeEnsembleClassifier)
.SetDomain("ai.onnx.ml")
.NumInputs(1)
.NumOutputs(2)
.SetDoc(R"DOC(
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
)DOC")
.Input(0, "X", "Input N,F", "T1")
.Output(0, "Y", "N, Top class for each point", "T2")
.Output(
    1,
    "Z",
    "N,E the class score for each class, for each point",
    "tensor(float)")
.TypeConstraint(
    "T1",
    { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)" },
    " allowed types.")
.TypeConstraint(
    "T2",
    { "tensor(string)", "tensor(int64)" },
    " allowed types.")
.Attr("nodes_treeids", "tree id for this node", AttrType::INTS)
.Attr(
    "nodes_nodeids",
    "node id for this node, node ids may restart at zero for each tree (but not required).",
    AttrType::INTS)
.Attr("nodes_featureids", "feature id for this node", AttrType::INTS)
.Attr(
    "nodes_values",
    "thresholds to do the splitting on for this node.",
    AttrType::FLOATS)
.Attr("nodes_hitrates", "", AttrType::FLOATS)
.Attr(
    "nodes_modes",
    "enum of behavior for this node 'BRANCH_LEQ', 'BRANCH_LT', 'BRANCH_GTE', 'BRANCH_GT', 'BRANCH_EQ', 'BRANCH_NEQ', 'LEAF'",
    AttrType::STRINGS)
.Attr(
    "nodes_truenodeids",
    "child node if expression is true",
    AttrType::INTS)
.Attr(
    "nodes_falsenodeids",
    "child node if expression is false",
    AttrType::INTS)
.Attr(
    "nodes_missing_value_tracks_true",
    "for each node, decide if the value is missing (nan) then use true branch, this field can be left unset and will assume false for all nodes",
    AttrType::INTS)
.Attr("class_treeids", "tree that this node is in", AttrType::INTS)
.Attr("class_nodeids", "node id that this weight is for", AttrType::INTS)
.Attr(
    "class_ids",
    "index of the class list that this weight is for",
    AttrType::INTS)
.Attr(
    "class_weights",
    "the weight for the class in class_id",
    AttrType::FLOATS)
.Attr(
    "classlabels_strings",
    "class labels if using string labels",
    AttrType::STRINGS)
.Attr(
    "classlabels_int64s",
    "class labels if using int labels",
    AttrType::INTS)
.Attr(
    "post_transform",
    "post eval transform for score, enum NONE, SOFTMAX, LOGISTIC, SOFTMAX_ZERO, PROBIT",
    AttrType::STRING)
.Attr(
    "base_values",
    "base values for classification, added to final class score, size must be the same as classes or can be left unassigned (assumed 0)",
    AttrType::FLOATS);

OPERATOR_SCHEMA(TreeEnsembleRegressor)
.SetDomain("ai.onnx.ml")
.NumInputs(1)
.NumOutputs(1)
.SetDoc(R"DOC(
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
)DOC")
.Input(0, "X", "Input N,F", "T")
.Output(0, "Y", "N classes", "tensor(float)")
.TypeConstraint(
    "T",
    { "tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)" },
    " allowed types.")
.Attr("nodes_treeids", "tree id for this node", AttrType::INTS)
.Attr(
    "nodes_nodeids",
    "node id for this node, node ids must restart at zero for each tree and increase sequentially.",
    AttrType::INTS)
.Attr("nodes_featureids", "feature id for this node", AttrType::INTS)
.Attr(
    "nodes_values",
    "thresholds to do the splitting on for this node.",
    AttrType::FLOATS)
.Attr(
    "nodes_hitrates",
    "popularity of the node, used for performance and may be omitted",
    AttrType::FLOATS)
.Attr(
    "nodes_modes",
    "enum of behavior for this node as enum of BRANCH_LEQ, BRANCH_LT, BRANCH_GTE, BRANCH_GT, BRANCH_EQ, BRANCH_NEQ, LEAF",
    AttrType::STRINGS)
.Attr(
    "nodes_truenodeids",
    "child node if expression is true",
    AttrType::INTS)
.Attr(
    "nodes_falsenodeids",
    "child node if expression is false",
    AttrType::INTS)
.Attr(
    "nodes_missing_value_tracks_true",
    "for each node, decide if the value is missing (nan) then use true branch, this field can be left unset and will assume false for all nodes",
    AttrType::INTS)
.Attr("target_treeids", "tree that this node is in", AttrType::INTS)
.Attr("target_nodeids", "node id that this weight is for", AttrType::INTS)
.Attr(
    "target_ids",
    "index of the class list that this weight is for",
    AttrType::INTS)
.Attr(
    "target_weights",
    "the weight for the class in target_id",
    AttrType::FLOATS)
.Attr("n_targets", "total number of targets", AttrType::INT)
.Attr(
    "post_transform",
    "post eval transform for score, enum NONE, SOFTMAX, LOGISTIC, SOFTMAX_ZERO, PROBIT",
    AttrType::STRING)
.Attr(
    "aggregate_function",
    " enum, how to aggregate leaf values within a target, AVERAGE,SUM,MIN,MAX",
    AttrType::STRING)
.Attr(
    "base_values",
    "base values for regression, added to final score, size must be the same as n_outputs or can be left unassigned (assumed 0)",
    AttrType::FLOATS);

OPERATOR_SCHEMA(ZipMap)
.SetDomain("ai.onnx.ml")
.NumInputs(1)
.NumOutputs(1)
.SetDoc(R"DOC(
    Makes a map from the input and the attributes.  
    Assumes input 0 are the values, and the keys are specified by the attributes.
    Must provide keys in either classlabels_strings or classlabels_int64s (but not both).
    Input 0 may have a batch size larger than 1, 
    but each input in the batch must be the size of the keys specified by the attributes.
    The order of the input and attributes determines the key-value mapping.
)DOC")
.Input(0, "X", "The input values", "tensor(float)")
.Output(0, "Z", "The output map", "T")
.TypeConstraint(
    "T",
    { "map(string, float)", "map(int64, float)" },
    " allowed types.")
.Attr("classlabels_strings", "keys if using string keys", AttrType::STRINGS)
.Attr("classlabels_int64s", "keys if using int keys", AttrType::INTS);

OPERATOR_SCHEMA(FeatureVectorizer)
.SetDomain("ai.onnx.ml")
.NumInputs(1, INT_MAX)
.NumOutputs(1)
.SetDoc(R"DOC(
    Concatenates input features into one continuous output.  
    Inputlist is a list of input feature names, inputdimensions is the size of each input feature.
    Inputs will be written to the output in the order of the input arguments.
    All inputs are tensors of float.  Any feature that is not a tensor of float should
    be converted using either Cast or CastMap.
)DOC")
.Input(0, "X", "ordered input tensors", "T", OpSchema::Variadic)
.Output(0, "Y", "Full output array, in order assigned in the inputlist, as floats", "T")
.TypeConstraint("T", { "tensor(float)" }, " allowed types.")
.Attr("inputdimensions", "the size of each input in the input list", AttrType::INT);

#endif