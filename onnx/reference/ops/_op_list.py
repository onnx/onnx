# SPDX-License-Identifier: Apache-2.0
# pylint: disable=C0415,R0912,R0913,R0914,R0915,W0611,W0603
"""
Every class imported in this module defines an implementation of
an operator of the main domain. Any class name uses `_` to specify a
version defined in a specific opset. The class name without `_`
defines the current implementation. If an operator has no class
with `_`, it means the implementation is valid for every opset.
The operator may have been updated to support more types but that
did not change the implementation.
"""
import textwrap
from typing import Any, List, Union

from onnx import FunctionProto, NodeProto, TypeProto
from onnx.defs import get_schema, onnx_opset_version
from onnx.onnx_cpp2py_export.defs import SchemaError
from onnx.reference.op_run import (
    OpFunction,
    OpRun,
    RuntimeContextError,
    RuntimeImplementationError,
    _split_class_name,
)

from .op_abs import Abs
from .op_acos import Acos
from .op_acosh import Acosh
from .op_add import Add
from .op_and import And
from .op_argmax import ArgMax
from .op_argmin import ArgMin
from .op_asin import Asin
from .op_asinh import Asinh
from .op_atan import Atan
from .op_atanh import Atanh
from .op_attribute_has_value import AttributeHasValue
from .op_average_pool import AveragePool
from .op_batch_normalization import (
    BatchNormalization,
    BatchNormalization_6,
    BatchNormalization_9,
    BatchNormalization_14,
)
from .op_bernoulli import Bernoulli
from .op_bitshift import BitShift
from .op_bitwise_and import BitwiseAnd
from .op_bitwise_not import BitwiseNot
from .op_bitwise_or import BitwiseOr
from .op_bitwise_xor import BitwiseXor
from .op_blackman_window import BlackmanWindow
from .op_cast import Cast
from .op_cast_like import CastLike
from .op_ceil import Ceil
from .op_celu import Celu
from .op_center_crop_pad import CenterCropPad
from .op_clip import Clip, Clip_6, Clip_11
from .op_col2im import Col2Im
from .op_compress import Compress
from .op_concat import Concat
from .op_concat_from_sequence import ConcatFromSequence
from .op_constant import Constant, Constant_1, Constant_9, Constant_11, Constant_12
from .op_constant_of_shape import ConstantOfShape
from .op_conv import Conv
from .op_conv_integer import ConvInteger
from .op_conv_transpose import ConvTranspose
from .op_cos import Cos
from .op_cosh import Cosh
from .op_cum_sum import CumSum
from .op_depth_to_space import DepthToSpace
from .op_dequantize_linear import DequantizeLinear
from .op_det import Det
from .op_dft import DFT
from .op_div import Div
from .op_dropout import Dropout, Dropout_7, Dropout_12
from .op_dynamic_quantize_linear import DynamicQuantizeLinear
from .op_einsum import Einsum
from .op_elu import Elu
from .op_equal import Equal
from .op_erf import Erf
from .op_exp import Exp
from .op_expand import Expand
from .op_eyelike import EyeLike
from .op_flatten import Flatten
from .op_floor import Floor
from .op_gather import Gather
from .op_gather_elements import GatherElements
from .op_gathernd import GatherND
from .op_gemm import Gemm, Gemm_6, Gemm_7
from .op_global_average_pool import GlobalAveragePool
from .op_global_max_pool import GlobalMaxPool
from .op_greater import Greater
from .op_greater_or_equal import GreaterOrEqual
from .op_grid_sample import GridSample
from .op_gru import GRU
from .op_hamming_window import HammingWindow
from .op_hann_window import HannWindow
from .op_hard_sigmoid import HardSigmoid
from .op_hardmax import Hardmax
from .op_identity import Identity
from .op_if import If
from .op_instance_normalization import InstanceNormalization
from .op_isinf import IsInf
from .op_isnan import IsNaN
from .op_layer_normalization import LayerNormalization
from .op_leaky_relu import LeakyRelu
from .op_less import Less
from .op_less_or_equal import LessOrEqual
from .op_log import Log
from .op_log_softmax import LogSoftmax
from .op_loop import Loop
from .op_lp_normalization import LpNormalization
from .op_lrn import LRN
from .op_lstm import LSTM
from .op_matmul import MatMul
from .op_matmul_integer import MatMulInteger
from .op_max import Max
from .op_max_pool import MaxPool
from .op_max_unpool import MaxUnpool
from .op_mean import Mean
from .op_mel_weight_matrix import MelWeightMatrix
from .op_min import Min
from .op_mod import Mod
from .op_mul import Mul
from .op_neg import Neg
from .op_negative_log_likelihood_loss import NegativeLogLikelihoodLoss
from .op_non_max_suppression import NonMaxSuppression
from .op_non_zero import NonZero
from .op_not import Not
from .op_one_hot import OneHot
from .op_optional import Optional
from .op_optional_get_element import OptionalGetElement
from .op_optional_has_element import OptionalHasElement
from .op_or import Or
from .op_pad import Pad, Pad_1, Pad_2, Pad_11, Pad_18
from .op_pow import Pow
from .op_prelu import PRelu
from .op_qlinear_conv import QLinearConv
from .op_qlinear_matmul import QLinearMatMul
from .op_quantize_linear import QuantizeLinear
from .op_random_normal import RandomNormal
from .op_random_normal_like import RandomNormalLike
from .op_random_uniform import RandomUniform
from .op_random_uniform_like import RandomUniformLike
from .op_range import Range
from .op_reciprocal import Reciprocal
from .op_reduce_l1 import ReduceL1
from .op_reduce_l2 import ReduceL2
from .op_reduce_log_sum import ReduceLogSum
from .op_reduce_log_sum_exp import ReduceLogSumExp
from .op_reduce_max import ReduceMax
from .op_reduce_mean import ReduceMean
from .op_reduce_min import ReduceMin
from .op_reduce_prod import ReduceProd
from .op_reduce_sum import ReduceSum, ReduceSum_1, ReduceSum_11, ReduceSum_13
from .op_reduce_sum_square import ReduceSumSquare
from .op_relu import Relu
from .op_reshape import Reshape, Reshape_5, Reshape_14
from .op_resize import Resize
from .op_reverse_sequence import ReverseSequence
from .op_rnn import RNN
from .op_roi_align import RoiAlign
from .op_round import Round
from .op_scan import Scan
from .op_scatter_elements import ScatterElements
from .op_scatternd import ScatterND
from .op_selu import Selu
from .op_sequence_at import SequenceAt
from .op_sequence_construct import SequenceConstruct
from .op_sequence_empty import SequenceEmpty
from .op_sequence_erase import SequenceErase
from .op_sequence_insert import SequenceInsert
from .op_sequence_length import SequenceLength
from .op_sequence_map import SequenceMap
from .op_shape import Shape
from .op_shrink import Shrink
from .op_sigmoid import Sigmoid
from .op_sign import Sign
from .op_sin import Sin
from .op_sinh import Sinh
from .op_size import Size
from .op_slice import Slice, Slice_1, Slice_10
from .op_softmax import Softmax
from .op_softmax_cross_entropy_loss import SoftmaxCrossEntropyLoss
from .op_softplus import Softplus
from .op_softsign import Softsign
from .op_space_to_depth import SpaceToDepth
from .op_split import Split, Split_2, Split_11, Split_13
from .op_split_to_sequence import SplitToSequence
from .op_sqrt import Sqrt
from .op_squeeze import Squeeze, Squeeze_1, Squeeze_11, Squeeze_13
from .op_stft import STFT
from .op_string_normalizer import StringNormalizer
from .op_sub import Sub
from .op_sum import Sum
from .op_tan import Tan
from .op_tanh import Tanh
from .op_tfidf_vectorizer import TfIdfVectorizer
from .op_thresholded_relu import ThresholdedRelu
from .op_tile import Tile
from .op_topk import TopK, TopK_1, TopK_10, TopK_11
from .op_transpose import Transpose
from .op_trilu import Trilu
from .op_unique import Unique
from .op_unsqueeze import Unsqueeze, Unsqueeze_1, Unsqueeze_11, Unsqueeze_13
from .op_upsample import Upsample
from .op_where import Where
from .op_xor import Xor


def _build_registered_operators():  # type: ignore
    clo = globals().copy()
    reg_ops = {}  # type: ignore
    for class_name, class_type in clo.items():
        if class_name[0] == "_" or class_name in {
            "Any",
            "cl",
            "clo",
            "class_name",
            "get_schema",
            "List",
            "textwrap",
            "Union",
        }:
            continue  # pragma: no cover
        if isinstance(class_type, type(load_op)):
            continue
        try:
            issub = issubclass(class_type, OpRun)
        except TypeError as e:
            raise TypeError(
                f"Unexpected variable type {class_type!r} and class_name={class_name!r}."
            ) from e
        if issub:
            op_type, op_version = _split_class_name(class_name)
            if op_type not in reg_ops:
                reg_ops[op_type] = {}
            reg_ops[op_type][op_version] = class_type
    if len(reg_ops) == 0:
        raise RuntimeError("No registered operators. The installation went wrong.")
    return reg_ops


def load_op(
    domain: str,
    op_type: str,
    version: Union[None, int] = None,
    custom: Any = None,
    node: Union[None, NodeProto] = None,
    input_types: Union[None, List[TypeProto]] = None,
) -> Any:
    """
    Loads the implemented for a specified operator.

    :param domain: domain
    :param op_type: oprator type
    :param version: requested version
    :param custom: custom implementation (like a function)
    :param node: used if no implementation was found and the operator defines a function
        which is context dependant
    :param input_types: used if no implementation was found and the operator defines a function
        which is context dependant
    :return: class
    """
    global _registered_operators
    schema = None
    if _registered_operators is None:
        _registered_operators = _build_registered_operators()
    if custom is not None:
        return lambda *args: OpFunction(*args, impl=custom)  # type: ignore
    if version is None:
        version = onnx_opset_version()
    if domain != "":
        raise ValueError(f"Domain must be '' not {domain!r}.")
    if op_type in _registered_operators:  # type: ignore
        found = True
    else:
        # maybe the operator can be replacted by a function
        try:
            schema = get_schema(op_type, version, domain)  # type: ignore
        except SchemaError:
            raise NotImplementedError(  # pylint: disable=W0707
                f"No registered schema for operator {op_type!r} "
                f"and domain {domain!r}. Did you recompile the sources after updating the repository?"
            ) from None
        if schema.has_function:  # type: ignore
            from onnx.reference import ReferenceEvaluator

            body = schema.function_body  # type: ignore
            sess = ReferenceEvaluator(body)
            return lambda *args, sess=sess: OpFunction(*args, impl=sess)  # type: ignore
        if schema.has_context_dependent_function:  # type: ignore
            if node is None or input_types is None:
                raise RuntimeContextError(
                    f"No registered implementation for operator {op_type!r} "
                    f"and domain {domain!r}, the operator has a context dependent function. "
                    f"but argument node or input_types is not defined."
                )
            from onnx.reference import ReferenceEvaluator

            body = schema.get_context_dependent_function(  # type: ignore
                node.SerializeToString(), [it.SerializeToString() for it in input_types]
            )
            proto = FunctionProto()
            proto.ParseFromString(body)
            sess = ReferenceEvaluator(proto)
            return lambda *args, sess=sess: OpFunction(*args, impl=sess)  # type: ignore
        found = False
    if not found:
        available = "\n".join(textwrap.wrap(", ".join(sorted(_registered_operators))))  # type: ignore
        has_function = schema.has_function if schema else None  # type: ignore
        has_context_dependent_function = (
            schema.has_context_dependent_function if schema else None  # type: ignore
        )
        raise RuntimeImplementationError(
            f"No registered implementation for operator {op_type!r} "
            f"and domain {domain!r}, schema.has_function is {has_function}, "
            f"schema.has_context_dependent_function is {has_context_dependent_function}. "
            f"You may either add one or skip the test in "
            f"'reference_evaluator_bakcend_test.py'. Available implementations:\n{available}"
        )
    impl = _registered_operators[op_type]  # type: ignore
    if None not in impl:
        raise RuntimeError(
            f"No default implementation for operator {op_type!r} "
            f"and domain {domain!r}, found "
            f"{', '.join(map(str, impl))}."
        )
    if version is None or len(impl) == 1:
        cl = impl[None]
    else:
        best = -1
        for v in impl:
            if v is None:
                continue
            if best < v <= version:
                best = v
        if best == -1:
            raise RuntimeError(
                f"No implementation for operator {op_type!r} "
                f"domain {domain!r} and version {version!r}, found "
                f"{', '.join(map(str, impl))}."
            )
        cl = impl[best]
    if cl is None:
        available = "\n".join(textwrap.wrap(", ".join(sorted(_registered_operators))))  # type: ignore
        raise ValueError(
            f"Not registered implementation for operator {op_type!r}, "
            f"domain {domain!r}, and {version!r} in\n{available}"
        )
    return cl


_registered_operators = None
