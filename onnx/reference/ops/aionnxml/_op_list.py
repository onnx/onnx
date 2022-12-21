# SPDX-License-Identifier: Apache-2.0
# pylint: disable=C0415,R0912,W0611,W0603
# Operator ZipMap is not implemented. Its use should
# be discouraged. It is just a different way to output
# probabilites not consumed by any operator.

import textwrap
from typing import Any, Union

from onnx.reference.op_run import OpFunction, _split_class_name

from ._op_run_aionnxml import OpRunAiOnnxMl
from .op_array_feature_extractor import ArrayFeatureExtractor
from .op_binarizer import Binarizer
from .op_dict_vectorizer import DictVectorizer
from .op_feature_vectorizer import FeatureVectorizer
from .op_imputer import Imputer
from .op_label_encoder import LabelEncoder
from .op_linear_classifier import LinearClassifier
from .op_linear_regressor import LinearRegressor
from .op_normalizer import Normalizer
from .op_one_hot_encoder import OneHotEncoder
from .op_scaler import Scaler
from .op_svm_classifier import SVMClassifier
from .op_svm_regressor import SVMRegressor
from .op_tree_ensemble_classifier import TreeEnsembleClassifier
from .op_tree_ensemble_regressor import TreeEnsembleRegressor


def _build_registered_operators():  # type: ignore
    clo = globals().copy()
    reg_ops = {}  # type: ignore
    for class_name, class_type in clo.items():
        if class_name[0] == "_" or class_name in {
            "Any",
            "cl",
            "clo",
            "class_name",
            "textwrap",
            "Union",
        }:
            continue  # pragma: no cover
        if isinstance(class_type, type(load_op)):
            continue
        try:
            issub = issubclass(class_type, OpRunAiOnnxMl)
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
    domain: str, op_type: str, version: Union[None, int], custom: Any = None
) -> Any:
    """
    Loads the implemented for a specified operator.

    :param domain: domain
    :param op_type: oprator type
    :param version: requested version
    :param custom: custom implementation (like a function)
    :return: class
    """
    global _registered_operators
    if _registered_operators is None:
        _registered_operators = _build_registered_operators()
    if custom is not None:
        return lambda *args: OpFunction(*args, impl=custom)  # type: ignore
    if domain != "ai.onnx.ml":
        raise ValueError(f"Domain must be '' not {domain!r}.")
    if op_type not in _registered_operators:  # type: ignore
        available = "\n".join(textwrap.wrap(", ".join(sorted(_registered_operators))))  # type: ignore
        raise NotImplementedError(
            f"No registered implementation for operator {op_type!r} "
            f"and domain {domain!r} in\n{available}"
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
