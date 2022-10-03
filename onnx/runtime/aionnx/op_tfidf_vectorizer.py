# SPDX-License-Identifier: Apache-2.0
# pylint: disable=C0200,R0902,R0912,R0913,R0914,R0915,R1716,W0612,W0221

from enum import IntEnum
from pprint import pformat
from typing import List

import numpy as np  # type: ignore

from ..op_run import OpRun


class IntMap(dict):  # type: ignore
    def __init__(self):
        dict.__init__(self)
        self.added_keys = []

    def emplace(self, key, value):
        if not isinstance(key, int):
            raise TypeError(f"key must be a NGramPart not {type(key)}.")
        if not isinstance(value, NgramPart):
            raise TypeError(f"value must be a NGramPart not {type(value)}.")
        self.added_keys.append(key)
        self[key] = value
        return self[key]

    def __repr__(self):
        vals = {k: repr(v) for k, v in self.items()}
        return f"IntMap({pformat(vals)})"

    @property
    def first_key(self):
        if len(self) == 0:
            raise ValueError("IntMap is empty.")
        return self.added_keys[0]


class NgramPart:
    def __init__(self, nid: int):
        self.id_ = nid  # 0 - means no entry, search for a bigger N
        self._leafs_ = None

    def init(self):
        self._leafs_ = IntMap()  # type: ignore

    def __repr__(self):
        if self.empty():
            return f"NgramPart({self.id_})"
        return f"NgramPart({self.id_}, {repr(self.leafs_)})"

    def empty(self):
        return self._leafs_ is None

    def has_leaves(self):
        return self._leafs_ is not None and len(self._leafs_) > 0

    @property
    def leafs_(self):
        if self._leafs_ is None:
            raise RuntimeError("NgramPart was not initialized.")
        return self._leafs_

    def find(self, key):
        if not self.has_leaves():
            return None
        if key in self._leafs_:  # type: ignore
            return key
        return None

    def emplace(self, key, value):
        return self.leafs_.emplace(key, value)

    def __getitem__(self, key):
        return self._leafs_[key]


class WeightingCriteria(IntEnum):
    kNone = 0
    kTF = 1
    kIDF = 2
    kTFIDF = 3


def populate_grams(
    els,
    els_index,
    n_ngrams: int,
    ngram_size: int,
    ngram_id: int,
    c,  # : ForwardIter ,  # Map
):
    for ngrams in range(n_ngrams, 0, -1):
        n = 1
        m = c
        while els_index < len(els):
            p = m.emplace(els[els_index], NgramPart(-1))
            if n == ngram_size:
                p.id_ = ngram_id
                ngram_id += 1
                break
            if p.empty():
                p.init()
            m = p.leafs_
            n += 1
            els_index += 1
    return ngram_id


class TfIdfVectorizer(OpRun):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRun.__init__(self, onnx_node, run_params)
        mode = self.mode  # type: ignore

        if mode == "TF":
            self.weighting_criteria_ = WeightingCriteria.kTF
        elif mode == "IDF":
            self.weighting_criteria_ = WeightingCriteria.kIDF
        elif mode == "TFIDF":
            self.weighting_criteria_ = WeightingCriteria.kTFIDF

        self.min_gram_length_ = self.min_gram_length  # type: ignore
        self.max_gram_length_ = self.max_gram_length  # type: ignore
        self.max_skip_count_ = self.max_skip_count  # type: ignore
        self.ngram_counts_ = self.ngram_counts  # type: ignore
        self.max_gram_length_ = self.max_gram_length  # type: ignore
        self.ngram_indexes_ = self.ngram_indexes  # type: ignore
        self.output_size_ = max(self.ngram_indexes_) + 1
        self.weights_ = self.weights  # type: ignore
        self.pool_int64s_ = self.pool_int64s  # type: ignore

        self.int64_map_ = NgramPart(-2)
        self.int64_map_.init()

        total_items = len(self.pool_int64s_)
        ngram_id = 1  # start with 1, 0 - means no n-gram
        # Load into dictionary only required gram sizes
        ngram_size = 1
        for i in range(len(self.ngram_counts_)):

            start_idx = self.ngram_counts_[i]
            end_idx = (
                self.ngram_counts_[i + 1]
                if (i + 1) < len(self.ngram_counts_)
                else total_items
            )
            items = end_idx - start_idx
            if items > 0:
                ngrams = items // ngram_size
                if (
                    ngram_size >= self.min_gram_length_
                    and ngram_size <= self.max_gram_length_
                ):
                    ngram_id = populate_grams(
                        self.pool_int64s_,
                        start_idx,
                        ngrams,
                        ngram_size,
                        ngram_id,
                        self.int64_map_,
                    )
                else:
                    ngram_id += ngrams
            ngram_size += 1

    def increment_count(
        self, ngram_id: int, row_num: int, frequencies: List[int]
    ) -> None:
        ngram_id -= 1
        # assert(ngram_id < ngram_indexes_.size());
        output_idx = row_num * self.output_size_ + self.ngram_indexes_[ngram_id]
        # assert(static_cast<size_t>(output_idx) < frequencies.size());
        frequencies[output_idx] += 1

    def output_result(self, B: int, frequencies: List[int]) -> np.ndarray:
        output_dims: List[int] = []
        if B == 0:
            output_dims.append(self.output_size_)
            B = 1
        else:
            output_dims.append(B)
            output_dims.append(self.output_size_)

        row_size = self.output_size_

        total_dims = np.prod(output_dims)
        Y = np.empty((total_dims,), dtype=np.float)

        w = self.weights_
        if self.weighting_criteria_ == WeightingCriteria.kTF:
            i = 0
            for f in frequencies:
                Y[i] = f
                i += 1
        elif self.weighting_criteria_ == WeightingCriteria.kIDF:
            if len(w) > 0:
                p = 0
                for batch in range(B):
                    for i in range(row_size):
                        Y[p] = w[i] if frequencies[p] > 0 else 0
                        p += 1
            else:
                p = 0
                for f in frequencies:
                    Y[p] = 1 if f > 0 else 0
                    p += 1
        elif self.weighting_criteria_ == WeightingCriteria.kTFIDF:
            if len(w) > 0:
                p = 0
                for batch in range(B):
                    for i in range(row_size):
                        Y[p] = w[i] * frequencies[p]
                        p += 1
            else:
                p = 0
                for f in frequencies:
                    Y[p] = f
                    p += 1
        else:
            raise RuntimeError("Unexpected weighting_criteria.")
        return Y

    def compute_impl(
        self,
        X: np.ndarray,
        row_num: int,
        row_size: int,
        frequencies: List[int],
        max_gram_length=None,
        max_skip_count=None,
        min_gram_length=None,
        mode=None,
        ngram_counts=None,
        ngram_indexes=None,
        pool_int64s=None,
        pool_strings=None,
        weights=None,
    ) -> None:

        X_flat = X.flatten()
        row_begin = row_num * row_size
        row_end = row_begin + row_size

        max_skip_distance = max_skip_count + 1
        start_ngram_size = min_gram_length

        for skip_distance in range(1, max_skip_distance + 1):
            ngram_start = row_begin
            ngram_row_end = row_end

            while ngram_start < ngram_row_end:
                # We went far enough so no n-grams of any size can be gathered
                at_least_this = ngram_start + skip_distance * (start_ngram_size - 1)
                if at_least_this >= ngram_row_end:
                    break

                ngram_item = ngram_start
                int_map = self.int64_map_
                ngram_size = 1
                while (
                    int_map.has_leaves()
                    and ngram_size <= max_gram_length
                    and ngram_item < ngram_row_end
                ):
                    val = X_flat[ngram_item]
                    hit = int_map.find(val)
                    if hit is None:
                        break
                    hit = int_map[val].id_
                    if ngram_size >= start_ngram_size and hit != -1:
                        self.increment_count(hit, row_num, frequencies)
                    int_map = int_map[val]
                    ngram_size += 1
                    ngram_item += skip_distance

                ngram_start += 1

            # We count UniGrams only once since they are not affected by skip_distance
            if start_ngram_size == 1:
                start_ngram_size += 1
                if start_ngram_size > max_gram_length:
                    break

    def _run(  # type: ignore
        self,
        X,
        max_gram_length=None,
        max_skip_count=None,
        min_gram_length=None,
        mode=None,
        ngram_counts=None,
        ngram_indexes=None,
        pool_int64s=None,
        pool_strings=None,
        weights=None,
    ):
        # weights should be identical to self.weights as well as
        # pool_strings, pool_int64s, ngram_indexes, ngram_counts, mode.
        # This means none of those attributes can be used in one function.

        total_items = np.prod(X.shape)

        num_rows = 0
        B = 0
        C = 0
        input_dims = X.shape
        if len(input_dims) == 0:
            num_rows = 1
            C = 1
            if total_items != 1:
                raise ValueError(f"Unexpected total of items {total_items}.")
        elif len(input_dims) == 1:
            num_rows = 1
            C = input_dims[0]
        elif len(input_dims) == 2:
            B = input_dims[0]
            C = input_dims[1]
            num_rows = B
            if B < 1:
                raise ValueError(
                    f"Input shape must have either [C] or [B,C] dimensions with B > 0, B={B}, C={C}."
                )
        else:
            raise ValueError(
                f"Input shape must have either [C] or [B,C] dimensions with B > 0, B={B}, C={C}."
            )

        if num_rows * C != total_items:
            raise ValueError(
                f"Unexpected total of items, num_rows * C = {num_rows * C} != total_items = {total_items}."
            )
        # Frequency holder allocate [B..output_size_] and init all to zero
        frequencies = [0] * (num_rows * self.output_size_)

        if total_items == 0 or self.int64_map_.empty():
            # TfidfVectorizer may receive an empty input when it follows a Tokenizer
            # (for example for a string containing only stopwords).
            # TfidfVectorizer returns a zero tensor of shape
            # {b_dim, output_size} when b_dim is the number of received observations
            # and output_size the is the maximum value in ngram_indexes attribute plus 1.
            return self.output_result(B, frequencies)

        def fn(row_num):
            self.compute_impl(
                X,
                row_num,
                C,
                frequencies,
                max_gram_length=max_gram_length,
                max_skip_count=max_skip_count,
                min_gram_length=min_gram_length,
                mode=mode,
                ngram_counts=ngram_counts,
                ngram_indexes=ngram_indexes,
                pool_int64s=pool_int64s,
                pool_strings=pool_strings,
                weights=weights,
            )

        # can be parallelized.
        for i in range(num_rows):
            fn(i)

        return (self.output_result(B, frequencies),)
