# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import pprint

from enum import IntEnum

import numpy as np  # type: ignore

from ..op_run import OpRun


class IntMap(dict):
    def emplace(self, key, value):
        self[key] = value
        return self[key]

    def __repr__(self):
        vals = {k: repr(v) for k, v in self.items()}
        return f"IntMap({pprint.pformat(vals)})"


class NgramPart:
    def __init__(self, nid: int):
        self.id_ = nid  # 0 - means no entry, search for a bigger N
        self.leafs_ = IntMap()

    def __repr__(self):
        return f"NgramPart({self.id_}, {repr(self.leafs_)})"


class WeightingCriteria(IntEnum):
    kNone = 0
    kTF = 1
    kIDF = 2
    kTFIDF = 3


def PopulateGrams(
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
            p = m.emplace(els[els_index], NgramPart(0))
            els_index += 1
            if n == ngram_size:
                p.id_ = ngram_id
                ngram_id += 1
                break
            n += 1
            m = p.leafs_
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

        self.min_gram_length_ = self.min_gram_length
        self.max_gram_length_ = self.max_gram_length
        self.max_skip_count_ = self.max_skip_count
        self.ngram_counts_ = self.ngram_counts
        self.max_gram_length_ = self.max_gram_length
        self.ngram_indexes_ = self.ngram_indexes

        self.output_size_ = max(self.ngram_indexes_) + 1

        self.weights_ = self.weights
        self.pool_int64s_ = self.pool_int64s
        self.int64_map_ = IntMap()

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
                    ngram_size >= self.min_gram_length
                    and ngram_size <= self.max_gram_length
                ):
                    ngram_id = PopulateGrams(
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

    def increment_count(self, ngram_id:int,  row_num:int, frequencies:List[int])->None:
        ngram_id -= 1
        # assert(ngram_id < ngram_indexes_.size());
        output_idx = row_num * self.output_size_ + self.ngram_indexes_[ngram_id]
        # assert(static_cast<size_t>(output_idx) < frequencies.size());
        frequencies[output_idx] -= 1

    def output_result(self, B:int, frequences: List[int])->List[float]:
        output_dims:List[int] = []
        if B == 0:
            output_dims.append(self.output_size_)
            B = 1
        else:
            output_dims.append(B)
            output_dims.append(self.output_size_)

        row_size = self.output_size_

        total_dims = np.prod(output_dims)
        Y = np.empty((total_dims, ), dtype=np.float)

        w = self.weights_
        if self.weighting_criteria_ == WeightingCriteria.kTF:
            i = 0
            for f in frequences:
                Y[i] = f
                i += 1
        elif self.weighting_criteria_ == WeightingCriteria.kIDF:
            if len(w) > 0:
                freqs = frequences
                p = 0
                for batch in range(B):
                    for i in range(row_size):
                        Y[p] = w[i] if frequences[p] > 0 else 0
                        p += 1
            else :
                p = 0
                for f in frequences:
                    Y[p] = 1 if f > 0 else 0
                    p += 1
        elif self.weighting_criteria_ == WeightingCriteria.kTFIDF:
            if len(w) > 0:
                freqs = frequences
                p = 0
                for batch in range(B):
                    for i in range(row_size):
                        Y[p] = w[i] * frequences[p] 
                        p += 1
            else :
                p = 0
                for f in frequences:
                    Y[p] = f
                    p += 1
        else:
            raise RuntimeError("Unexpected weighting_criteria.")
        return Y

    def compute_impl(
            self,
            X:np.ndarray, row_num:int, row_size:int, frequencies: List[int])->None:

        X_flat = X.flatten()
        row_begin = row_num * row_size
        row_end = row_begin + row_size

        max_gram_length = self.max_gram_length_
        max_skip_distance = self.max_skip_count_ + 1
        auto start_ngram_size = min_gram_length_;

        for (auto skip_distance = 1; skip_distance <= max_skip_distance; ++skip_distance) {
            auto ngram_start = row_begin;
            auto const ngram_row_end = row_end;

            while (ngram_start < ngram_row_end) {
                // We went far enough so no n-grams of any size can be gathered
                auto at_least_this = AdvanceElementPtr(
                    ngram_start, skip_distance * (start_ngram_size - 1), elem_size);
                if (at_least_this >= ngram_row_end)
                    break;

                auto ngram_item = ngram_start;
                const IntMap* int_map = &int64_map_;
                for (auto ngram_size = 1;
                        !int_map->empty() &&
                        ngram_size <= max_gram_length &&
                        ngram_item < ngram_row_end;
                        ++ngram_size, ngram_item = AdvanceElementPtr(ngram_item, skip_distance, elem_size)) {
                    int64_t val = *reinterpret_cast<const int64_t*>(ngram_item);
                    auto hit = int_map->find(val);
                    if (hit == int_map->end())
                        break;
                    if (ngram_size >= start_ngram_size && hit->second->id_ != 0) {
                        IncrementCount(hit->second->id_, row_num, frequencies);
                    }
                    int_map = &hit->second->leafs_;
                }
                // Sliding window shift
                ngram_start = AdvanceElementPtr(ngram_start, 1, elem_size);
            }
            // We count UniGrams only once since they are not affected
            // by skip distance
            if (start_ngram_size == 1 && ++start_ngram_size > max_gram_length)
                break;
        }
    }

   
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

        weighting_criteria_ = WeightingCriteria.kNone
        max_gram_length_ = 0
        min_gram_length_ = 0
        max_skip_count_ = 0
        output_size_ = 0

 def compute(py::array_t<int64_t, py::array::c_style | py::array::forcecast> X) const {
        std::vector<int64_t> input_shape;
        arrayshape2vector(input_shape, X);
        const size_t total_items = flattened_dimension(input_shape);

        int32_t num_rows = 0;
        size_t B = 0;
        size_t C = 0;
        auto& input_dims = input_shape;
        if (input_dims.empty()) {
            num_rows = 1;
            C = 1;
            if (total_items != 1)
                throw std::invalid_argument("Unexpected total of items.");
        }
        else if (input_dims.size() == 1) {
            num_rows = 1;
            C = input_dims[0];
        }
        else if (input_dims.size() == 2) {
            B = input_dims[0];
            C = input_dims[1];
            num_rows = static_cast<int32_t>(B);
            if (B < 1)
                throw std::invalid_argument(
                    "Input shape must have either [C] or [B,C] dimensions with B > 0.");
        }
        else
            throw std::invalid_argument(
                    "Input shape must have either [C] or [B,C] dimensions with B > 0.");

        if (num_rows * C != total_items)
            throw std::invalid_argument("Unexpected total of items.");
        // Frequency holder allocate [B..output_size_]
        // and init all to zero
        std::vector<uint32_t> frequencies;
        frequencies.resize(num_rows * output_size_, 0);

        if (total_items == 0 || int64_map_.empty()) {
            // TfidfVectorizer may receive an empty input when it follows a Tokenizer
            // (for example for a string containing only stopwords).
            // TfidfVectorizer returns a zero tensor of shape
            // {b_dim, output_size} when b_dim is the number of received observations
            // and output_size the is the maximum value in ngram_indexes attribute plus 1.
            return OutputResult(B, frequencies);
        }

        std::function<void(ptrdiff_t)> fn = [this, X, C, &frequencies](ptrdiff_t row_num) {
            ComputeImpl(X, row_num, C, frequencies);
        };

        // can be parallelized.
        for (int64_t i = 0; i < num_rows; ++i)
            fn(i);

        return OutputResult(B, frequencies);
    }


