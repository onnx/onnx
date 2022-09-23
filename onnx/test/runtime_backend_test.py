# SPDX-License-Identifier: Apache-2.0
# type: ignore
# pylint: disable=C0415,R0912,R0913,R0914,R0915,W0640,W0703
"""
These test evaluates the python runtime (class Inference) against
all the backend tests (in onnx/backend/test/case/node) and checks
the runtime produces the expected outputs.
"""

import os
import unittest

import numpy as np
from numpy import object_ as dtype_object
from numpy.testing import assert_allclose  # type: ignore

import onnx.runtime as rt
from onnx import (
    ModelProto,
    OptionalProto,
    SequenceProto,
    TensorProto,
    TypeProto,
    load,
    load_model_from_string,
    load_tensor_from_string,
)
from onnx.backend.test import __file__ as backend_folder
from onnx.helper import __file__ as onnx_file
from onnx.mapping import OPTIONAL_ELEMENT_TYPE_TO_FIELD, TENSOR_TYPE_TO_NP_TYPE
from onnx.numpy_helper import to_array, to_list, to_optional


def assert_allclose_string(expected, value):
    """
    Compares two arrays knowing they contain strings.
    Raises an exception if the test fails.

    :param expected: expected array
    :param value: value
    """

    def is_float(x):
        try:
            float(x)
            return True
        except ValueError:
            return False

    if all(map(is_float, expected.ravel())):
        expected_float = expected.astype(np.float32)
        value_float = value.astype(np.float32)
        assert_allclose(expected_float, value_float)
    else:
        if expected.tolist() != value.tolist():
            raise AssertionError(f"Mismatches {expected} != {value}.")


class OnnxBackendTest:
    """
    Definition of a backend test. It starts with a folder,
    in this folder, one onnx file must be there, then a subfolder
    for each test to run with this model.

    :param folder: test folder
    :param onnx_path: onnx file
    :param onnx_model: loaded onnx file
    :param tests: list of test
    """

    @staticmethod
    def _sort(filenames):
        temp = []
        for f in filenames:
            name = os.path.splitext(f)[0]
            i = name.split("_")[-1]
            temp.append((int(i), f))
        temp.sort()
        return [_[1] for _ in temp]

    @staticmethod
    def _read_proto_from_file(full):
        if not os.path.exists(full):
            raise FileNotFoundError(f"File not found: {full!r}.")
        with open(full, "rb") as f:
            serialized = f.read()
        try:
            loaded = to_array(load_tensor_from_string(serialized))
        except Exception as e:
            proto_types = [SequenceProto, TypeProto, OptionalProto]
            read_obj = None
            for pt in proto_types:
                obj = pt()
                try:
                    obj.ParseFromString(serialized)
                    read_obj = obj
                    break
                except Exception:
                    try:
                        loaded = load_model_from_string(serialized)
                    except Exception:
                        raise RuntimeError(
                            f"Unable to read {full!r}, error is {e}, content is {serialized[:100]!r}."
                        ) from e
            if read_obj is not None:
                if isinstance(obj, SequenceProto):
                    if obj.elem_type == 0:
                        loaded = []
                    else:
                        try:
                            loaded = to_list(read_obj)
                        except Exception as ee:
                            raise AssertionError(f"Unable to read {full!r}.") from ee
                else:
                    loaded = read_obj
        return loaded

    @staticmethod
    def _load(folder, names):
        res = []
        for name in names:
            full = os.path.join(folder, name)
            new_tensor = OnnxBackendTest._read_proto_from_file(full)
            if isinstance(new_tensor, (np.ndarray, ModelProto, list)):
                t = new_tensor
            elif isinstance(new_tensor, TensorProto):
                t = to_array(new_tensor)
            elif isinstance(new_tensor, OptionalProto):
                try:
                    t = to_optional(new_tensor)
                except TypeError as e:
                    if new_tensor.name == "seq_empty":
                        t = None
                    else:
                        raise TypeError(
                            f"Unable to convert {type(new_tensor)} into python.\n{str(new_tensor)}\n."
                        ) from e
                except ValueError as e:
                    elem_type = new_tensor.elem_type
                    value_field = OPTIONAL_ELEMENT_TYPE_TO_FIELD[elem_type]
                    value = getattr(new_tensor, value_field)
                    if isinstance(value, TensorProto):
                        # something went wrong, one reason is the dimension do not fit raw_data
                        el_type = value.data_type
                        dtype = TENSOR_TYPE_TO_NP_TYPE[el_type]
                        t = np.frombuffer(value.raw_data, dtype=dtype)
                    else:
                        raise ValueError(
                            f"Unable to convert {type(new_tensor)} into python.\n{str(new_tensor)}\n."
                        ) from e
            else:
                raise RuntimeError(f"Unexpected type {type(new_tensor)} for {full!r}.")
            res.append(t)
        return res

    def __repr__(self):
        "usual"
        return f"{self.__class__.__name__}({self.folder!r})"

    def __init__(self, folder):
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Unable to find folder {folder!r}.")
        content = os.listdir(folder)
        onx = [c for c in content if os.path.splitext(c)[-1] in {".onnx"}]
        if len(onx) != 1:
            raise ValueError(
                f"There is more than one onnx file in {folder!r} ({onx!r})."
            )
        self.folder = folder
        self.onnx_path = os.path.join(folder, onx[0])
        self.onnx_model = load(self.onnx_path)

        self.tests = []
        for sub in content:
            full = os.path.join(folder, sub)
            if os.path.isdir(full):
                pb = [c for c in os.listdir(full) if os.path.splitext(c)[-1] in {".pb"}]
                inputs = OnnxBackendTest._sort(c for c in pb if c.startswith("input_"))
                outputs = OnnxBackendTest._sort(
                    c for c in pb if c.startswith("output_")
                )
                t = dict(
                    inputs=OnnxBackendTest._load(full, inputs),
                    outputs=OnnxBackendTest._load(full, outputs),
                )
                self.tests.append(t)

    @property
    def name(self):
        "Returns the test name."
        return os.path.split(self.folder)[-1]

    def __len__(self):
        "Returns the number of tests."
        return len(self.tests)

    def _compare_results(self, index, i_output, desired, output, rtol=0, atol=0):
        """
        Compares the expected output and the output produced
        by the runtime. Raises an exception if not equal.

        :param index: test index
        :param i_output: output index
        :param desired: expected output
        :param output: output
        :param rtol: relative tolerance
        :param atol: absolute tolerance
        """
        if atol is None:
            atol = 0
        if rtol is None:
            rtol = 0
        if isinstance(desired, np.ndarray):
            if isinstance(output, np.ndarray):
                if rtol == 0:
                    if desired.dtype == np.float32:
                        rtl = 1e-5
                    elif desired.dtype == np.float64:
                        rtl = 1e-12
                    else:
                        rtl = rtol
                else:
                    rtl = rtol
                if desired.dtype == dtype_object:
                    try:
                        assert_allclose_string(desired, output)
                    except AssertionError as ex:
                        raise AssertionError(
                            f"Output {i_output} of test {index} in folder {self.folder!r} failed."
                        ) from ex
                else:
                    try:
                        assert_allclose(desired, output, atol=atol, rtol=rtl)
                    except AssertionError as ex:
                        raise AssertionError(
                            f"Output {i_output} of test {index} in folder {self.folder!r} failed "
                            f"(rtol={rtl}, atol={atol})\n---\n{desired}\n----\n{output}."
                        ) from ex
            elif hasattr(output, "is_compatible"):
                # A shape
                if desired.dtype != output.dtype:
                    raise AssertionError(
                        f"Output {i_output} of test {index} in folder {self.folder!r} failed "
                        f"(desired.dtype={desired.dtype!r}, output={output!r})."
                    )
                if not output.is_compatible(desired.shape):
                    raise AssertionError(
                        f"Output {i_output} of test {index} in folder {self.folder!r} failed "
                        f"(desired.shape={desired.shape}, output={output!r})."
                    )
        elif isinstance(desired, list):
            if not isinstance(output, list):
                raise AssertionError(
                    f"Expected result is 'list' but output type is {type(output)} for output {i_output}"
                    f"\n--EXPECTED--\n{desired}\n--GOT--\n{output}."
                )
            if len(desired) != len(output):
                raise AssertionError(
                    f"Expected has {len(desired)} but output has {len(output)} for output {i_output}"
                    f"\n--EXPECTED--\n{desired}\n--GOT--\n{output}."
                )
            for a, b in zip(desired, output):
                self._compare_results(index, i_output, a, b, rtol=rtol, atol=atol)
        else:
            raise NotImplementedError(
                f"Comparison not implemented for type {type(desired)} and output {i_output}."
            )

    def is_random(self):
        "Tells if a test is random or not."
        if "bernoulli" in self.folder:
            return True
        return False

    def run(self, load_fct, run_fct, index=None, rtol=1e-07, atol=0):
        """
        Executes a tests or all tests if index is None.
        The function crashes if the tests fails.

        :param load_fct: loading function, takes a loaded onnx graph,
            and returns an object
        :param run_fct: running function, takes the result of previous
            function, the inputs, and returns the outputs
        :param index: index of the test to run or all.
        :param rtol: relative tolerance
        :param atol: absolute tolerance
        """
        if index is None:
            for i in range(len(self)):
                self.run(load_fct, run_fct, index=i, atol=atol, rtol=rtol)
            return

        obj = load_fct(self.onnx_model)

        got = run_fct(obj, *self.tests[index]["inputs"])
        expected = self.tests[index]["outputs"]
        if len(got) != len(expected):
            raise AssertionError(
                f"Unexpected number of output (test {index}, folder {self.folder!r}), "
                f"got {len(got)}, expected {len(expected)}."
            )
        for i, (e, o) in enumerate(zip(expected, got)):
            if self.is_random():
                if e.dtype != o.dtype:
                    raise AssertionError(
                        f"Output {i} of test {index} in folder {self.folder!r} failed "
                        f"(type mismatch {e.dtype} != {o.dtype!r})."
                    )
                if e.shape != o.shape:
                    raise AssertionError(
                        f"Output {i} of test {index} in folder {self.folder!r} failed "
                        f"(shape mismatch {e.shape} != {o.shape})."
                    )
            else:
                self._compare_results(index, i, e, o, atol=atol, rtol=rtol)


def enumerate_onnx_tests(series, fct_filter=None):
    """
    Collects test from a sub folder of `onnx/backend/test`.
    Works as an enumerator to start processing them
    without waiting or storing too much of them.

    :param series: which subfolder to load, possible values:
        (`'node'`, ...)
    :param fct_filter: function `lambda testname: boolean`
        to load or skip the test, None for all
    :return: list of @see cl OnnxBackendTest
    """
    root = os.path.dirname(backend_folder)
    sub = os.path.join(root, "data", series)
    if not os.path.exists(sub):
        content = "\n".join(os.listdir(root))
        raise FileNotFoundError(
            f"Unable to find series of tests in {root!r}, subfolders:\n{content}"
        )
    tests = os.listdir(sub)
    for t in tests:
        if fct_filter is not None and not fct_filter(t):
            continue
        folder = os.path.join(sub, t)
        content = os.listdir(folder)
        onx = [c for c in content if os.path.splitext(c)[-1] in {".onnx"}]
        if len(onx) == 1:
            yield OnnxBackendTest(folder)


class TestOnnxBackEnd(unittest.TestCase):

    folder = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "onnx_backend_test_code"
    )

    def test_onnx_backend_test(self):
        name = "test_abs"
        code = []
        for te in enumerate_onnx_tests("node", lambda folder: folder == name):
            code.append(te)
        self.assertEqual(len(code), 1)

    @staticmethod
    def load_fct(obj, verbose=0):
        return rt.Inference(obj, verbose=verbose)

    @staticmethod
    def run_fct(obj, *inputs, verbose=0):  # pylint: disable=W0613
        if hasattr(obj, "input_names"):
            input_names = obj.input_names
        elif hasattr(obj, "get_inputs"):
            input_names = [_.name for _ in obj.get_inputs()]
        else:
            raise AttributeError(
                f"Unable to extract the number to guess the number of inputs for type {type(obj)}."
            )
        if len(input_names) < len(inputs):
            raise AssertionError(
                f"Got {len(inputs)} inputs but expecting {len(obj.input_names)}."
            )
        feeds = {input_names[i]: inputs[i] for i in range(len(inputs))}
        got = obj.run(None, feeds)
        return got

    def test_enumerate_onnx_tests_run_one(self):
        done = 0
        for te in enumerate_onnx_tests("node", lambda folder: folder == "test_abs"):
            self.assertIn(te.name, repr(te))
            self.assertGreater(len(te), 0)
            te.run(TestOnnxBackEnd.load_fct, TestOnnxBackEnd.run_fct)
            done += 1
        self.assertEqual(done, 1)

    def common_test_enumerate_onnx_tests_run(
        self, valid, verbose=0, rtol=None, atol=None, check_ort_mismath=False
    ):
        if rtol is None:
            rtol = {}
        if atol is None:
            atol = {}
        with self.assertRaises(FileNotFoundError):
            list(enumerate_onnx_tests("NNN"))
        missed = []
        skipped = []
        load_failed = []
        exec_failed = []
        mismatch = []
        success = 0
        for te in enumerate_onnx_tests("node"):
            if not valid(te.name):
                skipped.append((te,))
                continue
            if verbose > 6:
                print("TEST:", te.name)
            with self.subTest(name=te.name):
                if verbose > 7:
                    print("  check runtime")
                self.assertIn(te.name, repr(te))
                self.assertGreater(len(te), 0)
                try:
                    if verbose > 7:
                        print("  run")
                    if verbose > 5:
                        te.run(
                            lambda *args, verbose=verbose: TestOnnxBackEnd.load_fct(
                                *args, verbose
                            ),
                            TestOnnxBackEnd.run_fct,
                            atol=atol.get(te.name, None),
                            rtol=rtol.get(te.name, None),
                        )
                    else:
                        te.run(
                            TestOnnxBackEnd.load_fct,
                            TestOnnxBackEnd.run_fct,
                            atol=atol.get(te.name, None),
                            rtol=rtol.get(te.name, None),
                        )
                    if verbose > 7:
                        print("  end run")
                        if verbose > 8:
                            print(te.onnx_model)
                except NotImplementedError as e:
                    if verbose > 7:
                        print("  ", e, type(e))
                    missed.append((te, e))
                    continue
                except AssertionError as e:
                    if verbose > 7:
                        print("  ", e, type(e))
                    mismatch.append((te, e))

                    if check_ort_mismath:
                        from onnxruntime import InferenceSession

                        te.run(
                            lambda obj: InferenceSession(obj.SerializeToString()),
                            lambda *a, **b: TestOnnxBackEnd.run_fct(*a, verbose=1, **b),
                        )
                    continue
                except Exception as e:
                    if verbose > 7:
                        print("  ", e, type(e))
                    with open(f"issue_{te.name}.onnx", "wb") as f:
                        f.write(te.onnx_model.SerializeToString())
                    raise AssertionError(
                        f"Unable to run model due to {e}\n{str(te.onnx_model)}"
                    ) from e
                success += 1
                if verbose > 7:
                    print("  end example.")

        failed = [
            len(missed),
            len(skipped),
            len(load_failed),
            len(exec_failed),
            len(mismatch),
        ]
        coverage = success / (success + sum(failed))

        if verbose:
            path = os.path.dirname(onnx_file)
            print("-----------")
            print(
                f"success={success}, skipped={len(skipped)}, missed={len(missed)}, load_failed={len(load_failed)}, "
                f"exec_failed={len(exec_failed)}, mismatch={len(mismatch)}"
            )
            print(
                f"coverage {coverage * 100:.1f}% out of {success + sum(failed)} tests"
            )

            if verbose > 3:

                def _print(s, path):
                    return (
                        str(s)
                        .replace("\\\\", "\\")
                        .replace(path, "onnx")
                        .replace("\\", "/")
                    )

                print("-----------")
                for t in sorted(load_failed, key=lambda m: m[0].name):
                    print("loading failed", _print(t[0], path))
                for t in sorted(exec_failed, key=lambda m: m[0].name):
                    print("execution failed", _print(t[0], path))
                for t in sorted(mismatch, key=lambda m: m[0].name):
                    print("mismatch", _print(t[0], path))
                for t in sorted(missed, key=lambda m: m[0].name):
                    print("missed ", _print(t[0], path))
                for t in sorted(skipped, key=lambda m: m[0].name):
                    print("skipped", _print(t[0], path))

                if success > 30:
                    print("-----------")
                    print(
                        f"success={success}, skipped={len(skipped)}, missed={len(missed)}, load_failed={len(load_failed)}, "
                        f"exec_failed={len(exec_failed)}, mismatch={len(mismatch)}"
                    )
                    print(
                        f"coverage {coverage * 100:.1f}% out of {success + sum(failed)} tests"
                    )
                    print("-----------")

        if len(mismatch) > 0:
            te, e = mismatch[0]
            raise AssertionError(
                f"Mismatch in test {te.name!r}\n{te.onnx_model}."
            ) from e
        if 30 < success < 907:
            raise AssertionError(
                f"The coverage ({coverage * 100:.1f}% out of {success + sum(failed)} tests) "
                f"the runtime among has decreased. New operators were added with no "
                f"corresponding runtime."
            )

    def test_enumerate_onnx_tests_run(self):
        # test not supported yet
        # not supported yet
        # see http://onnx.ai/backend-scoreboard/onnxruntime_details_stable.html
        # to compare with onnxruntime
        skip_test = {
            "test_cast_FLOAT_to_BFLOAT16",
            "test_castlike_FLOAT_to_BFLOAT16_expanded",
            "test_cast_BFLOAT16_to_FLOAT",
            "test_castlike_BFLOAT16_to_FLOAT_expanded",
            "test_castlike_BFLOAT16_to_FLOAT",
            "test_castlike_FLOAT_to_BFLOAT16",
        }
        # discrepancies
        skip_test |= {
            # mismatches
            "test_center_crop_pad_crop_axes_hwc_expanded",
            "test_col2im_pads",
            "test_resize_downsample_scales_cubic_A_n0p5_exclude_outside",
            "test_resize_downsample_scales_cubic_antialias",
            "test_resize_downsample_scales_linear_antialias",
            "test_resize_downsample_sizes_cubic_antialias",
            "test_resize_downsample_sizes_linear_antialias",
            "test_nesterov_momentum",
            "test_resize_upsample_scales_cubic_A_n0p5_exclude_outside",
            # mismatches, problem of truncated values 0.5 -> 0 or 1?
            "test_dynamicquantizelinear_max_adjusted_expanded",
            "test_dynamicquantizelinear_min_adjusted_expanded",
            "test_dynamicquantizelinear_expanded",
            # mismatches, problem with constant pi
            "test_blackmanwindow_expanded",
            "test_blackmanwindow_symmetric_expanded",
            "test_hannwindow_expanded",
            "test_hannwindow_symmetric_expanded",
            "test_hammingwindow_expanded",
            "test_hammingwindow_symmetric_expanded",
            # mistmaches, shape dimension, example does not follow formula from the spec
            "test_convtranspose_autopad_same",
            # bug
            "test_loop16_seq_none",
            # bug
            "test_gru_batchwise",
            "test_resize_downsample_sizes_nearest_not_larger",
            "test_resize_downsample_sizes_nearest_not_smaller",
            "test_resize_tf_crop_and_resize_axes_3_2",
            "test_resize_tf_crop_and_resize_axes_2_3",
            "test_resize_upsample_scales_nearest_axes_2_3",
            "test_resize_upsample_scales_nearest_axes_3_2",
            "test_resize_upsample_sizes_nearest_axes_2_3",
            "test_resize_upsample_sizes_nearest_axes_3_2",
            "test_resize_upsample_sizes_nearest_not_larger",
            "test_scatter_elements_with_reduction_min",
            "test_scatter_elements_with_duplicate_indices",
            "test_simple_rnn_batchwise",
            "test_stft_with_window",
            "test_stft",
            # deprecated
            "test_scan_sum",  # opset 8 -> not implemented
            "test_scatter_with_axis",  # scatter is removed
            "test_scatter_without_axis",  # scatter is removed
        }
        rtol = {
            "test_adam_multiple": 1e-2,
            "test_blackmanwindow_expanded": 0,
            "test_blackmanwindow_symmetric_expanded": 0,
            "test_simple_rnn_batchwise": 0,
        }
        atol = {
            "test_blackmanwindow": 1e-7,
            "test_blackmanwindow_symmetric": 1e-7,
            "test_gru_seq_length": 1e-7,
            "test_hannwindow_symmetric": 1e-7,
            "test_layer_normalization_4d_axis_negative_1_expanded": 1e-6,
            "test_layer_normalization_4d_axis1_expanded": 1e-6,
            "test_layer_normalization_4d_axis_negative_3_expanded": 1e-6,
            "test_mish": 1e-6,
            "test_mish_expanded": 1e-6,
        }

        self.common_test_enumerate_onnx_tests_run(
            valid=lambda name: name not in skip_test,
            verbose=4 if __name__ == "__main__" else 0,
            rtol=rtol,
            atol=atol,
        )

    def test_enumerate_onnx_tests_run_one_case(self):
        self.common_test_enumerate_onnx_tests_run(
            lambda name: "test_abs" == name,
            verbose=0,
            atol={
                "test_blackmanwindow_expanded": 1e-4,
            },
        )


if __name__ == "__main__":
    TestOnnxBackEnd().test_enumerate_onnx_tests_run_one_case()
    unittest.main(verbosity=2)
