# SPDX-License-Identifier: Apache-2.0
# type: ignore
# pylint: disable=C0415,R0912,R0913,R0914,R0915,W0613,W0640,W0703
"""
These test evaluates the python runtime (class ReferenceEvaluator) against
all the backend tests (in onnx/backend/test/case/node) and checks
the runtime produces the expected outputs.
"""

import os
import pprint
import unittest
import warnings

import numpy as np
from numpy import object_ as dtype_object
from numpy.testing import assert_allclose  # type: ignore

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
from onnx.helper import bfloat16_to_float32
from onnx.mapping import OPTIONAL_ELEMENT_TYPE_TO_FIELD, TENSOR_TYPE_TO_NP_TYPE
from onnx.numpy_helper import to_array, to_list, to_optional
from onnx.reference import ReferenceEvaluator
from onnx.reference.ops.op_cast import cast_to


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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                te = load_tensor_from_string(serialized)
                loaded = to_array(te)
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
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", DeprecationWarning)
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

    @property
    def fname(self):
        folder = self.folder.replace("\\", "/").split("/")[-2]
        if folder.endswith("node"):
            fname = self.name
        else:
            fname = f"test__{folder.replace('-', '_')}_{self.name[5:]}"
            if "/" in fname or fname == "test__test_AvgPool1d_AvgPool1d":
                raise AssertionError(
                    f"name={self.name!r}, folder={folder!r}, self.folder={self.folder}."
                )
        return fname

    def __len__(self):
        "Returns the number of tests."
        return len(self.tests)

    def _compare_results(
        self, index, i_output, desired, output, rtol=0, atol=0, comment=""
    ):
        """
        Compares the expected output and the output produced
        by the runtime. Raises an exception if not equal.

        :param index: test index
        :param i_output: output index
        :param desired: expected output
        :param output: output
        :param rtol: relative tolerance
        :param atol: absolute tolerance
        :param comment: addition text to give more insights to the user
        """
        if comment == "":
            raise RuntimeError("Argument comment should be filled.")
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
                            f"Output {i_output} of test {index} in folder {self.folder!r} failed, comment={comment}."
                        ) from ex
                else:
                    try:
                        assert_allclose(desired, output, atol=atol, rtol=rtl)
                    except AssertionError as ex:
                        try:
                            diff = output - desired
                        except ValueError:
                            diff = None
                        raise AssertionError(
                            f"Output {i_output} of test {index} in folder {self.folder!r} failed "
                            f"(rtol={rtl}, atol={atol}), comment={comment}\n---\n{desired}\n----"
                            f"\n{output}\n-----\n{diff}."
                        ) from ex
            elif hasattr(output, "is_compatible"):
                # A shape
                if desired.dtype != output.dtype:
                    raise AssertionError(
                        f"Output {i_output} of test {index} in folder {self.folder!r} failed "
                        f"(desired.dtype={desired.dtype!r}, output={output!r}), comment={comment}."
                    )
                if not output.is_compatible(desired.shape):
                    raise AssertionError(
                        f"Output {i_output} of test {index} in folder {self.folder!r} failed "
                        f"(desired.shape={desired.shape}, output={output!r}), comment={comment}."
                    )
        elif isinstance(desired, list):
            if not isinstance(output, list):
                raise AssertionError(
                    f"Expected result is 'list' but output type is {type(output)} for output {i_output}"
                    f", comment={comment}\n--EXPECTED--\n{desired}\n--GOT--\n{output}."
                )
            if len(desired) != len(output):
                raise AssertionError(
                    f"Expected has {len(desired)} but output has {len(output)} for output {i_output}"
                    f", comment={comment}\n--EXPECTED--\n{desired}\n--GOT--\n{output}."
                )
            for a, b in zip(desired, output):
                self._compare_results(
                    index, i_output, a, b, rtol=rtol, atol=atol, comment=comment
                )
        else:
            raise NotImplementedError(
                f"Comparison not implemented for type {type(desired)} and output {i_output}, comment={comment}."
            )

    def is_random(self):
        "Tells if a test is random or not."
        if "bernoulli" in self.folder:
            return True
        return False

    def run(
        self,
        load_fct,
        run_fct,
        index=None,
        rtol=1e-07,
        atol=0,
        comment="",
        print_io=False,
    ):
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
        :param comment: additional information for the user
        :param print_io: prints out the input and output
        """
        if index is None:
            res = []
            for i in range(len(self)):
                res.append(
                    self.run(
                        load_fct,
                        run_fct,
                        index=i,
                        atol=atol,
                        rtol=rtol,
                        comment=comment,
                        print_io=print_io,
                    )
                )
            return res

        if print_io:
            print("------ INPUTS")
            for k, v in enumerate(self.tests[index]["inputs"]):
                print(f"input {k!r}, shape={v.shape}, dtype={v.dtype}")
            print("------ EXPECTED OUTPUTS")
            for k, v in enumerate(self.tests[index]["outputs"]):
                print(f"output {k!r}, shape={v.shape}, dtype={v.dtype}")

        obj = load_fct(self.onnx_model)

        got = run_fct(obj, *self.tests[index]["inputs"])
        expected = self.tests[index]["outputs"]
        if len(got) != len(expected):
            raise AssertionError(
                f"Unexpected number of output (test {index}, folder {self.folder!r}), "
                f"got {len(got)}, expected {len(expected)}."
            )
        res = dict(
            inputs=self.tests[index]["inputs"],
            expected=self.tests[index]["outputs"],
            results=got,
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
                self._compare_results(
                    index,
                    i,
                    e,
                    o,
                    atol=atol,
                    rtol=rtol,
                    comment=comment + "\n" + str(self.onnx_model),
                )
        return res


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


class TestOnnxBackEndWithReferenceEvaluator(unittest.TestCase):

    folder = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "onnx_backend_test_code"
    )

    @classmethod
    def add_test_methods(cls):
        for folder in ["node", "pytorch-converted", "pytorch-operator", "simple"]:
            for te in enumerate_onnx_tests(folder):

                def _test_(
                    self, te=te, check_other_runtime=None, verbose=0, print_io=False
                ):
                    if te.fname in getattr(cls, "skip_test", set()):
                        cls.skipped.append((te, None))
                        return
                    rtol = getattr(cls, "rtol", {})
                    atol = getattr(cls, "atol", {})
                    if len(rtol) == 0 or len(atol) == 0:
                        raise AssertionError("rtol or atol is empty.")
                    self.common_test_onnx_test_run(
                        te,
                        getattr(cls, "successes", []),
                        getattr(cls, "missed", []),
                        getattr(cls, "skipped", []),
                        getattr(cls, "load_failed", []),
                        getattr(cls, "exec_failed", []),
                        getattr(cls, "mismatch", []),
                        verbose=verbose,
                        rtol=rtol,
                        atol=atol,
                        check_other_runtime=check_other_runtime,
                        print_io=print_io,
                    )

                setattr(TestOnnxBackEndWithReferenceEvaluator, te.fname, _test_)

    def test_onnx_backend_test_abs(self):
        name = "test_abs"
        code = []
        for te in enumerate_onnx_tests("node", lambda folder: folder == name):
            code.append(te)
        self.assertEqual(len(code), 1)

    def test_onnx_backend_test_expand_shape_model1(self):
        name = "test_expand_shape_model1"
        code = []
        for te in enumerate_onnx_tests("simple", lambda folder: folder == name):
            code.append(te)
        self.assertEqual(len(code), 1)

    @staticmethod
    def load_fct(obj, verbose=0):
        return ReferenceEvaluator(obj, verbose=verbose)

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
        rewrite = False
        for i in range(len(inputs)):  # pylint: disable=C0200
            if (
                isinstance(inputs[i], np.ndarray)
                and inputs[i].dtype == np.uint16
                and obj.input_types[i].tensor_type.elem_type != TensorProto.UINT16
            ):
                rewrite = True
        if rewrite:
            # bfloat16 does not exist for numpy.
            inputs = list(inputs)
            for i in range(len(inputs)):  # pylint: disable=C0200
                if (
                    isinstance(inputs[i], np.ndarray)
                    and inputs[i].dtype == np.uint16
                    and obj.input_types[i].tensor_type.elem_type != TensorProto.UINT16
                ):
                    xr = inputs[i].ravel()
                    xf = np.empty(xr.shape[0], dtype=np.float32)
                    for ie in range(xr.shape[0]):
                        el = bfloat16_to_float32(xr[ie])
                        xf[ie] = el
                    inputs[i] = cast_to(
                        xf.astype(np.float32).reshape(inputs[i].shape),
                        TensorProto.BFLOAT16,
                    )
        feeds = {input_names[i]: inputs[i] for i in range(len(inputs))}
        got = obj.run(None, feeds)
        return got

    def test_onnx_test_run_test_abs(self):
        done = 0
        for te in enumerate_onnx_tests("node", lambda folder: folder == "test_abs"):
            self.assertIn(te.name, repr(te))
            self.assertGreater(len(te), 0)
            te.run(
                TestOnnxBackEndWithReferenceEvaluator.load_fct,
                TestOnnxBackEndWithReferenceEvaluator.run_fct,
                comment="[runtime=ReferenceEvaluator]",
            )
            done += 1
        self.assertEqual(done, 1)

    def common_test_onnx_test_run(
        self,
        te,
        successes,
        missed,
        skipped,
        load_failed,
        exec_failed,
        mismatch,
        verbose=0,
        rtol=None,
        atol=None,
        check_other_runtime=None,
        print_io=False,
    ):
        if verbose > 6:
            print("TEST:", te.name)
        if verbose > 7:
            print("  check runtime")
        self.assertIn(te.name, repr(te))
        self.assertGreater(len(te), 0)
        try:
            if verbose > 7:
                print("  run")
            if verbose > 5:
                te.run(
                    lambda *args, verbose=verbose: TestOnnxBackEndWithReferenceEvaluator.load_fct(
                        *args, verbose
                    ),
                    TestOnnxBackEndWithReferenceEvaluator.run_fct,
                    atol=atol.get(te.name, None),
                    rtol=rtol.get(te.name, None),
                    comment=f"[runtime=ReferenceEvaluator, verbose={verbose}]",
                    print_io=print_io,
                )
            else:
                te.run(
                    TestOnnxBackEndWithReferenceEvaluator.load_fct,
                    TestOnnxBackEndWithReferenceEvaluator.run_fct,
                    atol=atol.get(te.fname, atol.get(te.name, None)),
                    rtol=rtol.get(te.fname, rtol.get(te.name, None)),
                    comment="[runtime=ReferenceEvaluator]",
                    print_io=print_io,
                )
            if verbose > 7:
                print("  end run")
                if verbose > 8:
                    print(te.onnx_model)
        except NotImplementedError as e:
            if verbose > 7:
                print("  ", e, type(e))
            missed.append((te, e))
            with open(f"missed_{te.name}.onnx", "wb") as f:
                f.write(te.onnx_model.SerializeToString())
            raise e
        except (AssertionError, ValueError) as e:
            if verbose > 7:
                print("  ", e, type(e))
            mismatch.append((te, e))
            with open(f"mismatch_{te.name}.onnx", "wb") as f:
                f.write(te.onnx_model.SerializeToString())
            if check_other_runtime is None:
                raise e
            if "onnxruntime" in check_other_runtime:
                print("CHECK RUNTIME onnxruntime")
                from onnxruntime import InferenceSession

                te.run(
                    lambda obj: InferenceSession(obj.SerializeToString()),
                    lambda *a, **b: TestOnnxBackEndWithReferenceEvaluator.run_fct(
                        *a, verbose=1, **b
                    ),
                    atol=1e-5,
                    rtol=1e-3,
                    comment="[runtime=onnxruntime]",
                )
                print("done")
            if "mlprodict" in check_other_runtime:
                print("CHECK RUNTIME mlprodict")
                from mlprodict.onnxrt import OnnxInference

                class _Wrap:
                    def __init__(self, sess):
                        self.sess = sess

                    @property
                    def input_names(self):
                        return [i.name for i in self.sess.obj.graph.input]

                    def run(self, unused, feeds, *args, **kwargs):
                        res = self.sess.run(feeds)
                        lres = [res[o.name] for o in self.sess.obj.graph.output]
                        print("#", lres[0].shape)
                        return lres

                te.run(
                    lambda obj: _Wrap(OnnxInference(obj)),
                    lambda *a, **b: TestOnnxBackEndWithReferenceEvaluator.run_fct(
                        *a, verbose=1, **b
                    ),
                    atol=atol.get(te.fname, None),
                    rtol=rtol.get(te.fname, None),
                    comment="[runtime=mlprodict]",
                )
                print("done")
            raise e
        except Exception as e:
            if verbose > 7:
                print("  ", e, type(e))
            with open(f"issue_{te.name}.onnx", "wb") as f:
                f.write(te.onnx_model.SerializeToString())
            raise AssertionError(
                f"Unable to run test {te.name!r} due to {e}\n{str(te.onnx_model)}"
            ) from e
        successes.append((te, atol.get(te.fname, None), rtol.get(te.fname, None)))
        if verbose > 7:
            print("  end example.")

    @staticmethod
    def _postprocess(
        successes, missed, skipped, load_failed, exec_failed, mismatch, verbose
    ):
        success = len(successes)
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
                for t in sorted(load_failed, key=lambda m: m[0].fname):
                    print("loading failed", t[0].fname, "---", _print(t[0], path))
                for t in sorted(exec_failed, key=lambda m: m[0].fname):
                    print("execution failed", t[0].fname, "---", _print(t[0], path))
                for t in sorted(mismatch, key=lambda m: m[0].fname):
                    print("mismatch", t[0].fname, "---", _print(t[0], path))
                for t in sorted(missed, key=lambda m: m[0].fname):
                    print("missed ", t[0].fname, "---", _print(t[0], path))
                for t in sorted(skipped, key=lambda m: m[0].fname):
                    print("skipped", t[0].fname, "---", _print(t[0], path))

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

    @classmethod
    def setUpClass(cls, all_tests=False):
        # test not supported yet
        # not supported yet
        # see http://onnx.ai/backend-scoreboard/onnxruntime_details_stable.html
        # to compare with onnxruntime
        cls.skip_test = {
            # incomplete implementation
            "test_nesterov_momentum",
            # mismatches
            "test_center_crop_pad_crop_axes_hwc_expanded",  # shapes (10, 9, 3), (10, 8, 3) mismatch
            "test_col2im_pads",  # mismatch by one value
            "test_resize_downsample_scales_cubic_A_n0p5_exclude_outside",  # mismatch
            "test_resize_upsample_scales_cubic_A_n0p5_exclude_outside",  # mismatch
            "test_convtranspose_autopad_same",  # shapes (1, 2, 6, 6), (1, 2, 7, 7) mismatch, onnxruntime fails too
            # antialias or keep_aspect_ratio_policy not implemented
            "test_resize_downsample_scales_cubic_antialias",
            "test_resize_downsample_scales_linear_antialias",
            "test_resize_downsample_sizes_cubic_antialias",
            "test_resize_downsample_sizes_linear_antialias",
            "test_resize_downsample_sizes_nearest_not_larger",  # shape mismatch (1, 1, 1, 2), (1, 1, 1, 3)
            "test_resize_downsample_sizes_nearest_not_smaller",  # shape mismatch (1, 1, 2, 3), (1, 1, 1, 3)
            "test_resize_upsample_sizes_nearest_not_larger",
            # bug
            "test_scatter_elements_with_reduction_min",
            "test_scatter_elements_with_duplicate_indices",
            "test_stft_with_window",  # RuntimeError: DFT is not implemented when normalize is True.
            "test_stft",  # RuntimeError: DFT is not implemented when normalize is True.
            # deprecated
            "test_scan_sum",  # deprecated, opset 8 -> not implemented
            "test_scatter_with_axis",  # deprecated, scatter is removed
            "test_scatter_without_axis",  # deprecated, scatter is removed
        }
        cls.skip_test |= {
            # extended list
            # not implemented
            "test__simple_gradient_of_add",  # gradient not implemented
            "test__simple_gradient_of_add_and_mul",  # gradient not implemented
            # mismatch
            # "test__pytorch_converted_ConvTranspose2d",
            "test__pytorch_converted_ConvTranspose2d_no_bias",
            "test__pytorch_operator_operator_convtranspose",
        }
        if all_tests:
            cls.skip_test = set()
        cls.rtol = {
            "test_adam_multiple": 1e-2,
            "test_blackmanwindow_expanded": 0,
            "test_blackmanwindow_symmetric_expanded": 0,
            "test_simple_rnn_batchwise": 0,
            "test__pytorch_converted_Conv1d_pad1": 1e-4,
            "test__pytorch_converted_Conv2d": 1e-5,
            "test__pytorch_converted_Conv2d_no_bias": 1e-3,
            "test__pytorch_converted_Conv2d_strided": 1e-4,
        }
        cls.atol = {
            "test_blackmanwindow": 1e-7,
            "test_blackmanwindow_expanded": 1e-4,
            "test_blackmanwindow_symmetric": 1e-7,
            "test_blackmanwindow_symmetric_expanded": 1e-4,
            "test_gridsample_bicubic": 1e-4,
            "test_gru_seq_length": 1e-7,
            "test_hammingwindow_expanded": 1e-4,
            "test_hammingwindow_symmetric_expanded": 1e-4,
            "test_hannwindow_expanded": 1e-4,
            "test_hannwindow_symmetric": 1e-7,
            "test_hannwindow_symmetric_expanded": 1e-4,
            "test_layer_normalization_4d_axis_negative_1_expanded": 1e-6,
            "test_layer_normalization_4d_axis1_expanded": 1e-6,
            "test_layer_normalization_4d_axis_negative_3_expanded": 1e-6,
            "test_mish": 1e-6,
            "test_mish_expanded": 1e-6,
            "test_roialign_aligned_false": 1e-4,
            "test_roialign_aligned_true": 1e-4,
            # extended list
            "test__pytorch_converted_Linear_no_bias": 1e-5,
            "test_Linear_no_bias": 1e-5,
            "test__pytorch_converted_Conv1d_pad1": 1e-6,
            "test__pytorch_converted_Conv2d": 1e-5,
            "test__pytorch_converted_Conv2d_depthwise": 1e-4,
            "test__pytorch_converted_Conv2d_depthwise_strided": 1e-4,
            "test__pytorch_converted_Conv2d_depthwise_with_multiplier": 1e-4,
            "test__pytorch_converted_Conv2d_groups": 1e-4,
            "test__pytorch_converted_Conv2d_groups_thnn": 1e-4,
            "test__pytorch_converted_Conv2d_no_bias": 1e-5,
            "test__pytorch_converted_Conv2d_strided": 1e-4,
            "test__pytorch_operator_operator_symbolic_override": 1e-5,
            "test_operator_symbolic_override": 1e-4,
            # "test__pytorch_converted_Conv3d": 1e-4,
            # "test__pytorch_converted_Conv3d_dilated": 1e-4,
            "test__pytorch_converted_Conv3d_dilated_strided": 1e-4,
            "test__pytorch_converted_Conv3d_groups": 1e-4,
            # "test__pytorch_converted_Conv3d_no_bias": 1e-4,
            # "test__pytorch_converted_Conv3d_stride": 1e-4,
        }
        cls.successes = []
        cls.missed = []
        cls.skipped = []
        cls.load_failed = []
        cls.exec_failed = []
        cls.mismatch = []

    @classmethod
    def tearDownClass(cls):
        if len(cls.successes) == 0:
            raise RuntimeError("No test was successful.")
        cls._postprocess(
            cls.successes,
            cls.missed,
            cls.skipped,
            cls.load_failed,
            cls.exec_failed,
            cls.mismatch,
            10,
        )


TestOnnxBackEndWithReferenceEvaluator.add_test_methods()


if __name__ == "__main__":
    unittest.main(verbosity=2)
