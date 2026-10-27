# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Compare DequantizeGGUF with pinned upstream C, without adding a runtime dependency."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import logging
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from onnx.reference.ops._gguf_codebooks import CODEBOOKS
from onnx.reference.ops.op_dequantize_gguf import GGUF_BLOCK_SIZES, _decode_block

logger = logging.getLogger(__name__)

_SOURCE_HASHES = {
    "ggml-common.h": "0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c",
    "ggml-quants.c": "5574a2dccf7c07e75b143733e04a5412d3d8c819e7945f5217a7b83a2b2ff8ab",
    "ggml-impl.h": "43564db0238aebb7ed68501e346c194866b5dac218d1d37b26baff9f458c00d3",
}
_NAMES = {
    2: "q4_0",
    3: "q4_1",
    6: "q5_0",
    7: "q5_1",
    8: "q8_0",
    9: "q8_1",
    10: "q2_K",
    11: "q3_K",
    12: "q4_K",
    13: "q5_K",
    14: "q6_K",
    15: "q8_K",
    16: "iq2_xxs",
    17: "iq2_xs",
    18: "iq3_xxs",
    19: "iq1_s",
    20: "iq4_nl",
    21: "iq3_s",
    22: "iq2_s",
    23: "iq4_xs",
    29: "iq1_m",
    34: "tq1_0",
    35: "tq2_0",
    39: "mxfp4",
    40: "nvfp4",
    41: "q1_0",
    42: "q2_0",
}
_PREAMBLE = r"""
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#define GGML_RESTRICT
#define GGML_FP16_TO_FP32 half_to_float
#define GGML_E8M0_TO_FP32_HALF ggml_e8m0_to_fp32_half
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif
static float half_to_float(uint16_t h) {
    int exp = (h >> 10) & 31, man = h & 1023;
    float v = exp == 0 ? ldexpf((float)man, -24)
        : exp == 31 ? (man ? NAN : INFINITY)
        : ldexpf(1.0f + man/1024.0f, exp-15);
    return h & 0x8000 ? -v : v;
}
"""
_Q8_1 = """
EXPORT void dequantize_row_q8_1(const block_q8_1 * x, float * y, int64_t k) {
    for (int64_t j = 0; j < k; ++j)
        y[j] = half_to_float(x[j/32].d) * x[j/32].qs[j%32];
}
"""


def _function(source: str, name: str) -> str:
    match = re.search(
        rf"^(?:static inline |static )?(?:void|float) {name}\(", source, re.MULTILINE
    )
    if match is None:
        raise ValueError(f"Missing upstream function {name}")
    start = source.index("{", match.start())
    depth, end = 1, start + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start() : end]


def _free_library(handle: int) -> None:
    if sys.platform == "win32":
        free_library = ctypes.WinDLL("kernel32", use_last_error=True).FreeLibrary
        free_library.argtypes = [ctypes.c_void_p]
        free_library.restype = ctypes.c_int
        if not free_library(handle):
            raise ctypes.WinError(ctypes.get_last_error())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Directory containing ggml-common.h, ggml-quants.c, and ggml-impl.h "
        "from GGML da8e2a54b919557b816b1c2beb432ccb766f7ce0",
    )
    args = parser.parse_args()
    sources = {}
    for name, digest in _SOURCE_HASHES.items():
        data = (args.source_dir / name).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"{name} does not match the pinned upstream revision")
        sources[name] = data.decode("utf-8")
    test_path = (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "python"
        / "dequantize_gguf_test.py"
    )
    spec = importlib.util.spec_from_file_location("gguf_tests", test_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load test corpus from {test_path}")
    tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tests)
    source = _PREAMBLE
    impl = sources["ggml-impl.h"]
    source += _function(impl, "ggml_e8m0_to_fp32_half") + "\n"
    source += _function(impl, "ggml_ue4m3_to_fp32") + "\n"  # typos: ignore
    source += _function(sources["ggml-quants.c"], "get_scale_min_k4") + "\n"
    for name in re.findall(
        r"^void (dequantize_row_\w+)\(", sources["ggml-quants.c"], re.MULTILINE
    ):
        source += "EXPORT " + _function(sources["ggml-quants.c"], name) + "\n"
    source += _Q8_1
    cmake = shutil.which("cmake")
    if cmake is None:
        raise RuntimeError("CMake is required to build the upstream oracle.")
    with (
        tempfile.TemporaryDirectory(prefix="onnx-gguf-oracle-") as directory,
        ExitStack() as cleanup,
    ):
        root = Path(directory)
        (root / "oracle.c").write_text(source, encoding="utf-8")
        (root / "ggml-common.h").write_text(sources["ggml-common.h"], encoding="utf-8")
        (root / "CMakeLists.txt").write_text(
            """cmake_minimum_required(VERSION 3.26)
project(gguf_oracle C)
add_library(gguf_oracle SHARED oracle.c)
if(MSVC)
  target_compile_options(gguf_oracle PRIVATE /O2 /fp:strict)
else()
  target_compile_options(gguf_oracle PRIVATE -O2 -ffp-contract=off)
  target_link_libraries(gguf_oracle PRIVATE m)
endif()
""",
            encoding="utf-8",
        )
        build = root / "build"
        subprocess.run([cmake, "-S", str(root), "-B", str(build)], check=True)  # noqa: S603
        subprocess.run(  # noqa: S603
            [cmake, "--build", str(build), "--config", "Release"], check=True
        )
        filename = (
            "gguf_oracle.dll"
            if sys.platform == "win32"
            else (
                "libgguf_oracle.dylib"
                if sys.platform == "darwin"
                else "libgguf_oracle.so"
            )
        )
        libraries = list(build.rglob(filename))
        if len(libraries) != 1:
            raise RuntimeError(f"Expected one oracle library, found {libraries}")
        library = ctypes.CDLL(str(libraries[0]))
        cleanup.callback(_free_library, library._handle)
        for qtype, (width, _) in GGUF_BLOCK_SIZES.items():
            blocks = tests._corpus(qtype)
            expected = np.empty((len(blocks), width), dtype=np.float32)
            decode = getattr(library, "dequantize_row_" + _NAMES[qtype])
            decode.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
            decode.restype = None
            decode(blocks.ctypes.data, expected.ctypes.data, expected.size)
            with np.errstate(over="ignore", invalid="ignore"):
                actual = np.stack(
                    [_decode_block(block.tobytes(), qtype) for block in blocks]
                )
            np.testing.assert_array_equal(
                actual.view(np.uint32), expected.view(np.uint32)
            )
            digest = hashlib.sha256(expected.astype("<f4").tobytes()).hexdigest()
            if digest != tests._ORACLE_DIGESTS[qtype]:
                raise AssertionError(
                    f"Stored oracle digest disagrees with upstream for type {qtype}"
                )
            logger.info("GGML type %s: upstream C comparison and digest passed", qtype)
    for name, (width, _, _, _) in CODEBOOKS.items():
        match = re.search(
            rf"GGML_TABLE_BEGIN\(\w+, {name}, \w+\)(.*?)GGML_TABLE_END\(\)",
            sources["ggml-common.h"],
            re.DOTALL,
        )
        if match is None:
            raise ValueError(f"Missing upstream codebook {name}")
        data = b"".join(
            int(word, 16).to_bytes(width, "little")
            for word in re.findall(r"0x[0-9a-fA-F]+", match[1])
        )
        if hashlib.sha256(data).hexdigest() != tests._CODEBOOK_DIGESTS[name]:
            raise AssertionError(f"Codebook digest disagrees with upstream for {name}")
    logger.info("All 27 quantized formats and six complete codebooks passed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
