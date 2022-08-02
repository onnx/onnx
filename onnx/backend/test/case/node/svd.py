# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np  # type: ignore

import onnx

from ..base import Base
from . import expect


class Svd(Base):
    @staticmethod
    def export() -> None:
        dimensions = [((3, 4), "2d"), ((2, 3, 4), "3d"), ((5, 4, 3, 2, 1, 3, 4), "7d")]

        for compute_uv in [0, 1, None]:
            for full_matrices in [0, 1, None]:
                for (dims, dims_name) in dimensions:
                    A = np.random.randn(*dims)

                    node_args: Any = {"inputs": ["A"]}
                    numpy_args = {}
                    name = f"test_svd_{dims_name}"

                    if full_matrices is not None:
                        node_args["full_matrices"] = full_matrices
                        numpy_args["full_matrices"] = bool(full_matrices)
                        if full_matrices:
                            name += "_full_manually_set"
                        else:
                            name += "_partial"

                    if compute_uv is not None:
                        node_args["compute_uv"] = compute_uv
                        numpy_args["compute_uv"] = bool(compute_uv)
                        if compute_uv:
                            name += "_compute_uv_manually_set"
                        else:
                            name += "_compute_only_singular_vals"

                    if compute_uv == 0:
                        S = np.linalg.svd(A, **numpy_args)
                        outputs = [S]
                        node_args["outputs"] = ["S"]
                    else:
                        U, S, Vh = np.linalg.svd(A, **numpy_args)
                        # Flip output ordering to account for ONNX putting S first
                        outputs = [S, U, Vh]
                        node_args["outputs"] = ["S", "U", "Vh"]

                    node = onnx.helper.make_node("SVD", **node_args)
                    expect(node, inputs=[A], outputs=outputs, name=name)
