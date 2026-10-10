# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from onnx.backend.base import Device, DeviceType


class TestDevice:
    def test_parses_device_type_only(self) -> None:
        device = Device("CPU")
        assert device.type == DeviceType.CPU
        assert device.device_id == 0

    def test_parses_device_type_and_id(self) -> None:
        device = Device("CUDA:1")
        assert device.type == DeviceType.CUDA
        assert device.device_id == 1

    def test_unsupported_device_type_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unsupported device type 'GPU'"):
            Device("GPU")

    def test_empty_device_string_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unsupported device type ''"):
            Device("")

    def test_non_integer_device_id_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid device id 'abc'"):
            Device("CUDA:abc")
