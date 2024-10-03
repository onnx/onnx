:: SPDX-License-Identifier: Apache-2.0

@echo off
python -u "%~dp0\protoc-gen-mypy.py" || py -u "%~dp0\protoc-gen-mypy.py"
