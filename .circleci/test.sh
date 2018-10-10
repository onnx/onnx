#!/bin/bash

set -ex

pip install pytest nbval
pytest
