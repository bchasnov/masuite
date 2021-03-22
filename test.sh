#!/bin/bash

# Fail on any error.
set -e
# Display commands being run.
set -x

# Set up a new virtual environment.
python3 -m venv masuite_testing
source masuite_testing/bin/activate

# Install all dependencies.
pip install --upgrade pip setuptools
pip install .
#pip install .[baselines_jax]
#pip install .[baselines]

# Install test dependencies.
#pip install .[testing]

N_CPU=$(grep -c ^processor /proc/cpuinfo)

# Run static type-checking.
#pytype -j "${N_CPU}" bsuite

# Run all tests.
pytest -n "${N_CPU}" masuite

# Clean-up.
deactivate
rm -rf bsuite_testing/
