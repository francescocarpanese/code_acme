#!/bin/bash

# Bash settings: fail on any error and display all commands being run.
set -e
set -x


# Update
apt-get update

# Install dependencies.
pip install --upgrade pip setuptools
pip --version

# Set up a virtual environment.
pip install virtualenv
virtualenv code_acme_testing
source code_acme_testing/bin/activate


# Install depedencies
pip install .[dev]

N_CPU=$(grep -c ^processor /proc/cpuinfo)

# Run all unit tests (non integration tests).
pytest --durations=10 -n "${N_CPU}"

# Clean-up.
deactivate
rm -rf code_acme_testing/