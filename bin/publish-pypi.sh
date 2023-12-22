#!/bin/bash

# Activate virtual environment
source ../venv/bin/activate

# Optional: Run tests again or build
./build.sh

# Push to PyPI
twine upload ../dist/*

# Deactivate the virtual environment
deactivate
