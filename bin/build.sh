#!/bin/bash

# Activate virtual environment
source ../venv/bin/activate

# Run tests first
python -m pytest ../tests/
if [ $? -ne 0 ]; then
  echo "Tests failed, aborting build."
  exit 1
fi

# Clean out the dist directory
echo "Cleaning out the dist directory..."
rm -rf ../dist/*

# Build the package
python -m build ../

# Deactivate the virtual environment
../venv/bin/deactivate
