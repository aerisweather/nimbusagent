#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Run tests first
venv/bin/python -m pytest tests/
if [ $? -ne 0 ]; then
  echo "Tests failed, aborting build."
  exit 1
fi

# Build the package
python setup.py sdist bdist_wheel

# Deactivate the virtual environment
deactivate
