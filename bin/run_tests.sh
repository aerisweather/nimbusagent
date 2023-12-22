#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Run tests using the Python interpreter from the virtual environment
venv/bin/python -m pytest tests/

# Deactivate the virtual environment
deactivate
