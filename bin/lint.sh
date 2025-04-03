#!/bin/bash
set -e

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Running linting checks...${NC}"

# Default target directory is the current directory
TARGET_DIR="${1:-.}"

echo -e "\n${YELLOW}Running Black formatter...${NC}"
if black --check "${TARGET_DIR}"; then
    echo -e "${GREEN}Black formatting check passed!${NC}"
else
    echo -e "\n${RED}Black formatting check failed!${NC}"
    echo -e "Run the following command to fix formatting issues:"
    echo -e "  black ${TARGET_DIR}"
    FAILED=true
fi

echo -e "\n${YELLOW}Running Mypy type checker...${NC}"
if mypy "${TARGET_DIR}"; then
    echo -e "${GREEN}Mypy type checking passed!${NC}"
else
    echo -e "\n${RED}Mypy type checking failed!${NC}"
    FAILED=true
fi

if [ -n "$FAILED" ]; then
    echo -e "\n${RED}Linting failed! Please fix the issues above.${NC}"
    exit 1
else
    echo -e "\n${GREEN}All linting checks passed!${NC}"
fi