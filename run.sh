#!/bin/bash
# This script is a wrapper for the PDF RAG Pipeline CLI.
# It activates the Python virtual environment and then executes the cli.py script,
# passing along all command-line arguments.

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Path to the virtual environment
VENV_PATH="$SCRIPT_DIR/venv"

# Check if the virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at '$VENV_PATH'."
    echo "Please run the setup instructions in README.md to create it."
    exit 1
fi

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# Check if cli.py exists
CLI_SCRIPT_PATH="$SCRIPT_DIR/src/cli.py"
if [ ! -f "$CLI_SCRIPT_PATH" ]; then
    echo "Error: Main CLI script not found at '$CLI_SCRIPT_PATH'."
    exit 1
fi

# Run the Python CLI script with all arguments passed to this shell script
python "$CLI_SCRIPT_PATH" "$@"
