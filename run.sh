#!/bin/bash
# This script is a wrapper for the PDF RAG Pipeline Developer Toolkit (cli.py).
# It activates the Python virtual environment and then executes the cli.py script,
# passing along all command-line arguments.
#
# For production data extraction, use:
#   python src/run_extraction.py --issue-id <issue_id> [--extraction-field <field>]

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
    echo "Error: Developer toolkit script not found at '$CLI_SCRIPT_PATH'."
    exit 1
fi

# Main script logic
case "$1" in
  index|query|clear|extract-html-text)
    python3 src/cli.py "$@"
    ;;
  *)
    echo "Usage: $0 {index|query|clear|extract-html-text}"
    echo ""
    echo "For production data extraction, use:"
    echo "  python src/run_extraction.py --issue-id <issue_id> [--extraction-field <field>]"
    exit 1
    ;;
esac
