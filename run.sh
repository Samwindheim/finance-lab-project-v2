#!/bin/bash
# This script is a unified wrapper for the Finance Lab Pipeline (main.py).
# It handles both production extractions and developer utility commands.
#
# Usage:
#   ./run.sh extract <link> [--extraction-field <field>]
#   ./run.sh index <pdf_path>
#   ./run.sh query <pdf_path> <query>

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

# Check if main.py exists
MAIN_SCRIPT_PATH="$SCRIPT_DIR/src/main.py"
if [ ! -f "$MAIN_SCRIPT_PATH" ]; then
    echo "Error: Unified script not found at '$MAIN_SCRIPT_PATH'."
    exit 1
fi

# Main script logic
case "$1" in
  extract|index|query|clear|html-text)
    python3 src/main.py "$@"
    ;;
  *)
    # Default to 'extract' if the first argument looks like a link or starts with --issue-id
    # This maintains backward compatibility with the old direct run_extraction.py calls
    if [[ "$1" == http* ]] || [[ "$1" == --issue-id* ]]; then
      python3 src/main.py extract "$@"
    else
      echo "Usage: $0 {extract|index|query|clear|html-text}"
      echo ""
      echo "Example:"
      echo "  $0 extract https://example.com/doc.pdf"
      echo "  $0 index pdfs/document.pdf"
      exit 1
    fi
    ;;
esac
