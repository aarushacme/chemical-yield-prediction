#!/bin/bash
# Navigate to project root directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -d "$SCRIPT_DIR/../../frontend" ]; then
    cd "$SCRIPT_DIR/../.."
elif [ -d "$SCRIPT_DIR/../frontend" ]; then
    cd "$SCRIPT_DIR/.."
else
    cd "$SCRIPT_DIR"
fi

echo "===================================================="
echo "Starting Chemical Reaction Yield Predictor..."
echo "Project Folder: $(pwd)"
echo "===================================================="
echo "Step 1: Installing required packages..."
pip install -r requirements.txt
echo ""
echo "Step 2: Launching Streamlit frontend..."
streamlit run frontend/app.py
