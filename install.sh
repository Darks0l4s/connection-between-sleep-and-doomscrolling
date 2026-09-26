#!/bin/bash

VENV_DIR="SleepVenv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Setup complete. Activate with: source $VENV_DIR/bin/activate"