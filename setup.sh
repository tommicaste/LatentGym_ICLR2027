#!/bin/bash

## Set executation path to file location
cd "$(dirname "$0")"


echo "Loading VENV_DIR from project_config.sh"
source ./project_config.sh


if [ -z "$VENV_DIR" ]; then
  echo "ERROR: VENV_DIR is not set in project_config.sh."
  exit 1
fi

## Create virtual environment
uv venv "$VENV_DIR" --python 3.12.7
source "$VENV_DIR/bin/activate"

cd skyrl-train

source "$VENV_DIR/bin/activate"
uv sync --active --extra vllm

echo "Setup complete. To activate the virtual environment, run 'source $VENV_DIR/bin/activate'."

