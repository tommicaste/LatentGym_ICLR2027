#!/bin/bash
_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load .env (auto-export everything in it)
if [ -f "$_PROJECT_ROOT/.env" ]; then
  set -a; source "$_PROJECT_ROOT/.env"; set +a
else
  echo "WARN: .env not found. Set up your .env from .env.example"
fi

# Defaults (only apply if .env didn't set them)
: "${SCRATCH_DIR:="${SCRATCH:-$HOME/scratch}"}"
: "${VENV_DIR:?VENV_DIR must be set in .env}"
: "${RAY_TMPDIR:="$SCRATCH_DIR/ray"}"
: "${HF_HOME:="$SCRATCH_DIR/.cache/huggingface"}"
: "${NLTK_DATA:="$SCRATCH_DIR/.cache/nltk_data"}"

export RAY_TMPDIR HF_HOME NLTK_DATA
export PYTHONPATH="${_PROJECT_ROOT}:${_PROJECT_ROOT}/TextArena:${PYTHONPATH:-}"
