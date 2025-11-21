#!/usr/bin/env bash
set -euo pipefail

# Repo setup script: ensure Python 3.12 and create/activate a venv.
# Usage:
#   ./setup.sh        # creates venv and prints activation instructions
#   source ./setup.sh  # creates venv and activates it in your shell

# detect whether script is being sourced
sourced=0
if [ "${BASH_SOURCE[0]}" != "$0" ]; then sourced=1; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"

have_cmd() { command -v "$1" >/dev/null 2>&1; }

ensure_python() {
    # prefer explicit python3.12
    if have_cmd python3.12; then
        PYBIN="$(command -v python3.12)"
        return 0
    fi

    # accept system python3 if it's 3.12
    if have_cmd python3; then
        if python3 - <<'PY' 2>/dev/null
import sys
sys.exit(0 if sys.version_info[:2]==(3,12) else 1)
PY
        then
            PYBIN="$(command -v python3)"
            return 0
        fi
    fi

    OS="$(uname -s)"
    if [ "$OS" = "Linux" ]; then
        if have_cmd apt-get; then
            echo "Attempting to install Python 3.12 via apt..."
            sudo apt-get update
            sudo apt-get install -y software-properties-common || true
            # add deadsnakes for older Ubuntu if available
            if have_cmd add-apt-repository; then
                sudo add-apt-repository -y ppa:deadsnakes/ppa || true
                sudo apt-get update
            fi
            sudo apt-get install -y python3.12 python3.12-venv python3.12-distutils || true
            if have_cmd python3.12; then PYBIN="$(command -v python3.12)"; return 0; fi
        elif have_cmd dnf; then
            echo "Attempting to install Python 3.12 via dnf..."
            sudo dnf install -y python3.12 python3.12-venv || true
            if have_cmd python3.12; then PYBIN="$(command -v python3.12)"; return 0; fi
        elif have_cmd yum; then
            echo "Attempting to install Python 3.12 via yum..."
            sudo yum install -y python3.12 python3.12-venv || true
            if have_cmd python3.12; then PYBIN="$(command -v python3.12)"; return 0; fi
        fi
    elif [ "$OS" = "Darwin" ]; then
        if have_cmd brew; then
            echo "Attempting to install Python 3.12 via Homebrew..."
            brew install python@3.12 || true
            if have_cmd python3.12; then PYBIN="$(command -v python3.12)"; return 0; fi
            prefix="$(brew --prefix python@3.12 2>/dev/null || true)"
            if [ -n "$prefix" ] && [ -x "$prefix/bin/python3.12" ]; then
                PYBIN="$prefix/bin/python3.12"
                return 0
            fi
        fi
    fi

    return 1
}

if ! ensure_python; then
    echo "ERROR: Python 3.12 not found and automatic installation failed."
    echo "Please install Python 3.12 (and the venv module) manually and re-run this script."
    exit 2
fi

echo "Using Python: $("$PYBIN" --version 2>&1)"

# create virtualenv if missing
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at: $VENV_DIR"
else
    echo "Creating virtual environment at: $VENV_DIR"
    "$PYBIN" -m venv "$VENV_DIR"
fi

# make sure pip/tools are up-to-date
if [ -x "$VENV_DIR/bin/python" ]; then
    "$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel >/dev/null 2>&1 || true
fi

if [ "$sourced" -eq 1 ]; then
    # shellcheck disable=SC1091
    . "$VENV_DIR/bin/activate"
    echo "Virtual environment activated."
else
    echo ""
    echo "To activate the virtual environment in your shell, run:"
    echo "  source \"$VENV_DIR/bin/activate\""
    echo ""
    echo "Or to create + activate in one step, source this script instead of executing it:"
    echo "  source \"${BASH_SOURCE[0]}\""
fi