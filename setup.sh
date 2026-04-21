#!/usr/bin/env bash

echo "==> Upgrading pip, setuptools, wheel"
pip install --upgrade pip setuptools wheel

echo "==> Installing project dependencies"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
elif [ -f "pyproject.toml" ]; then
    pip install -e .
else
    echo "  [!] No requirements.txt or pyproject.toml found. Skipping."
fi

echo "==> Verifying key binaries"
echo "  python  -> $(which python)"
echo "  pip     -> $(which pip)"
echo "  torchrun -> $(which torchrun)"

echo "==> Done. Environment is ready."