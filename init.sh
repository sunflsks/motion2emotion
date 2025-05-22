#!/bin/sh

export TMPDIR=/dev/shm

curl -LsSf https://astral.sh/uv/install.sh | sh
~/.local/bin/uv run main.py
