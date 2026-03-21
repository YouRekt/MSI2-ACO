#!/usr/bin/env bash
set -e
uv run python experiments/runner.py "$@"
