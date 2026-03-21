import pytest
import json
import csv


def test_viz_imports():
    from src import viz  # noqa: F401 — just check it imports cleanly
