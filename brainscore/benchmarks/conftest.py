"""Benchmark plugins keep their tests in their own folders (``test.py``, ``test_<topic>.py``).

Use the marking of ``tests/conftest.py``, so that a test here has the same tier
markers when it runs with the suite or on its own.
"""
from tests.conftest import pytest_collection_modifyitems  # noqa: F401
