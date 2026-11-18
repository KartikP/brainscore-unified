"""Benchmark plugins keep their tests in their own folders (``test.py``, ``test_<topic>.py``).

The same plugins as ``tests/conftest.py``, so that a test here has the same tier markers and the
CPU fixture when it runs with the suite or alone from the checkout.
"""
pytest_plugins = ['brainscore.validation.cpu_profile', 'tests.markers']
