"""Deterministic disk capacity for synthetic trace-writer fixtures only.

Private test tmpfs capacity must not decide whether a fixture can be recorded.
The disk-guard suite overrides this mock for low-space and failed-check cases.
Production defaults and actual filesystem writes are unchanged.
"""
from unittest.mock import patch


_capacity = None


def setUpModule():
    global _capacity
    _capacity = patch("src.engine.physical_trace._free_gb", return_value=100.0)
    _capacity.start()


def tearDownModule():
    global _capacity
    _capacity.stop()
    _capacity = None
