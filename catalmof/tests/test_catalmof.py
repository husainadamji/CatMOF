"""
Unit and regression test for the catalmof package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import catalmof


def test_catalmof_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "catalmof" in sys.modules
