"""
Unit and regression test for the catmof package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import catmof


def test_catmof_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "catmof" in sys.modules
