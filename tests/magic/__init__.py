"""
Benson test magic module.

This module provides test cases for Benson's magic methods,
including ECT and other topological and geometric representation learning techniques.
"""

from tests.magic.test_base import TestMagic
from tests.magic.test_ect import TestECT

__all__ = ["TestMagic", "TestECT"]
