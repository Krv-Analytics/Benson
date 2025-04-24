"""
Benson magic module.

This module provides various methods for generating low-dimensional descriptors
of point clouds, with a focus on topological and geometric representation learning techniques.
"""

from benson.magic.base import Magic
from benson.magic.ect import ECT
from benson.magic.config import ECTConfig

__all__ = ["Magic", "ECT", "ECTConfig"]
