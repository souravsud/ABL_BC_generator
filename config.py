"""
Backward-compatible entry point for config classes.

The canonical package is ``abl_bc_generator``.  Install it with::

    pip install .

and use::

    from abl_bc_generator import ABLConfig, AtmosphericConfig, TurbulenceConfig, MeshConfig
"""

import sys
import os

_repo_root = os.path.dirname(os.path.abspath(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from abl_bc_generator.config import (  # noqa: F401
    ABLConfig,
    AtmosphericConfig,
    TurbulenceConfig,
    MeshConfig,
    OpenFOAMConfig,
)
