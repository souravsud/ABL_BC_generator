"""
Backward-compatible entry point.

The canonical package is ``abl_bc_generator``.  Install it with::

    pip install .

and use::

    from abl_bc_generator import generate_inlet_data_workflow, ABLConfig

This file is kept so that ``python generateBCs.py <case_dir>`` continues to
work when the repo is used without installation.
"""

import sys
import os

# Ensure the package directory is importable when running this file directly
# without a prior ``pip install``.
_repo_root = os.path.dirname(os.path.abspath(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from abl_bc_generator.generate import generate_inlet_data_workflow  # noqa: F401
from abl_bc_generator.config import (  # noqa: F401
    ABLConfig,
    AtmosphericConfig,
    TurbulenceConfig,
    MeshConfig,
    OpenFOAMConfig,
)

if __name__ == "__main__":
    from abl_bc_generator.generate import _cli
    _cli()
