"""
Command-line entry point for the ``abl_bc_generator`` package.

Enables both of the following invocations after ``pip install .``::

    abl-bc-generator <case_dir> [options]   # via console_scripts entry point
    python -m abl_bc_generator <case_dir> [options]   # direct module execution
"""

import argparse

from .config import ABLConfig
from .generate import generate_inlet_data_workflow


def main():
    parser = argparse.ArgumentParser(
        prog="abl-bc-generator",
        description="Generate OpenFOAM ABL boundary conditions for a case directory.",
    )
    parser.add_argument("case_dir", help="Path to the OpenFOAM case directory")
    parser.add_argument(
        "--plot",
        dest="plot_profiles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate inlet profile plots (default: enabled). Use --no-plot to disable.",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable verbose logging (default: disabled). Use --verbose to enable.",
    )
    args = parser.parse_args()

    config = ABLConfig()
    generate_inlet_data_workflow(
        args.case_dir,
        config,
        plot_profiles=args.plot_profiles,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
