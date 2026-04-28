#!/usr/bin/env python3
"""
run.py – Standalone runner for the ABL BC Generator.

Usage
-----
    python run.py                              # uses config.yaml in the same directory
    python run.py --config my_config.yaml
    python run.py --config config.yaml --case /path/to/openfoam/case

The script reads a YAML config file, builds the ABLConfig objects, and calls
generate_inlet_data_workflow().  No package installation is required as long as
you run the script from the repository root (the abl_bc_generator package is
discovered automatically via sys.path).

Requirements
------------
    pip install pyyaml          # only extra dependency beyond requirements.txt
    pip install -r requirements.txt
"""

import argparse
import sys
import os

# Allow running from the repo root without installing the package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import yaml
except ImportError:
    sys.exit(
        "PyYAML is required.  Install it with:  pip install pyyaml"
    )

from abl_bc_generator import (
    generate_inlet_data_workflow,
    ABLConfig,
    AtmosphericConfig,
    TurbulenceConfig,
    MeshConfig,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def build_abl_config(cfg: dict) -> ABLConfig:
    atm_cfg = cfg.get("atmospheric", {})
    turb_cfg = cfg.get("turbulence", {})
    mesh_cfg = cfg.get("mesh", {})

    atmospheric = AtmosphericConfig(
        u_star=atm_cfg.get("u_star", 0.25),
        z0=atm_cfg.get("z0", 0.0),
        h_bl=atm_cfg.get("h_bl", 1500.0),
        wind_dir_met=atm_cfg.get("wind_dir_met", 225.0),
        U_ref=atm_cfg.get("U_ref"),
        z_ref=atm_cfg.get("z_ref"),
    )

    turbulence = TurbulenceConfig(
        Cmu=turb_cfg.get("Cmu", 0.033),
        kappa=turb_cfg.get("kappa", 0.40),
    )

    patch_names = mesh_cfg.get("patch_names")  # None → MeshConfig uses defaults
    mesh = MeshConfig(patch_names=patch_names)

    return ABLConfig(atmospheric=atmospheric, turbulence=turbulence, mesh=mesh)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

    parser = argparse.ArgumentParser(
        description="Generate OpenFOAM ABL boundary condition files."
    )
    parser.add_argument(
        "--config", "-c",
        default=default_config,
        help="Path to the YAML configuration file (default: config.yaml next to run.py)",
    )
    parser.add_argument(
        "--case",
        help="Path to the OpenFOAM case directory (overrides case_dir in config)",
    )
    args = parser.parse_args()

    # Load YAML
    if not os.path.isfile(args.config):
        sys.exit(f"Config file not found: {args.config}")

    cfg = load_config(args.config)

    # Resolve case directory
    case_dir = args.case or cfg.get("case_dir")
    if not case_dir:
        sys.exit(
            "No case directory specified.  "
            "Set 'case_dir' in the config file or pass --case /path/to/case."
        )

    plot_profiles = cfg.get("plot_profiles", True)
    verbose = cfg.get("verbose", False)

    abl_config = build_abl_config(cfg)

    print(f"Case directory : {case_dir}")
    print(f"Wind direction : {abl_config.atmospheric.wind_dir_met}° (met)")
    if abl_config.atmospheric.U_ref is not None:
        print(
            f"Reference wind : {abl_config.atmospheric.U_ref} m/s "
            f"at z={abl_config.atmospheric.z_ref} m  →  u* derived automatically"
        )
    else:
        print(f"u*             : {abl_config.atmospheric.u_star} m/s")
    print(f"h_bl           : {abl_config.atmospheric.h_bl} m")
    print()

    results = generate_inlet_data_workflow(
        case_dir=case_dir,
        config=abl_config,
        plot_profiles=plot_profiles,
        verbose=verbose,
    )

    print("\nDone.  Files written to", os.path.join(case_dir, "0/"))
    if plot_profiles:
        print("Profile plot   :", os.path.join(case_dir, "inlet_profiles.png"))

    return results


if __name__ == "__main__":
    main()
