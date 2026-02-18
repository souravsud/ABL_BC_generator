#!/usr/bin/env python3
"""
Standalone script to validate ABL inlet profiles without running OpenFOAM.

This script reads the generated inlet profile files and performs comprehensive
validation checks including:
- Mathematical correctness of velocity, TKE, and epsilon profiles
- Verification of ground elevation usage
- Flow direction and perpendicularity checks
- Comparison with theoretical ABL profiles

Usage:
    python run_validation.py <case_directory> [options]

Example:
    python run_validation.py /path/to/case --config config.json
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from config import ABLConfig
from generateBCs import read_inlet_face_file, calculate_multiregion_z_distribution
from validate_profiles import run_comprehensive_validation


def read_generated_profiles(case_dir: str):
    """
    Read the generated inlet profile files.
    
    Args:
        case_dir: Case directory containing the generated files
        
    Returns:
        Tuple of (U_profiles, k_profiles, epsilon_profiles)
    """
    include_dir = Path(case_dir) / '0' / 'include'
    
    # Read U profiles
    with open(include_dir / 'inletU', 'r') as f:
        lines = f.readlines()
        n_profiles = int(lines[0].strip())
        U_profiles = []
        skipped_u = 0
        for line_num, line in enumerate(lines[2:n_profiles+2], start=2):  # Skip header lines
            line = line.strip().strip('()').split()
            if len(line) == 3:
                U_profiles.append([float(x) for x in line])
            else:
                skipped_u += 1
                print(f"Warning: Skipped malformed U profile at line {line_num}: '{line}'")
        U_profiles = np.array(U_profiles)
        if skipped_u > 0:
            print(f"Warning: Skipped {skipped_u} malformed U profile lines")
    
    # Read k profiles
    with open(include_dir / 'inletK', 'r') as f:
        lines = f.readlines()
        n_profiles = int(lines[0].strip())
        k_profiles = []
        skipped_k = 0
        for line_num, line in enumerate(lines[2:n_profiles+2], start=2):
            try:
                k_profiles.append(float(line.strip()))
            except ValueError:
                skipped_k += 1
                print(f"Warning: Skipped malformed k profile at line {line_num}: '{line.strip()}'")
        k_profiles = np.array(k_profiles)
        if skipped_k > 0:
            print(f"Warning: Skipped {skipped_k} malformed k profile lines")
    
    # Read epsilon profiles
    with open(include_dir / 'inletEpsilon', 'r') as f:
        lines = f.readlines()
        n_profiles = int(lines[0].strip())
        epsilon_profiles = []
        skipped_eps = 0
        for line_num, line in enumerate(lines[2:n_profiles+2], start=2):
            try:
                epsilon_profiles.append(float(line.strip()))
            except ValueError:
                skipped_eps += 1
                print(f"Warning: Skipped malformed epsilon profile at line {line_num}: '{line.strip()}'")
        epsilon_profiles = np.array(epsilon_profiles)
        if skipped_eps > 0:
            print(f"Warning: Skipped {skipped_eps} malformed epsilon profile lines")
    
    return U_profiles, k_profiles, epsilon_profiles


def main():
    """Main validation script."""
    parser = argparse.ArgumentParser(
        description='Validate ABL inlet profiles without running OpenFOAM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate with default configuration
  python run_validation.py /path/to/case

  # Specify custom parameters
  python run_validation.py /path/to/case --u-star 0.3 --z0 0.1 --h-bl 1500

  # Disable plots (text report only)
  python run_validation.py /path/to/case --no-plots
        """
    )
    
    parser.add_argument('case_dir', type=str,
                       help='Case directory containing inlet profile files')
    parser.add_argument('--u-star', type=float, default=0.25,
                       help='Friction velocity (m/s), default: 0.25')
    parser.add_argument('--z0', type=float, default=0.0,
                       help='Surface roughness (m), 0 = use from file, default: 0.0')
    parser.add_argument('--h-bl', type=float, default=1500.0,
                       help='Boundary layer height (m), default: 1500')
    parser.add_argument('--flow-dir', type=float, default=45.0,
                       help='Flow direction (degrees), default: 45')
    parser.add_argument('--kappa', type=float, default=0.40,
                       help='Von Karman constant, default: 0.40')
    parser.add_argument('--cmu', type=float, default=0.033,
                       help='Turbulence constant, default: 0.033')
    parser.add_argument('--no-plots', action='store_true',
                       help='Disable plot generation (text report only)')
    parser.add_argument('--use-internal-faces', action='store_true',
                       help='Use internal faces instead of cell centers')
    
    args = parser.parse_args()
    
    # Validate case directory
    case_dir = Path(args.case_dir)
    if not case_dir.exists():
        print(f"Error: Case directory '{case_dir}' does not exist")
        sys.exit(1)
    
    inlet_file = case_dir / '0' / 'include' / 'inletFaceInfo.txt'
    if not inlet_file.exists():
        print(f"Error: Inlet face file not found at '{inlet_file}'")
        print("This file should be generated by the terrain_following_mesh_generator")
        sys.exit(1)
    
    # Check if profile files exist
    include_dir = case_dir / '0' / 'include'
    required_files = ['inletU', 'inletK', 'inletEpsilon']
    missing_files = [f for f in required_files if not (include_dir / f).exists()]
    
    if missing_files:
        print(f"Error: The following profile files are missing: {missing_files}")
        print(f"Expected location: {include_dir}")
        print("\nPlease run the profile generation first using generateBCs.py")
        sys.exit(1)
    
    print("=" * 80)
    print("ABL INLET PROFILE VALIDATION")
    print("=" * 80)
    print(f"\nCase directory: {case_dir}")
    print(f"Validation mode: {'Text report only' if args.no_plots else 'Full validation with plots'}")
    print()
    
    # Create configuration
    from config import AtmosphericConfig, TurbulenceConfig, MeshConfig, OpenFOAMConfig
    
    config = ABLConfig(
        atmospheric=AtmosphericConfig(
            u_star=args.u_star,
            z0=args.z0,
            h_bl=args.h_bl,
            flow_dir_deg=args.flow_dir
        ),
        turbulence=TurbulenceConfig(
            Cmu=args.cmu,
            kappa=args.kappa
        ),
        mesh=MeshConfig(),
        openfoam=OpenFOAMConfig()
    )
    
    print("Configuration:")
    print(f"  u* = {config.atmospheric.u_star} m/s")
    print(f"  z0 = {config.atmospheric.z0} m {'(will use from file)' if config.atmospheric.z0 == 0 else ''}")
    print(f"  h_bl = {config.atmospheric.h_bl} m")
    print(f"  Flow direction = {config.atmospheric.flow_dir_deg}°")
    print(f"  kappa = {config.turbulence.kappa}")
    print(f"  Cmu = {config.turbulence.Cmu}")
    print()
    
    # Read inlet data
    print("Reading inlet face data...")
    inlet_data = read_inlet_face_file(str(inlet_file))
    inlet_blocks, mesh_params = inlet_data
    
    # Calculate z-coordinates
    use_face_centers = not args.use_internal_faces
    z_coords = calculate_multiregion_z_distribution(
        mesh_params['avg_inlet_height'],
        mesh_params['domain_height'],
        mesh_params,
        use_face_centers
    )
    
    print(f"Profile points: {len(z_coords)}")
    print(f"Height range: {np.min(z_coords):.2f} to {np.max(z_coords):.2f} m")
    print()
    
    # Read generated profiles
    print("Reading generated profile files...")
    U_profiles, k_profiles, epsilon_profiles = read_generated_profiles(str(case_dir))
    
    print(f"  U profiles: {U_profiles.shape}")
    print(f"  k profiles: {k_profiles.shape}")
    print(f"  epsilon profiles: {epsilon_profiles.shape}")
    print()
    
    # Run validation
    try:
        validation_results = run_comprehensive_validation(
            str(case_dir),
            config,
            inlet_data,
            U_profiles,
            k_profiles,
            epsilon_profiles,
            z_coords,
            generate_plots=not args.no_plots
        )
        
        print("\n" + "=" * 80)
        print("VALIDATION COMPLETE")
        print("=" * 80)
        print(f"\nResults saved to:")
        print(f"  - Text report: {case_dir / 'validation_report.txt'}")
        print(f"  - JSON data: {case_dir / 'validation_results.json'}")
        if not args.no_plots:
            print(f"  - Plots: {case_dir / 'validation_plots.png'}")
        
        # Determine exit code based on validation results
        all_pass = all([
            validation_results.get('ground_elevation', {}).get('uses_ground_elevation', False),
            validation_results.get('flow_direction', {}).get('direction_correct', False),
            validation_results.get('flow_direction', {}).get('is_horizontal', False),
            validation_results.get('velocity_profile', {}).get('passes', False),
            validation_results.get('tke_profile', {}).get('passes', False),
            validation_results.get('epsilon_profile', {}).get('passes', False)
        ])
        
        if all_pass:
            print("\n✓ All validation checks PASSED")
            return 0
        else:
            print("\n✗ Some validation checks FAILED - please review the report")
            return 1
            
    except Exception as e:
        print(f"\nError during validation: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == '__main__':
    sys.exit(main())
