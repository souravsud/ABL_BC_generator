"""
Validation and inspection tools for ABL inlet profiles.

This module provides comprehensive validation and visualization capabilities
to verify the correctness of generated velocity profiles without running
OpenFOAM simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, Tuple, Optional
import json


def validate_log_law_profile(z_coords: np.ndarray, u_mag: np.ndarray, 
                             u_star: float, z0: float, kappa: float,
                             z_ground: float = 0.0) -> Dict:
    """
    Validate that velocity profile follows logarithmic law in surface layer.
    
    Args:
        z_coords: Height coordinates above ground
        u_mag: Velocity magnitude at each height
        u_star: Friction velocity
        z0: Surface roughness length
        kappa: Von Karman constant
        z_ground: Ground elevation
        
    Returns:
        Dictionary with validation results
    """
    # Calculate theoretical profile
    heights = np.maximum(z_coords - z_ground, 0.01)
    u_theoretical = (u_star / kappa) * np.log(1.0 + heights / z0)
    
    # Calculate relative errors
    relative_errors = np.abs(u_mag - u_theoretical) / (u_theoretical + 1e-10)
    
    # Statistics
    max_error = np.max(relative_errors) * 100
    mean_error = np.mean(relative_errors) * 100
    rmse = np.sqrt(np.mean((u_mag - u_theoretical)**2))
    
    return {
        'u_theoretical': u_theoretical,
        'relative_errors': relative_errors,
        'max_error_percent': max_error,
        'mean_error_percent': mean_error,
        'rmse': rmse,
        'passes': max_error < 0.1  # Less than 0.1% error
    }


def validate_ground_elevation_usage(inlet_blocks: list, z_coords: np.ndarray,
                                    U_profiles: np.ndarray) -> Dict:
    """
    Verify that profiles correctly use ground elevations at inlet.
    
    Args:
        inlet_blocks: List of inlet block information with ground elevations
        z_coords: Height coordinates used in profiles
        U_profiles: Velocity profiles array
        
    Returns:
        Dictionary with ground elevation validation results
    """
    # Extract ground elevations from blocks
    z_grounds = np.array([block['z_ground'] for block in inlet_blocks])
    
    # Calculate statistics
    z_ground_min = np.min(z_grounds)
    z_ground_max = np.max(z_grounds)
    z_ground_mean = np.mean(z_grounds)
    z_ground_std = np.std(z_grounds)
    
    # Check if z_coords start from reasonable ground level
    z_min_used = np.min(z_coords)
    
    # Verify ground elevation is being used
    uses_ground_elevation = np.isclose(z_min_used, z_ground_mean, rtol=0.1)
    
    return {
        'z_ground_min': z_ground_min,
        'z_ground_max': z_ground_max,
        'z_ground_mean': z_ground_mean,
        'z_ground_std': z_ground_std,
        'z_ground_range': z_ground_max - z_ground_min,
        'z_min_used_in_profiles': z_min_used,
        'uses_ground_elevation': uses_ground_elevation,
        'terrain_variation': z_ground_std
    }


def validate_flow_direction(U_profiles: np.ndarray, flow_dir_deg: float,
                            tolerance_deg: float = 0.1) -> Dict:
    """
    Verify that flow is in the correct direction and perpendicular to inlet.
    
    Args:
        U_profiles: Velocity profiles [n_faces, 3] with (u, v, w) components
        flow_dir_deg: Expected flow direction in degrees from x-axis
        tolerance_deg: Tolerance for direction check in degrees
        
    Returns:
        Dictionary with flow direction validation results
    """
    # Calculate actual flow directions for each profile
    flow_dir_rad = np.radians(flow_dir_deg)
    expected_u = np.cos(flow_dir_rad)
    expected_v = np.sin(flow_dir_rad)
    
    # Get horizontal components (excluding vertical)
    u_horiz = U_profiles[:, 0]
    v_horiz = U_profiles[:, 1]
    w_vert = U_profiles[:, 2]
    
    # Calculate magnitude of horizontal flow
    u_mag_horiz = np.sqrt(u_horiz**2 + v_horiz**2)
    
    # Calculate actual direction for each profile
    actual_dirs_rad = np.arctan2(v_horiz, u_horiz)
    actual_dirs_deg = np.degrees(actual_dirs_rad)
    
    # Normalize to [0, 360]
    actual_dirs_deg = np.mod(actual_dirs_deg, 360)
    expected_dir_normalized = np.mod(flow_dir_deg, 360)
    
    # Direction errors with proper wraparound handling
    dir_errors_deg = np.abs(actual_dirs_deg - expected_dir_normalized)
    # Handle wraparound at 360 degrees
    dir_errors_deg = np.minimum(dir_errors_deg, 360 - dir_errors_deg)
    
    # Check vertical component (should be near zero)
    w_magnitude = np.abs(w_vert)
    max_w = np.max(w_magnitude)
    mean_w = np.mean(w_magnitude)
    
    # Normalized vertical component
    w_normalized = w_magnitude / (u_mag_horiz + 1e-10)
    max_w_normalized = np.max(w_normalized)
    
    return {
        'expected_direction_deg': flow_dir_deg,
        'mean_direction_deg': np.mean(actual_dirs_deg),
        'direction_std_deg': np.std(actual_dirs_deg),
        'max_direction_error_deg': np.max(dir_errors_deg),
        'mean_direction_error_deg': np.mean(dir_errors_deg),
        'max_vertical_component': max_w,
        'mean_vertical_component': mean_w,
        'max_vertical_normalized': max_w_normalized,
        'is_horizontal': max_w_normalized < 1e-6,
        'direction_correct': np.max(dir_errors_deg) < tolerance_deg
    }


def validate_tke_profile(z_coords: np.ndarray, k_profiles: np.ndarray,
                        u_star: float, h_bl: float, Cmu: float,
                        z_ground: float = 0.0) -> Dict:
    """
    Validate TKE profile against theoretical expectations.
    
    Args:
        z_coords: Height coordinates
        k_profiles: TKE values
        u_star: Friction velocity
        h_bl: Boundary layer height
        Cmu: Turbulence constant
        z_ground: Ground elevation
        
    Returns:
        Dictionary with TKE validation results
    """
    heights = np.maximum(z_coords - z_ground, 0.01)
    
    # Calculate theoretical TKE profile
    k_theoretical = np.zeros_like(heights)
    for i, h in enumerate(heights):
        if h <= 0.99 * h_bl:
            ratio = min(h / h_bl, 0.99)
            k_theoretical[i] = (Cmu**(-0.5)) * u_star**2 * (1.0 - ratio)**2
        else:
            k_theoretical[i] = (Cmu**(-0.5)) * u_star**2 * (1.0 - 0.99)**2
    
    # Ensure minimum values
    k_theoretical = np.maximum(k_theoretical, 1e-6)
    
    # Calculate errors
    relative_errors = np.abs(k_profiles - k_theoretical) / (k_theoretical + 1e-10)
    
    return {
        'k_theoretical': k_theoretical,
        'relative_errors': relative_errors,
        'max_error_percent': np.max(relative_errors) * 100,
        'mean_error_percent': np.mean(relative_errors) * 100,
        'min_k': np.min(k_profiles),
        'max_k': np.max(k_profiles),
        'passes': np.max(relative_errors) < 0.01  # Less than 1% error
    }


def validate_epsilon_profile(z_coords: np.ndarray, epsilon_profiles: np.ndarray,
                            k_profiles: np.ndarray, u_star: float, h_bl: float,
                            kappa: float, Cmu: float, z0: float,
                            z_ground: float = 0.0) -> Dict:
    """
    Validate epsilon (dissipation) profile against theoretical expectations.
    
    Args:
        z_coords: Height coordinates
        epsilon_profiles: Epsilon values
        k_profiles: TKE values (used in calculation)
        u_star: Friction velocity
        h_bl: Boundary layer height
        kappa: Von Karman constant
        Cmu: Turbulence constant
        z0: Surface roughness
        z_ground: Ground elevation
        
    Returns:
        Dictionary with epsilon validation results
    """
    heights = np.maximum(z_coords - z_ground, 0.01)
    
    # Calculate theoretical epsilon profile
    epsilon_theoretical = np.zeros_like(heights)
    for i, h in enumerate(heights):
        if h <= 0.95 * h_bl:
            denom = kappa * (h + z0)
        else:
            denom = kappa * (0.95 * h_bl + z0)
        
        epsilon_theoretical[i] = (Cmu**0.75) * (k_profiles[i]**1.5) / max(denom, 1e-6)
        epsilon_theoretical[i] = max(epsilon_theoretical[i], 1e-8)
    
    # Calculate errors
    relative_errors = np.abs(epsilon_profiles - epsilon_theoretical) / (epsilon_theoretical + 1e-10)
    
    return {
        'epsilon_theoretical': epsilon_theoretical,
        'relative_errors': relative_errors,
        'max_error_percent': np.max(relative_errors) * 100,
        'mean_error_percent': np.mean(relative_errors) * 100,
        'min_epsilon': np.min(epsilon_profiles),
        'max_epsilon': np.max(epsilon_profiles),
        'passes': np.max(relative_errors) < 0.01
    }


def generate_validation_report(validation_results: Dict, config, save_path: Optional[str] = None) -> str:
    """
    Generate a comprehensive text report of validation results.
    
    Args:
        validation_results: Dictionary containing all validation results
        config: ABL configuration object
        save_path: Optional path to save the report
        
    Returns:
        Formatted validation report as string
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("ABL INLET PROFILE VALIDATION REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Configuration summary
    report_lines.append("CONFIGURATION:")
    report_lines.append(f"  Friction velocity (u*): {config.atmospheric.u_star:.4f} m/s")
    report_lines.append(f"  Surface roughness (z0): {config.atmospheric.z0:.6f} m")
    report_lines.append(f"  Boundary layer height: {config.atmospheric.h_bl:.1f} m")
    report_lines.append(f"  Flow direction: {config.atmospheric.flow_dir_deg:.1f} degrees")
    report_lines.append(f"  Von Karman constant: {config.turbulence.kappa:.3f}")
    report_lines.append(f"  Turbulence constant (Cmu): {config.turbulence.Cmu:.4f}")
    report_lines.append("")
    
    # Ground elevation validation
    if 'ground_elevation' in validation_results:
        ge = validation_results['ground_elevation']
        report_lines.append("GROUND ELEVATION VALIDATION:")
        report_lines.append(f"  Mean ground elevation: {ge['z_ground_mean']:.3f} m")
        report_lines.append(f"  Ground elevation range: {ge['z_ground_min']:.3f} to {ge['z_ground_max']:.3f} m")
        report_lines.append(f"  Terrain variation (std): {ge['z_ground_std']:.4f} m")
        report_lines.append(f"  Minimum z in profiles: {ge['z_min_used_in_profiles']:.3f} m")
        status = "✓ PASS" if ge['uses_ground_elevation'] else "✗ FAIL"
        report_lines.append(f"  Ground elevation used: {status}")
        report_lines.append("")
    
    # Flow direction validation
    if 'flow_direction' in validation_results:
        fd = validation_results['flow_direction']
        report_lines.append("FLOW DIRECTION VALIDATION:")
        report_lines.append(f"  Expected direction: {fd['expected_direction_deg']:.2f} degrees")
        report_lines.append(f"  Mean actual direction: {fd['mean_direction_deg']:.2f} degrees")
        report_lines.append(f"  Direction std deviation: {fd['direction_std_deg']:.4f} degrees")
        report_lines.append(f"  Max direction error: {fd['max_direction_error_deg']:.4f} degrees")
        report_lines.append(f"  Mean direction error: {fd['mean_direction_error_deg']:.4f} degrees")
        report_lines.append(f"  Max vertical component: {fd['max_vertical_component']:.6e} m/s")
        report_lines.append(f"  Max vertical/horizontal ratio: {fd['max_vertical_normalized']:.6e}")
        status = "✓ PASS" if fd['direction_correct'] and fd['is_horizontal'] else "✗ FAIL"
        report_lines.append(f"  Flow perpendicular to inlet: {status}")
        report_lines.append("")
    
    # Velocity profile validation
    if 'velocity_profile' in validation_results:
        vp = validation_results['velocity_profile']
        report_lines.append("VELOCITY PROFILE VALIDATION (Log-Law):")
        report_lines.append(f"  Max relative error: {vp['max_error_percent']:.4f} %")
        report_lines.append(f"  Mean relative error: {vp['mean_error_percent']:.4f} %")
        report_lines.append(f"  RMSE: {vp['rmse']:.6f} m/s")
        status = "✓ PASS" if vp['passes'] else "✗ FAIL"
        report_lines.append(f"  Mathematical correctness: {status}")
        report_lines.append("")
    
    # TKE profile validation
    if 'tke_profile' in validation_results:
        tp = validation_results['tke_profile']
        report_lines.append("TURBULENT KINETIC ENERGY (TKE) VALIDATION:")
        report_lines.append(f"  Min TKE: {tp['min_k']:.6e} m²/s²")
        report_lines.append(f"  Max TKE: {tp['max_k']:.6e} m²/s²")
        report_lines.append(f"  Max relative error: {tp['max_error_percent']:.4f} %")
        report_lines.append(f"  Mean relative error: {tp['mean_error_percent']:.4f} %")
        status = "✓ PASS" if tp['passes'] else "✗ FAIL"
        report_lines.append(f"  Mathematical correctness: {status}")
        report_lines.append("")
    
    # Epsilon profile validation
    if 'epsilon_profile' in validation_results:
        ep = validation_results['epsilon_profile']
        report_lines.append("DISSIPATION RATE (EPSILON) VALIDATION:")
        report_lines.append(f"  Min epsilon: {ep['min_epsilon']:.6e} m²/s³")
        report_lines.append(f"  Max epsilon: {ep['max_epsilon']:.6e} m²/s³")
        report_lines.append(f"  Max relative error: {ep['max_error_percent']:.4f} %")
        report_lines.append(f"  Mean relative error: {ep['mean_error_percent']:.4f} %")
        status = "✓ PASS" if ep['passes'] else "✗ FAIL"
        report_lines.append(f"  Mathematical correctness: {status}")
        report_lines.append("")
    
    # Overall status
    report_lines.append("=" * 80)
    all_pass = all([
        validation_results.get('ground_elevation', {}).get('uses_ground_elevation', False),
        validation_results.get('flow_direction', {}).get('direction_correct', False),
        validation_results.get('flow_direction', {}).get('is_horizontal', False),
        validation_results.get('velocity_profile', {}).get('passes', False),
        validation_results.get('tke_profile', {}).get('passes', False),
        validation_results.get('epsilon_profile', {}).get('passes', False)
    ])
    
    if all_pass:
        report_lines.append("OVERALL VALIDATION: ✓ ALL CHECKS PASSED")
    else:
        report_lines.append("OVERALL VALIDATION: ✗ SOME CHECKS FAILED - REVIEW ABOVE")
    report_lines.append("=" * 80)
    
    report = "\n".join(report_lines)
    
    # Save to file if requested
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"Validation report saved to: {save_path}")
    
    return report


def plot_validation_profiles(z_coords: np.ndarray, U_profiles: np.ndarray,
                            k_profiles: np.ndarray, epsilon_profiles: np.ndarray,
                            validation_results: Dict, config,
                            inlet_blocks: list = None,
                            save_dir: Optional[str] = None):
    """
    Create comprehensive validation plots comparing actual vs theoretical profiles.
    
    Args:
        z_coords: Height coordinates
        U_profiles: Velocity profiles array
        k_profiles: TKE profiles
        epsilon_profiles: Dissipation profiles
        validation_results: Dictionary with validation results
        config: ABL configuration
        inlet_blocks: Optional list of inlet blocks for ground elevation plot
        save_dir: Optional directory to save plots
    """
    # Calculate velocity magnitude for first inlet block
    n_z = len(z_coords)
    u_mag = np.linalg.norm(U_profiles[:n_z], axis=1)
    k_vals = k_profiles[:n_z]
    eps_vals = epsilon_profiles[:n_z]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # --- Velocity Profile ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(u_mag, z_coords, 'b-', linewidth=2, label='Actual', marker='o', markersize=3)
    if 'velocity_profile' in validation_results:
        vp = validation_results['velocity_profile']
        ax1.plot(vp['u_theoretical'], z_coords, 'r--', linewidth=2, label='Theoretical (Log-law)')
    ax1.set_xlabel('Velocity [m/s]', fontsize=10)
    ax1.set_ylabel('Height [m]', fontsize=10)
    ax1.set_title('Velocity Profile', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    if hasattr(config.atmospheric, 'h_bl'):
        ax1.axhline(y=config.atmospheric.h_bl, color='k', linestyle=':', alpha=0.5)
    
    # --- Velocity Error ---
    ax2 = fig.add_subplot(gs[0, 1])
    if 'velocity_profile' in validation_results:
        vp = validation_results['velocity_profile']
        ax2.plot(vp['relative_errors'] * 100, z_coords, 'g-', linewidth=2, marker='o', markersize=3)
        ax2.set_xlabel('Relative Error [%]', fontsize=10)
        ax2.set_ylabel('Height [m]', fontsize=10)
        ax2.set_title('Velocity Profile Error', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=0.1, color='r', linestyle='--', alpha=0.5, label='0.1% threshold')
        ax2.legend()
    
    # --- Flow Direction ---
    ax3 = fig.add_subplot(gs[0, 2])
    if 'flow_direction' in validation_results:
        fd = validation_results['flow_direction']
        # Create a compass-like plot
        angles = [fd['expected_direction_deg'], fd['mean_direction_deg']]
        labels = ['Expected', 'Actual']
        colors = ['red', 'blue']
        
        for angle, label, color in zip(angles, labels, colors):
            rad = np.radians(angle)
            ax3.arrow(0, 0, np.cos(rad), np.sin(rad), head_width=0.1, 
                     head_length=0.1, fc=color, ec=color, label=label, linewidth=2)
        
        ax3.set_xlim(-1.5, 1.5)
        ax3.set_ylim(-1.5, 1.5)
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel('X (East)', fontsize=10)
        ax3.set_ylabel('Y (North)', fontsize=10)
        ax3.set_title('Flow Direction', fontsize=12, fontweight='bold')
        ax3.legend()
        
        # Add text with error
        error_text = f"Error: {fd['max_direction_error_deg']:.4f}°"
        ax3.text(0.05, 0.95, error_text, transform=ax3.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # --- TKE Profile ---
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(k_vals, z_coords, 'b-', linewidth=2, label='Actual', marker='o', markersize=3)
    if 'tke_profile' in validation_results:
        tp = validation_results['tke_profile']
        ax4.plot(tp['k_theoretical'], z_coords, 'r--', linewidth=2, label='Theoretical')
    ax4.set_xlabel('TKE [m²/s²]', fontsize=10)
    ax4.set_ylabel('Height [m]', fontsize=10)
    ax4.set_title('Turbulent Kinetic Energy', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    if hasattr(config.atmospheric, 'h_bl'):
        ax4.axhline(y=config.atmospheric.h_bl, color='k', linestyle=':', alpha=0.5)
    
    # --- TKE Error ---
    ax5 = fig.add_subplot(gs[1, 1])
    if 'tke_profile' in validation_results:
        tp = validation_results['tke_profile']
        ax5.plot(tp['relative_errors'] * 100, z_coords, 'g-', linewidth=2, marker='o', markersize=3)
        ax5.set_xlabel('Relative Error [%]', fontsize=10)
        ax5.set_ylabel('Height [m]', fontsize=10)
        ax5.set_title('TKE Profile Error', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.axvline(x=1.0, color='r', linestyle='--', alpha=0.5, label='1% threshold')
        ax5.legend()
    
    # --- Ground Elevation Map ---
    ax6 = fig.add_subplot(gs[1, 2])
    if inlet_blocks:
        x_coords = [block['x_ground'] for block in inlet_blocks]
        y_coords = [block['y_ground'] for block in inlet_blocks]
        z_grounds = [block['z_ground'] for block in inlet_blocks]
        
        scatter = ax6.scatter(x_coords, y_coords, c=z_grounds, cmap='terrain', s=50, edgecolors='k', linewidth=0.5)
        cbar = plt.colorbar(scatter, ax=ax6)
        cbar.set_label('Ground Elevation [m]', fontsize=9)
        ax6.set_xlabel('X [m]', fontsize=10)
        ax6.set_ylabel('Y [m]', fontsize=10)
        ax6.set_title('Inlet Ground Elevation', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.set_aspect('equal')
    
    # --- Epsilon Profile ---
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(eps_vals, z_coords, 'b-', linewidth=2, label='Actual', marker='o', markersize=3)
    if 'epsilon_profile' in validation_results:
        ep = validation_results['epsilon_profile']
        ax7.plot(ep['epsilon_theoretical'], z_coords, 'r--', linewidth=2, label='Theoretical')
    ax7.set_xlabel('Epsilon [m²/s³]', fontsize=10)
    ax7.set_ylabel('Height [m]', fontsize=10)
    ax7.set_title('Dissipation Rate', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    if hasattr(config.atmospheric, 'h_bl'):
        ax7.axhline(y=config.atmospheric.h_bl, color='k', linestyle=':', alpha=0.5)
    
    # --- Epsilon Error ---
    ax8 = fig.add_subplot(gs[2, 1])
    if 'epsilon_profile' in validation_results:
        ep = validation_results['epsilon_profile']
        ax8.plot(ep['relative_errors'] * 100, z_coords, 'g-', linewidth=2, marker='o', markersize=3)
        ax8.set_xlabel('Relative Error [%]', fontsize=10)
        ax8.set_ylabel('Height [m]', fontsize=10)
        ax8.set_title('Epsilon Profile Error', fontsize=12, fontweight='bold')
        ax8.grid(True, alpha=0.3)
        ax8.axvline(x=1.0, color='r', linestyle='--', alpha=0.5, label='1% threshold')
        ax8.legend()
    
    # --- Validation Summary ---
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    # Create summary text
    summary_text = "VALIDATION SUMMARY\n" + "="*30 + "\n\n"
    
    checks = []
    if 'ground_elevation' in validation_results:
        ge = validation_results['ground_elevation']
        status = "✓" if ge['uses_ground_elevation'] else "✗"
        checks.append(f"{status} Ground elevation used")
    
    if 'flow_direction' in validation_results:
        fd = validation_results['flow_direction']
        status = "✓" if fd['direction_correct'] and fd['is_horizontal'] else "✗"
        checks.append(f"{status} Flow direction correct")
    
    if 'velocity_profile' in validation_results:
        vp = validation_results['velocity_profile']
        status = "✓" if vp['passes'] else "✗"
        checks.append(f"{status} Velocity (log-law)")
        checks.append(f"   Error: {vp['mean_error_percent']:.4f}%")
    
    if 'tke_profile' in validation_results:
        tp = validation_results['tke_profile']
        status = "✓" if tp['passes'] else "✗"
        checks.append(f"{status} TKE profile")
        checks.append(f"   Error: {tp['mean_error_percent']:.4f}%")
    
    if 'epsilon_profile' in validation_results:
        ep = validation_results['epsilon_profile']
        status = "✓" if ep['passes'] else "✗"
        checks.append(f"{status} Epsilon profile")
        checks.append(f"   Error: {ep['mean_error_percent']:.4f}%")
    
    summary_text += "\n".join(checks)
    
    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.suptitle('ABL Inlet Profile Validation', fontsize=14, fontweight='bold', y=0.995)
    
    # Save if directory provided
    if save_dir:
        save_path = Path(save_dir) / 'validation_plots.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Validation plots saved to: {save_path}")
    
    return fig


def run_comprehensive_validation(case_dir: str, config, inlet_data, 
                                 U_profiles: np.ndarray, k_profiles: np.ndarray,
                                 epsilon_profiles: np.ndarray, z_coords: np.ndarray,
                                 generate_plots: bool = True) -> Dict:
    """
    Run complete validation suite on generated profiles.
    
    Args:
        case_dir: Case directory for saving outputs
        config: ABL configuration
        inlet_data: Tuple of (inlet_blocks, mesh_params)
        U_profiles: Velocity profiles
        k_profiles: TKE profiles
        epsilon_profiles: Dissipation profiles
        z_coords: Height coordinates
        generate_plots: Whether to generate validation plots
        
    Returns:
        Dictionary containing all validation results
    """
    inlet_blocks, mesh_params = inlet_data
    
    # Extract parameters
    z_ground = mesh_params['avg_inlet_height']
    
    # Get z0 for first block (representative)
    if config.atmospheric.z0 == 0.0 and 'z0' in inlet_blocks[0]:
        z0_local = inlet_blocks[0]['z0']
    else:
        z0_local = config.atmospheric.z0
    
    # Calculate velocity magnitude for validation
    n_z = len(z_coords)
    u_mag = np.linalg.norm(U_profiles[:n_z], axis=1)
    k_vals = k_profiles[:n_z]
    eps_vals = epsilon_profiles[:n_z]
    
    # Run all validations
    validation_results = {}
    
    print("\nRunning validation checks...")
    print("-" * 60)
    
    # 1. Ground elevation validation
    print("✓ Validating ground elevation usage...")
    validation_results['ground_elevation'] = validate_ground_elevation_usage(
        inlet_blocks, z_coords, U_profiles
    )
    
    # 2. Flow direction validation
    print("✓ Validating flow direction...")
    validation_results['flow_direction'] = validate_flow_direction(
        U_profiles, config.atmospheric.flow_dir_deg
    )
    
    # 3. Velocity profile validation
    print("✓ Validating velocity profile (log-law)...")
    validation_results['velocity_profile'] = validate_log_law_profile(
        z_coords, u_mag, config.atmospheric.u_star, z0_local,
        config.turbulence.kappa, z_ground
    )
    
    # 4. TKE profile validation
    print("✓ Validating TKE profile...")
    validation_results['tke_profile'] = validate_tke_profile(
        z_coords, k_vals, config.atmospheric.u_star,
        config.atmospheric.h_bl, config.turbulence.Cmu, z_ground
    )
    
    # 5. Epsilon profile validation
    print("✓ Validating epsilon profile...")
    validation_results['epsilon_profile'] = validate_epsilon_profile(
        z_coords, eps_vals, k_vals, config.atmospheric.u_star,
        config.atmospheric.h_bl, config.turbulence.kappa,
        config.turbulence.Cmu, z0_local, z_ground
    )
    
    print("-" * 60)
    
    # Generate report
    report_path = Path(case_dir) / 'validation_report.txt'
    report = generate_validation_report(validation_results, config, str(report_path))
    print("\n" + report)
    
    # Generate plots
    if generate_plots:
        print("\nGenerating validation plots...")
        plot_validation_profiles(
            z_coords, U_profiles, k_profiles, epsilon_profiles,
            validation_results, config, inlet_blocks, save_dir=case_dir
        )
        plt.show()
    
    # Save validation results as JSON for programmatic access
    json_path = Path(case_dir) / 'validation_results.json'
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for key, value in validation_results.items():
        json_results[key] = {}
        for k, v in value.items():
            if isinstance(v, np.ndarray):
                json_results[key][k] = v.tolist()
            else:
                json_results[key][k] = v
    
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Validation results saved to: {json_path}")
    
    return validation_results
