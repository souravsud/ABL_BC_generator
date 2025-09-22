import numpy as np
from pathlib import Path
from typing import Tuple
from config import ABLConfig
import os
import matplotlib.pyplot as plt

def calculate_graded_z_distribution(z_ground: float, z_top: float, n_cells: int, 
                                  expansion_ratio: float, use_face_centers: bool = True) -> np.ndarray:
    """
    Calculate z-coordinates based on OpenFOAM simpleGrading (last_cell/first_cell ratio).
    
    Args:
        z_ground: Bottom boundary z-coordinate
        z_top: Top boundary z-coordinate  
        n_cells: Number of cells in z-direction
        expansion_ratio: Expansion ratio (last_cell_height/first_cell_height)
        use_face_centers: If True, return cell centers; if False, return internal faces
        
    Returns:
        Array of z-coordinates
    """
    domain_height = z_top - z_ground
    
    if expansion_ratio == 1.0:
        # Uniform spacing
        z_faces = np.linspace(z_ground, z_top, n_cells + 1)
    else:
        # Calculate first cell height based on expansion ratio
        # expansion_ratio = h_last / h_first
        # For geometric progression: h_i = h_first * r^i, where r^(n-1) = expansion_ratio
        r = expansion_ratio**(1.0/(n_cells - 1))  # Geometric ratio
        h_first = domain_height * (r - 1) / (r**n_cells - 1)
        
        # Calculate face positions
        z_faces = np.zeros(n_cells + 1)
        z_faces[0] = z_ground
        
        for i in range(n_cells):
            cell_height = h_first * (r**i)
            z_faces[i + 1] = z_faces[i] + cell_height
    
    if use_face_centers:
        # Return cell centers
        return 0.5 * (z_faces[:-1] + z_faces[1:])
    else:
        # Return internal faces (excluding boundaries)
        return z_faces[1:-1]


def calculate_inlet_profiles_from_mesh(config: ABLConfig, inlet_blocks = None, use_face_centers: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate U, k, epsilon profiles for inlet based on mesh grading
    
    Args:
        config: ABL configuration object
        use_face_centers: If True, use cell centers; if False, use internal faces
        
    Returns:
        Tuple of (U_profiles, k_profiles, epsilon_profiles)
    """
    atm = config.atmospheric
    turb = config.turbulence
    mesh = config.mesh
    
    # Calculate z-coordinates based on mesh grading
    z_coords = calculate_graded_z_distribution(
        mesh.inlet_height,
        mesh.domain_height,
        mesh.num_cells_z,
        mesh.expansion_ratio_R,
        use_face_centers
    )
    
    # Generate profiles for each block x each z-level
    total_faces = len(inlet_blocks) * len(z_coords)
    U_profiles = np.zeros((total_faces, 3))
    k_profiles = np.zeros(total_faces)
    epsilon_profiles = np.zeros(total_faces)
    
    # Flow direction
    flow_dir_rad = np.radians(atm.flow_dir_deg)
    flow_dir_x = np.cos(flow_dir_rad)
    flow_dir_y = np.sin(flow_dir_rad)
    
    
    face_idx = 0
    for block in inlet_blocks:
        for i, z in enumerate(z_coords):
            height = max(z - mesh.inlet_height, 0.01)
            
            # Velocity profile
            if height <= atm.h_bl:
                u_mag = (atm.u_star / turb.kappa) * np.log(1.0 + height / atm.z0)
            else:
                u_mag = (atm.u_star / turb.kappa) * np.log(1.0 + atm.h_bl / atm.z0)
                
            U_profiles[face_idx] = [u_mag * flow_dir_x, u_mag * flow_dir_y, 0.0]
            
            # TKE profile  
            if height <= 0.99 * atm.h_bl:
                ratio = min(height / atm.h_bl, 0.99)
                k_profiles[face_idx] = (turb.Cmu**(-0.5)) * atm.u_star**2 * (1.0 - ratio)**2
            else:
                k_profiles[face_idx] = (turb.Cmu**(-0.5)) * atm.u_star**2 * (1.0 - 0.99)**2
                
            k_profiles[face_idx] = max(k_profiles[face_idx], 1e-6)
            
            # Epsilon profile
            if height <= 0.95 * atm.h_bl:
                denom = turb.kappa * (height + atm.z0)
            else:
                denom = turb.kappa * (0.95 * atm.h_bl + atm.z0)
                
            epsilon_profiles[face_idx] = (turb.Cmu**0.75) * (k_profiles[face_idx]**1.5) / max(denom, 1e-6)
            epsilon_profiles[face_idx] = max(epsilon_profiles[face_idx], 1e-8)

            face_idx += 1
        
    return U_profiles, k_profiles, epsilon_profiles


def write_openfoam_data_files(case_dir: str, U_profiles: np.ndarray, k_profiles: np.ndarray, 
                             epsilon_profiles: np.ndarray, config: ABLConfig):
    """Write boundary condition data files for OpenFOAM"""
    constant_dir = Path(case_dir) / 'constant'
    constant_dir.mkdir(exist_ok=True)
    
    # Write velocity data
    with open(constant_dir / 'inletU', 'w') as f:
        f.write(f"{len(U_profiles)}\n(\n")
        for u_vec in U_profiles:
            f.write(f"({u_vec[0]:.6f} {u_vec[1]:.6f} {u_vec[2]:.6f})\n")
        f.write(")\n\n// ************************************************************************* //\n")
    
    # Write k data
    with open(constant_dir / 'inletK', 'w') as f:
        f.write(f"{len(k_profiles)}\n(\n")
        for k_val in k_profiles:
            f.write(f"{k_val:.8f}\n")
        f.write(")\n\n// ************************************************************************* //\n")
    
    # Write epsilon data  
    with open(constant_dir / 'inletEpsilon', 'w') as f:
        f.write(f"{len(epsilon_profiles)}\n(\n")
        for eps_val in epsilon_profiles:
            f.write(f"{eps_val:.10f}\n")
        f.write(")\n\n// ************************************************************************* //\n")


def generate_boundary_condition_files(case_dir: str, config: ABLConfig, initial_vals):
    """Generate boundary condition files that read from data files"""
    zero_dir = Path(case_dir) / '0'
    zero_dir.mkdir(exist_ok=True)
    
    patches = config.mesh.patch_names
    foam_version = config.openfoam.foam_version
    
    # U boundary condition file
    u_content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  {foam_version}                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     {config.openfoam.version};
    format      ascii;
    class       volVectorField;
    object      U;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform ({initial_vals['flowVelocity'][0]:.3f} {initial_vals['flowVelocity'][1]:.3f} {initial_vals['flowVelocity'][2]:.3f});

boundaryField
{{
    {patches['inlet']}
    {{
        type            fixedValue;
        value           nonuniform
        #include        "../constant/inletU"
        ;
    }}
    
    {patches['outlet']}
    {{
        type            {config.openfoam.boundary_conditions['U']['outlet']['type']};
    }}
    
    {patches['ground']}
    {{
        type             {config.openfoam.boundary_conditions['U']['ground']['type']};
    }}
    
    {patches['sky']}
    {{
        type             {config.openfoam.boundary_conditions['U']['sky']['type']};
    }}
    
    {patches['sides']}
    {{
        type             {config.openfoam.boundary_conditions['U']['sides']['type']};
    }}
    
    "proc.*"
    {{
        type            processor;
    }}
}}

// ************************************************************************* //
"""
    
    # k boundary condition file
    k_content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  {foam_version}                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     {config.openfoam.version};
    format      ascii;
    class       volScalarField;
    object      k;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform {initial_vals['turbulentKE']:.6f};

boundaryField
{{
    {patches['inlet']}
    {{
        type            fixedValue;
        value           nonuniform
        #include        "../constant/inletK"
        ;
    }}
    
    {patches['outlet']}
    {{
        type            {config.openfoam.boundary_conditions['k']['outlet']['type']};
    }}
    
    {patches['ground']}
    {{
        type            {config.openfoam.wall_functions['ground_k']['type']};
        value           uniform {config.openfoam.wall_functions['ground_k']['value']};
    }}
    
    {patches['sky']}
    {{
        type            {config.openfoam.boundary_conditions['k']['sky']['type']};
    }}
    
    {patches['sides']}
    {{
        type            {config.openfoam.boundary_conditions['k']['sides']['type']};
    }}
    
    "proc.*"
    {{
        type            processor;
    }}
}}

// ************************************************************************* //
"""

    # epsilon boundary condition file
    eps_wall = config.openfoam.wall_functions['ground_epsilon']
    epsilon_content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  {foam_version}                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     {config.openfoam.version};
    format      ascii;
    class       volScalarField;
    object      epsilon;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -3 0 0 0 0];

internalField   uniform {initial_vals['turbulentEpsilon']:.8f};

boundaryField
{{
    {patches['inlet']}
    {{
        type            fixedValue;
        value           nonuniform
        #include        "../constant/inletEpsilon"
        ;
    }}
    
    {patches['outlet']}
    {{
        type            {config.openfoam.boundary_conditions['epsilon']['outlet']['type']};
    }}
    
    {patches['ground']}
    {{
        type            {eps_wall['type']};
        Cmu             {eps_wall['Cmu']};
        kappa           {eps_wall['kappa']};
        E               {eps_wall['E']};
        value           uniform {eps_wall['value']};
    }}
    
    {patches['sky']}
    {{
        type            {config.openfoam.boundary_conditions['epsilon']['sky']['type']};
    }}
    
    {patches['sides']}
    {{
        type            {config.openfoam.boundary_conditions['epsilon']['sides']['type']};
    }}
    
    "proc.*"
    {{
        type            processor;
    }}
}}

// ************************************************************************* //
"""
    
    # Write the files
    with open(zero_dir / 'U', 'w') as f:
        f.write(u_content)
        
    with open(zero_dir / 'k', 'w') as f:
        f.write(k_content)
        
    with open(zero_dir / 'epsilon', 'w') as f:
        f.write(epsilon_content)


def generate_inlet_data_workflow(case_dir: str, config: ABLConfig = None, 
                               use_face_centers: bool = True, plot_profiles: bool = True):
    """
    Complete workflow for mesh-based ABL inlet data generation
    """
    if config is None:
        config = ABLConfig()
    
    # Add the missing Ustar attribute if not present
    if not hasattr(config.atmospheric, 'u_star'):
        config.atmospheric.u_star = 0.40  # Use your reference solver value
    
    # Read inlet blocks from saved file
    inlet_file = os.path.join(case_dir, "0/include/inletFaceInfo.txt")
    inlet_blocks = read_inlet_face_file(inlet_file)

    # Calculate z-coordinates based on mesh grading
    z_coords = calculate_graded_z_distribution(
        config.mesh.inlet_height,
        config.mesh.domain_height,
        config.mesh.num_cells_z,
        config.mesh.expansion_ratio_R,
        use_face_centers
    )

    initial_vals = calculate_initial_conditions(config)

    # Calculate profiles based on mesh grading
    U_profiles, k_profiles, epsilon_profiles = calculate_inlet_profiles_from_mesh(
        config, inlet_blocks, use_face_centers)
    
    # Write data files
    write_openfoam_data_files(case_dir, U_profiles, k_profiles, epsilon_profiles, config)
    
    # Generate boundary condition files
    generate_boundary_condition_files(case_dir, config, initial_vals)
    
    # Generate initial conditions file  # <-- NEW
    write_initial_conditions_file(case_dir, config, initial_vals)  # <-- NEW
    
    # Optional plotting
    if plot_profiles:
        plot_inlet_profiles(z_coords, U_profiles, k_profiles, epsilon_profiles, 
                          config, save_dir=case_dir)
    
    return {
        'U_profiles': U_profiles,
        'k_profiles': k_profiles,
        'epsilon_profiles': epsilon_profiles,
        'z_coords': z_coords,
        'config': config
    }

def read_inlet_face_file(file_path):
    """Read inlet face information from blockMesh generator"""
    inlet_blocks = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split(',')
            if len(parts) == 5:
                inlet_blocks.append({
                    'block_i': int(parts[0]),
                    'block_j': int(parts[1]), 
                    'x_ground': float(parts[2]),
                    'y_ground': float(parts[3]),
                    'z_ground': float(parts[4])
                })
    
    return inlet_blocks

def calculate_initial_conditions(config: ABLConfig, use_face_centers: bool = True) -> dict:
    """
    Calculate representative initial condition values based on inlet profile equations
    
    Args:
        config: ABL configuration object
        use_face_centers: If True, use cell centers; if False, use internal faces
        
    Returns:
        Dictionary with flowVelocity, turbulentKE, turbulentEpsilon values
    """
    atm = config.atmospheric
    turb = config.turbulence
    
    zref = 800
    # Use reference height for initial conditions
    ref_height = max(zref, 100.0)  # Use at least 100m above ground
    
    # Calculate velocity at reference height
    u_mag = (atm.u_star / turb.kappa) * np.log(1.0 + ref_height / atm.z0)
    
    # Flow direction
    flow_dir_rad = np.radians(atm.flow_dir_deg)
    flow_dir_x = np.cos(flow_dir_rad)
    flow_dir_y = np.sin(flow_dir_rad)
    
    flow_velocity = (u_mag * flow_dir_x, u_mag * flow_dir_y, 0.0)
    
    # Calculate k at reference height
    if ref_height <= 0.99 * atm.h_bl:
        ratio = min(ref_height / atm.h_bl, 0.99)
        k_val = (turb.Cmu**(-0.5)) * atm.u_star**2 * (1.0 - ratio)**2
    else:
        k_val = (turb.Cmu**(-0.5)) * atm.u_star**2 * (1.0 - 0.99)**2
    
    k_val = max(k_val, 1e-6)
    
    # Calculate epsilon at reference height
    if ref_height <= 0.95 * atm.h_bl:
        denom = turb.kappa * (ref_height + atm.z0)
    else:
        denom = turb.kappa * (0.95 * atm.h_bl + atm.z0)
    
    eps_val = (turb.Cmu**0.75) * (k_val**1.5) / max(denom, 1e-6)
    eps_val = max(eps_val, 1e-8)
    
    return {
        'flowVelocity': flow_velocity,
        'turbulentKE': k_val,
        'turbulentEpsilon': eps_val,
        'pressure': 0.0
    }

def write_initial_conditions_file(case_dir: str, config: ABLConfig, initial_vals):
    """Write initialConditions file based on inlet profile equations"""
    include_dir = Path(case_dir) / '0' / 'include'
    include_dir.mkdir(parents=True, exist_ok=True)
    
    
    
    content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
========= |
\\\\ / F ield | OpenFOAM: The Open Source CFD Toolbox
\\\\ / O peration | Website: https://openfoam.org
\\\\ / A nd | Version: 12
\\\\/ M anipulation |
\\*---------------------------------------------------------------------------*/
flowVelocity ({initial_vals['flowVelocity'][0]:.3f} {initial_vals['flowVelocity'][1]:.3f} {initial_vals['flowVelocity'][2]:.3f});
pressure {initial_vals['pressure']};
turbulentKE {initial_vals['turbulentKE']:.6f};
turbulentEpsilon {initial_vals['turbulentEpsilon']:.8f};
// ************************************************************************* //
"""
    
    with open(include_dir / 'initialConditions', 'w') as f:
        f.write(content)
    
    print(f"Generated initialConditions file with:")
    print(f"  flowVelocity: {initial_vals['flowVelocity']}")
    print(f"  turbulentKE: {initial_vals['turbulentKE']:.6f}")
    print(f"  turbulentEpsilon: {initial_vals['turbulentEpsilon']:.8f}")

def plot_inlet_profiles(z_coords: np.ndarray, U_profiles: np.ndarray, 
                    k_profiles: np.ndarray, epsilon_profiles: np.ndarray,
                    config, save_dir: str = None):
    """
    Plot ABL inlet profiles for verification
    
    Args:
        z_coords: Height coordinates
        U_profiles: Velocity profiles [n_faces, 3]
        k_profiles: TKE profiles [n_faces]  
        epsilon_profiles: Dissipation profiles [n_faces]
        config: ABL configuration
        save_dir: Directory to save plots (optional)
    """
    
    # Calculate velocity magnitude for first inlet block (representative)
    n_z = len(z_coords)
    u_mag = np.linalg.norm(U_profiles[:n_z], axis=1)  # First n_z faces
    k_vals = k_profiles[:n_z]  # First n_z faces
    eps_vals = epsilon_profiles[:n_z]  # First n_z faces
    
    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))
    
    # Plot velocity magnitude
    ax1.plot(u_mag, z_coords, 'b-', linewidth=2, label='Velocity magnitude')
    ax1.set_xlabel('Velocity magnitude [m/s]')
    ax1.set_ylabel('Height [m]')
    ax1.set_title('Velocity Profile')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add reference lines
    if hasattr(config.atmospheric, 'h_bl'):
        ax1.axhline(y=config.atmospheric.h_bl, color='r', linestyle='--', 
                alpha=0.7, label=f'BL height ({config.atmospheric.h_bl}m)')
    
    # Plot TKE
    ax2.plot(k_vals, z_coords, 'g-', linewidth=2, label='TKE')
    ax2.set_xlabel('TKE [m²/s²]')
    ax2.set_ylabel('Height [m]')
    ax2.set_title('Turbulent Kinetic Energy')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    if hasattr(config.atmospheric, 'h_bl'):
        ax2.axhline(y=config.atmospheric.h_bl, color='r', linestyle='--', 
                alpha=0.7, label=f'BL height ({config.atmospheric.h_bl}m)')
    
    # Plot epsilon
    ax3.plot(eps_vals, z_coords, 'r-', linewidth=2, label='Epsilon')
    ax3.set_xlabel('Epsilon [m²/s³]')
    ax3.set_ylabel('Height [m]')
    ax3.set_title('Turbulent Dissipation Rate')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    if hasattr(config.atmospheric, 'h_bl'):
        ax3.axhline(y=config.atmospheric.h_bl, color='r', linestyle='--', 
                alpha=0.7, label=f'BL height ({config.atmospheric.h_bl}m)')
    
    plt.tight_layout()
    
    # Save if directory provided
    if save_dir:
        save_path = Path(save_dir) / 'inlet_profiles.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
# Example usage
if __name__ == "__main__":
    
    # Simple usage with defaults
    config = ABLConfig()
    config.mesh.inlet_height = 0.0
    config.mesh.domain_height = 4000.0
    config.mesh.num_cells_z = 50
    config.mesh.expansion_ratio_R = 100.0
    
    case_dir = "/Users/ssudhakaran/Documents/Simulations/API/openFoam/meshStructured"
    
    # Generate using cell centers (default)
    results = generate_inlet_data_workflow(case_dir, config, use_face_centers=True)
    
    # Or generate using internal faces
    # results = generate_inlet_data_workflow(case_dir, config, use_face_centers=False)