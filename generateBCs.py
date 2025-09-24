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


def calculate_inlet_profiles_from_mesh(config: ABLConfig, inlet_data, use_face_centers: bool = True):
    """
    Calculate U, k, epsilon profiles for inlet based on mesh grading from file
    
    Args:
        config: ABL configuration object (only for atmospheric/turbulence params)
        inlet_data: tuple of (inlet_blocks, mesh_params) from read_inlet_face_file()
        use_face_centers: If True, use cell centers; if False, use internal faces
        
    Returns:
        Tuple of (U_profiles, k_profiles, epsilon_profiles)
    """
    inlet_blocks, mesh_params = inlet_data
    
    atm = config.atmospheric  
    turb = config.turbulence
    
    # Get domain parameters from mesh_params instead of config
    domain_height = mesh_params['domain_height']
    avg_inlet_height = mesh_params['avg_inlet_height']
    
    # Calculate z-coordinates using parameters from file
    z_coords = calculate_multiregion_z_distribution(
        avg_inlet_height,
        domain_height,
        mesh_params,
        use_face_centers
    )
    
    # Generate profiles for each block x each z-level
    total_faces = len(inlet_blocks) * len(z_coords)
    print("Inlet blocks found:", len(inlet_blocks), "with", len(z_coords), "z-levels each.")
    print(f"Calculating profiles for {total_faces} inlet faces...")
    
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
            height = max(z - avg_inlet_height, 0.01)
            
            # [Rest of profile calculation remains the same]
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
    Now reads mesh parameters from inlet face file instead of config
    """
    if config is None:
        config = ABLConfig()
    
    # Add the missing Ustar attribute if not present
    if not hasattr(config.atmospheric, 'u_star'):
        config.atmospheric.u_star = 0.40
    
    # Read inlet blocks AND mesh parameters from saved file
    inlet_file = os.path.join(case_dir, "0/include/inletFaceInfo.txt")
    inlet_data = read_inlet_face_file(inlet_file)  # Returns (inlet_blocks, mesh_params)
    inlet_blocks, mesh_params = inlet_data

    # Calculate z-coordinates from file parameters (not config)
    z_coords = calculate_multiregion_z_distribution(
        mesh_params['avg_inlet_height'],
        mesh_params['domain_height'],
        mesh_params,
        use_face_centers
    )

    initial_vals = calculate_initial_conditions(config)

    # Calculate profiles using file data
    U_profiles, k_profiles, epsilon_profiles = calculate_inlet_profiles_from_mesh(
        config, inlet_data, use_face_centers)
    
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

def create_blockMesh_spacing(n_points, grading_spec):
    """
    Create variable spacing coordinates from 0 to 1 using blockMesh-style grading.
    
    Parameters:
    -----------
    n_points : int
        Total number of points
    grading_spec : list of tuples
        [(length_fraction, cell_fraction, expansion_ratio), ...]
        - length_fraction: fraction of domain length for this region
        - cell_fraction: fraction of total cells for this region  
        - expansion_ratio: last_cell_size/first_cell_size in this region
    
    Returns:
    --------
    np.ndarray
        Coordinate array from 0 to 1 with blockMesh-style spacing
    """
    
    total_cells = n_points - 1
    n_regions = len(grading_spec)
    
    # Extract specifications
    length_fractions = np.array([spec[0] for spec in grading_spec])
    cell_fractions = np.array([spec[1] for spec in grading_spec])
    expansion_ratios = np.array([spec[2] for spec in grading_spec])
    
    # Validate inputs
    if abs(length_fractions.sum() - 1.0) > 1e-6:
        raise ValueError(f"Length fractions sum to {length_fractions.sum():.6f}, must sum to 1.0")
    
    if abs(cell_fractions.sum() - 1.0) > 1e-6:
        raise ValueError(f"Cell fractions sum to {cell_fractions.sum():.6f}, must sum to 1.0")
    
    # Calculate target cell counts (may not be integers)
    target_cells = cell_fractions * total_cells
    
    # Round to integers and adjust to maintain total
    actual_cells = np.round(target_cells).astype(int)
    
    # Adjust for rounding errors
    cell_diff = total_cells - actual_cells.sum()
    if cell_diff != 0:
        # Add/subtract cells from regions with largest rounding errors
        errors = target_cells - actual_cells
        if cell_diff > 0:
            # Need to add cells - add to regions with most positive error
            indices = np.argsort(errors)[::-1]
        else:
            # Need to remove cells - remove from regions with most negative error  
            indices = np.argsort(errors)
        
        for i in range(abs(cell_diff)):
            actual_cells[indices[i]] += np.sign(cell_diff)
    
    # Generate coordinates for each region
    coords = [0.0]  # Start at 0
    current_pos = 0.0
    
    for i, (length_frac, actual_cell_count, expansion_ratio) in enumerate(zip(length_fractions, actual_cells, expansion_ratios)):
        region_length = length_frac
        
        if actual_cell_count == 0:
            continue
            
        # Generate spacing within this region
        region_coords = generate_region_coordinates(actual_cell_count, expansion_ratio)
        
        # Scale to region length and add to current position
        region_coords_scaled = region_coords * region_length + current_pos
        
        # Add coordinates (skip the first one as it's already included)
        coords.extend(region_coords_scaled[1:])
        
        current_pos += region_length
    
    return np.array(coords)

def generate_region_coordinates(n_cells, expansion_ratio):
    """
    Generate coordinates within a single region [0,1] with given expansion ratio.
    
    Parameters:
    -----------
    n_cells : int
        Number of cells in this region
    expansion_ratio : float
        Ratio of last_cell_size/first_cell_size
        
    Returns:
    --------
    np.ndarray
        Coordinates from 0 to 1 for this region
    """
    
    if n_cells == 0:
        return np.array([0.0, 1.0])
    
    if n_cells == 1:
        return np.array([0.0, 1.0])
    
    # For uniform spacing (expansion_ratio ≈ 1)
    if abs(expansion_ratio - 1.0) < 1e-6:
        return np.linspace(0.0, 1.0, n_cells + 1)
    
    # For geometric progression
    r = expansion_ratio**(1.0/(n_cells-1))  # Common ratio between adjacent cells
    
    # Calculate first cell size
    if abs(r - 1.0) < 1e-6:
        ds = 1.0 / n_cells
    else:
        ds = (r - 1.0) / (r**n_cells - 1.0)
    
    # Generate cell sizes
    cell_sizes = ds * r**np.arange(n_cells)
    
    # Generate coordinates
    coords = np.zeros(n_cells + 1)
    coords[1:] = np.cumsum(cell_sizes)
    
    return coords

def read_inlet_face_file(file_path):
    """Read inlet face information and mesh parameters from blockMesh generator"""
    inlet_blocks = []
    mesh_params = {}
    
    print("Reading inlet face information from:", file_path)
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Parse mesh parameters section
    in_mesh_section = False
    in_face_section = False
    
    for line in lines:
        line = line.strip()
        
        if line == "# MESH_PARAMETERS_START":
            in_mesh_section = True
            continue
        elif line == "# MESH_PARAMETERS_END":
            in_mesh_section = False
            continue
        elif line == "# FACE_DATA_START":
            in_face_section = True
            continue
        elif line == "# FACE_DATA_END":
            in_face_section = False
            continue
        
        # Parse mesh parameters
        if in_mesh_section and '=' in line:
            key, value = line.split('=', 1)
            if key == 'z_grading':
                # Parse z_grading format: "0.04,0.25,2.5;0.96,0.75,100.0"
                grading_specs = []
                for spec_str in value.split(';'):
                    parts = [float(x) for x in spec_str.split(',')]
                    grading_specs.append(tuple(parts))
                mesh_params[key] = grading_specs
            else:
                # Try to convert to appropriate type
                try:
                    if '.' in value:
                        mesh_params[key] = float(value)
                    else:
                        mesh_params[key] = int(value) if value.isdigit() else value
                except ValueError:
                    mesh_params[key] = value
        
        # Parse face data
        if in_face_section and not line.startswith('#') and line:
            parts = line.split(',')
            if len(parts) == 5:
                inlet_blocks.append({
                    'block_i': int(parts[0]),
                    'block_j': int(parts[1]), 
                    'x_ground': float(parts[2]),
                    'y_ground': float(parts[3]),
                    'z_ground': float(parts[4])
                })
    
    print(f"Read {len(inlet_blocks)} inlet blocks and mesh parameters:")
    for key, value in mesh_params.items():
        print(f"  {key}: {value}")
    
    return inlet_blocks, mesh_params

def calculate_multiregion_z_distribution(z_ground: float, z_top: float, mesh_params: dict, 
                                       use_face_centers: bool = True) -> np.ndarray:
    """
    Calculate z-coordinates for multi-region grading using mesh parameters from file.
    """
    domain_height = z_top - z_ground
    
    if mesh_params.get('mesh_type') == 'legacy':
        # Use legacy single expansion
        return calculate_graded_z_distribution(
            z_ground, z_top, 
            mesh_params['num_cells_z'], 
            mesh_params['expansion_ratio_z'],
            use_face_centers
        )
    
    # Handle new grading system
    total_cells = mesh_params['total_z_cells']
    z_grading = mesh_params['z_grading']
    first_cell_height = mesh_params.get('first_cell_height')
    
    # Create normalized coordinates using blockMesh spacing logic
    if first_cell_height is not None:
        # With first cell
        first_cell_frac = first_cell_height / domain_height
        remaining_height_frac = 1.0 - first_cell_frac
        
        # Create coordinates for remaining layers
        remaining_cells = total_cells - 1
        if remaining_cells > 0:
            # Use your blockMesh spacing function for remaining layers
            remaining_coords = create_blockMesh_spacing(remaining_cells + 1, z_grading)
            # Scale to remaining height and add first cell
            z_normalized = np.zeros(total_cells + 1)
            z_normalized[0] = 0.0
            z_normalized[1] = first_cell_frac
            z_normalized[2:] = first_cell_frac + remaining_coords[1:] * remaining_height_frac
        else:
            # Only first cell
            z_normalized = np.array([0.0, first_cell_frac, 1.0])
    else:
        # No first cell, use standard grading
        z_normalized = create_blockMesh_spacing(total_cells + 1, z_grading)
    
    # Convert to actual coordinates
    z_faces = z_ground + z_normalized * domain_height
    
    if use_face_centers:
        # Return cell centers
        return 0.5 * (z_faces[:-1] + z_faces[1:])
    else:
        # Return internal faces (excluding boundaries)
        return z_faces[1:-1]


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
    config.mesh.domain_height = 3000.0
    config.mesh.num_cells_z = 80
    config.mesh.expansion_ratio_R = 100.0
    
    case_dir = "/Users/ssudhakaran/Documents/Simulations/API/openFoam/meshRefine"
    
    # Generate using cell centers (default)
    results = generate_inlet_data_workflow(case_dir, config, use_face_centers=True)
    
    # Or generate using internal faces
    # results = generate_inlet_data_workflow(case_dir, config, use_face_centers=False)