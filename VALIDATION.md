# ABL Inlet Profile Validation

This document describes the validation tools available to inspect the correctness of generated ABL inlet profiles without running OpenFOAM simulations.

## Overview

The validation tools provide comprehensive checks to verify:
- **Mathematical Correctness**: Profiles follow correct ABL theory (logarithmic law, turbulence profiles)
- **Ground Elevation Usage**: Profiles correctly use terrain elevation data from inlet face
- **Flow Direction**: Flow is in the correct direction and perpendicular to inlet face
- **Numerical Accuracy**: Statistical analysis and error metrics

## Quick Start

### Method 1: Integrated Validation (Recommended)

Run validation automatically during profile generation:

```python
from config import ABLConfig
from generateBCs import generate_inlet_data_workflow

config = ABLConfig()
case_dir = "/path/to/your/case"

# Generate profiles with comprehensive validation
results = generate_inlet_data_workflow(
    case_dir, 
    config, 
    use_face_centers=True,
    validate_profiles=True  # Enable validation
)
```

### Method 2: Standalone Validation

Validate existing profiles after generation:

```bash
python run_validation.py /path/to/case
```

With custom parameters:

```bash
python run_validation.py /path/to/case \
    --u-star 0.3 \
    --z0 0.1 \
    --h-bl 1500 \
    --flow-dir 270
```

## Validation Outputs

The validation generates three types of outputs:

### 1. Text Report (`validation_report.txt`)

Comprehensive text summary with:
- Configuration parameters
- Ground elevation statistics
- Flow direction analysis
- Profile validation results (velocity, TKE, epsilon)
- Pass/fail status for each check

Example output:
```
================================================================================
ABL INLET PROFILE VALIDATION REPORT
================================================================================

CONFIGURATION:
  Friction velocity (u*): 0.2500 m/s
  Surface roughness (z0): 0.100000 m
  Boundary layer height: 1500.0 m
  Flow direction: 45.0 degrees

GROUND ELEVATION VALIDATION:
  Mean ground elevation: 123.456 m
  Ground elevation range: 120.000 to 126.500 m
  Terrain variation (std): 1.2345 m
  Ground elevation used: ✓ PASS

FLOW DIRECTION VALIDATION:
  Expected direction: 45.00 degrees
  Mean actual direction: 45.00 degrees
  Max direction error: 0.0001 degrees
  Flow perpendicular to inlet: ✓ PASS

VELOCITY PROFILE VALIDATION (Log-Law):
  Max relative error: 0.0234 %
  Mean relative error: 0.0123 %
  Mathematical correctness: ✓ PASS

OVERALL VALIDATION: ✓ ALL CHECKS PASSED
```

### 2. Visualization Plots (`validation_plots.png`)

Multi-panel figure showing:
- **Velocity Profile**: Actual vs theoretical with log-law overlay
- **Velocity Error**: Relative error at each height
- **Flow Direction**: Compass plot showing expected vs actual flow direction
- **TKE Profile**: Actual vs theoretical turbulent kinetic energy
- **TKE Error**: Relative error in TKE
- **Ground Elevation Map**: 2D map of terrain elevation at inlet
- **Epsilon Profile**: Actual vs theoretical dissipation rate
- **Epsilon Error**: Relative error in epsilon
- **Validation Summary**: Text summary of all checks

### 3. JSON Data (`validation_results.json`)

Machine-readable validation results for programmatic access:

```json
{
  "ground_elevation": {
    "z_ground_mean": 123.456,
    "z_ground_std": 1.2345,
    "uses_ground_elevation": true
  },
  "flow_direction": {
    "expected_direction_deg": 45.0,
    "mean_direction_deg": 45.0,
    "direction_correct": true,
    "is_horizontal": true
  },
  "velocity_profile": {
    "max_error_percent": 0.0234,
    "mean_error_percent": 0.0123,
    "passes": true
  }
}
```

## Validation Checks

### 1. Logarithmic Velocity Profile

Validates that velocity follows the logarithmic law in the surface layer:

```
u(z) = (u*/κ) * ln(1 + (z-z_ground)/z0)
```

Where:
- `u*` = friction velocity
- `κ` = Von Karman constant (0.40)
- `z` = height above sea level
- `z_ground` = ground elevation at inlet
- `z0` = surface roughness length

**Pass criteria**: Max error < 0.1%

### 2. Ground Elevation Usage

Verifies that:
- Ground elevations from inlet face file are correctly read
- Profiles are calculated relative to actual terrain elevation
- Statistics: min/max/mean/std of ground elevations

**Pass criteria**: Profile minimum height matches mean ground elevation (±10%)

### 3. Flow Direction & Perpendicularity

Checks that:
- Flow is in the specified direction (degrees from x-axis)
- Horizontal velocity components match expected direction
- Vertical velocity component is negligible (flow is horizontal)

**Pass criteria**: 
- Direction error < 0.1°
- Vertical/horizontal ratio < 1e-6

### 4. Turbulent Kinetic Energy (TKE) Profile

Validates TKE profile against theory:

```
k(z) = (Cmu^(-0.5)) * u*^2 * (1 - z/h_bl)^2    for z < 0.99*h_bl
```

Where:
- `Cmu` = turbulence constant (0.033)
- `h_bl` = boundary layer height

**Pass criteria**: Max error < 1%

### 5. Dissipation Rate (Epsilon) Profile

Validates epsilon using the relationship:

```
ε(z) = (Cmu^0.75) * k^1.5 / (κ * (z + z0))    for z < 0.95*h_bl
```

**Pass criteria**: Max error < 1%

## Command Line Options

The standalone validation script supports various options:

```bash
python run_validation.py <case_dir> [options]

Required:
  case_dir              Case directory containing inlet profile files

Optional Configuration:
  --u-star FLOAT       Friction velocity in m/s (default: 0.25)
  --z0 FLOAT           Surface roughness in m, 0=use from file (default: 0.0)
  --h-bl FLOAT         Boundary layer height in m (default: 1500)
  --flow-dir FLOAT     Flow direction in degrees (default: 45)
  --kappa FLOAT        Von Karman constant (default: 0.40)
  --cmu FLOAT          Turbulence constant (default: 0.033)

Options:
  --no-plots           Generate text report only, no visualizations
  --use-internal-faces Use internal faces instead of cell centers
```

## Python API

### Basic Validation

```python
from validate_profiles import run_comprehensive_validation
from config import ABLConfig
from generateBCs import read_inlet_face_file, calculate_multiregion_z_distribution

# Setup
config = ABLConfig()
case_dir = "/path/to/case"

# Read data
inlet_file = f"{case_dir}/0/include/inletFaceInfo.txt"
inlet_data = read_inlet_face_file(inlet_file)
inlet_blocks, mesh_params = inlet_data

# Calculate coordinates
z_coords = calculate_multiregion_z_distribution(
    mesh_params['avg_inlet_height'],
    mesh_params['domain_height'],
    mesh_params,
    use_face_centers=True
)

# Run validation
validation_results = run_comprehensive_validation(
    case_dir, config, inlet_data,
    U_profiles, k_profiles, epsilon_profiles, z_coords,
    generate_plots=True
)
```

### Individual Validation Functions

```python
from validate_profiles import (
    validate_log_law_profile,
    validate_ground_elevation_usage,
    validate_flow_direction,
    validate_tke_profile,
    validate_epsilon_profile
)

# Validate specific aspects
velocity_results = validate_log_law_profile(
    z_coords, u_mag, u_star, z0, kappa, z_ground
)

ground_results = validate_ground_elevation_usage(
    inlet_blocks, z_coords, U_profiles
)

direction_results = validate_flow_direction(
    U_profiles, flow_dir_deg
)
```

### Generate Custom Reports

```python
from validate_profiles import (
    generate_validation_report,
    plot_validation_profiles
)

# Generate text report
report = generate_validation_report(
    validation_results, config, 
    save_path="/path/to/custom_report.txt"
)
print(report)

# Generate custom plots
plot_validation_profiles(
    z_coords, U_profiles, k_profiles, epsilon_profiles,
    validation_results, config, inlet_blocks,
    save_dir="/path/to/output"
)
```

## Interpreting Results

### All Checks Pass (✓)
Your profiles are correctly generated and ready for OpenFOAM simulation. The mathematical formulations are accurate, ground elevations are properly used, and flow direction is correct.

### Some Checks Fail (✗)

1. **Ground Elevation Failure**: 
   - Check that inletFaceInfo.txt contains correct terrain data
   - Verify mesh_params has correct avg_inlet_height

2. **Flow Direction Failure**:
   - Verify flow_dir_deg parameter is correct
   - Check for numerical precision issues in profile calculation

3. **Profile Math Failure**:
   - Review ABL parameters (u*, z0, h_bl)
   - Check for extreme values or edge cases
   - Examine error plots to identify problematic height ranges

## Troubleshooting

### "Inlet face file not found"
Ensure the case was prepared with terrain_following_mesh_generator and contains:
```
<case_dir>/0/include/inletFaceInfo.txt
```

### "Profile files missing"
Run the profile generation first:
```python
from generateBCs import generate_inlet_data_workflow
generate_inlet_data_workflow(case_dir, config)
```

### "Import errors"
Ensure all dependencies are installed:
```bash
pip install numpy matplotlib
```

### Large Errors in Validation
- Check that validation parameters match generation parameters
- Verify z0 setting (0 = use from file, >0 = use constant)
- Review log output for warnings during generation

## Best Practices

1. **Always validate** after generating profiles, especially for new terrain
2. **Review plots** visually to spot unexpected patterns
3. **Check text report** for quantitative metrics
4. **Save validation outputs** with your case for documentation
5. **Use JSON output** for automated testing/CI pipelines
6. **Adjust parameters** if validation fails and regenerate

## Integration with Workflow

Typical workflow:

1. Generate terrain mesh with terrain_following_mesh_generator
2. Generate inlet profiles with generateBCs.py
3. **Validate profiles** (this step - before running OpenFOAM)
4. If validation passes → proceed with OpenFOAM simulation
5. If validation fails → adjust parameters and regenerate

This validation step saves significant time by catching errors before expensive CFD simulations.
