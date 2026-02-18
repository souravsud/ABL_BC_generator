# ABL Inlet Profile Generation Tool

Tool for generating Atmospheric Boundary Layer (ABL) inlet profiles for OpenFOAM simulations over complex terrain.

## Features

- Generates velocity, TKE, and epsilon profiles for inlet boundaries
- Supports terrain-following meshes with variable ground elevation
- Reads inlet face data from terrain_following_mesh_generator
- Uses spatially-varying surface roughness (z0) from terrain data
- **NEW: Comprehensive validation tools** to verify profile correctness without running OpenFOAM

## Quick Start

```python
from config import ABLConfig
from generateBCs import generate_inlet_data_workflow

config = ABLConfig()
case_dir = "/path/to/your/openfoam/case"

# Generate profiles with validation
results = generate_inlet_data_workflow(
    case_dir, 
    config, 
    validate_profiles=True  # Recommended!
)
```

## Validation Features

The tool now includes comprehensive validation capabilities to verify profile correctness:

✓ **Mathematical Correctness**: Validates logarithmic velocity profile, TKE, and epsilon  
✓ **Ground Elevation**: Verifies profiles use actual terrain elevations  
✓ **Flow Direction**: Checks flow is perpendicular to inlet face  
✓ **Visual Inspection**: Generates detailed comparison plots  
✓ **Numerical Reports**: Provides statistics and error metrics  

### Standalone Validation

```bash
python run_validation.py /path/to/case
```

See [VALIDATION.md](VALIDATION.md) for detailed documentation.

## Requirements

- Python 3.6+
- numpy
- matplotlib

## Usage

See example in `generateBCs.py` main block and detailed validation guide in `VALIDATION.md`.

