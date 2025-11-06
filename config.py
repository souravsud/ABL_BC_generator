from dataclasses import dataclass
from typing import Dict


@dataclass
class AtmosphericConfig:
    """Atmospheric boundary layer parameters"""
    u_star: float = 0.25          # friction velocity
    z0: float = 0.0             # Surface roughness (m)- uses roughness map if 0
    h_bl: float = 1500.0        # Boundary layer height (m)
    flow_dir_deg: float = 45.0  # Flow direction (degrees from x-axis)


@dataclass
class TurbulenceConfig:
    """Turbulence model parameters"""
    Cmu: float = 0.033          # Turbulence constant
    kappa: float = 0.40         # Von Karman constant


@dataclass
class MeshConfig:
    """Mesh and boundary configuration"""
    patch_names: Dict[str, str] = None
    
    def __post_init__(self):
        if self.patch_names is None:
            self.patch_names = {
                'inlet': 'inlet',
                'outlet': 'outlet', 
                'ground': 'ground',
                'sky': 'sky',
                'sides': 'sides'
            }


@dataclass
class OpenFOAMConfig:
    """OpenFOAM file generation settings"""
    version: str = "2.0"
    foam_version: str = "v12"
    wall_functions: Dict[str, Dict] = None
    boundary_conditions: Dict[str, Dict[str, Dict]] = None
    
    def __post_init__(self):
        if self.wall_functions is None:
            self.wall_functions = {
                'ground_k': {'type': 'kqRWallFunction', 'value': 0.0},
                'ground_epsilon': {
                    'type': 'epsilonWallFunction',
                    'Cmu': 0.033,
                    'kappa': 0.4, 
                    'E': 9.8,
                    'value': 0.0016
                },
                'ground_nut': {'type': 'nutkAtmRoughWallFunction',
                               'value': 0.0},
            }
        
        # Default boundary conditions
        if self.boundary_conditions is None:
            self.boundary_conditions = {
                'U': {
                    'outlet': {'type': 'zeroGradient'},
                    'ground': {'type': 'noSlip'},
                    'sky': {'type': 'slip'},
                    'sides': {'type': 'slip'}
                    #example-long definitions:'outlet': {'type': 'pressureInletOutletVelocity', 'phi': 'phi', 'value': 'uniform (0 0 0)'}
                },
                'k': {
                    'outlet': {'type': 'zeroGradient'},
                    'sky': {'type': 'slip'},
                    'sides': {'type': 'slip'}
                },
                'epsilon': {
                    'outlet': {'type': 'zeroGradient'},
                    'sky': {'type': 'slip'},
                    'sides': {'type': 'slip'}
                },
                'nut': {
                    'sky': {'type': 'slip'},
                }
            }

@dataclass 
class ABLConfig:
    """Complete configuration for ABL simulation"""
    atmospheric: AtmosphericConfig = None
    turbulence: TurbulenceConfig = None
    mesh: MeshConfig = None
    openfoam: OpenFOAMConfig = None
    
    def __post_init__(self):
        if self.atmospheric is None:
            self.atmospheric = AtmosphericConfig()
        if self.turbulence is None:
            self.turbulence = TurbulenceConfig()
        if self.mesh is None:
            self.mesh = MeshConfig()
        if self.openfoam is None:
            self.openfoam = OpenFOAMConfig()