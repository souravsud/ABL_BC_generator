from dataclasses import dataclass
from typing import Dict


@dataclass
class AtmosphericConfig:
    """Atmospheric boundary layer parameters"""
    Uref: float = 6.87          # Reference wind speed (m/s)
    zref: float = 100.0         # Reference height (m)
    z0: float = 0.1             # Surface roughness (m)
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
    inlet_height: float = 0.0           # Fixed flat inlet height above reference (m)
    domain_height: float = 4000.0       # Sky/top boundary height (m)
    num_cells_z: int = 20               # Number of vertical cells
    expansion_ratio_R: float = 20.0     # Vertical grading expansion ratio
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
    initial_k: float = 0.88
    initial_epsilon: float = 0.0016
    wall_functions: Dict[str, Dict] = None
    boundary_conditions: Dict[str, Dict[str, Dict]] = None  # NEW
    
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
                }
            }
        
        # NEW: Default boundary conditions
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