from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class AtmosphericConfig:
    """Atmospheric boundary layer parameters"""
    u_star: float = 0.25              # friction velocity (m/s); overridden if U_ref/z_ref provided
    z0: float = 0.0                   # Surface roughness (m) - uses roughness map if 0
    h_bl: float = 1500.0             # Boundary layer height (m); set to 0 to use full (un-truncated) Richards & Hoxey log law
    wind_dir_met: float = 225.0      # Meteorological wind direction (degrees):
                                      # 0=FROM north, 90=FROM east, 180=FROM south, 270=FROM west
    U_ref: Optional[float] = None    # Reference wind speed (m/s); if set with z_ref, u_star is derived
    z_ref: Optional[float] = None    # Reference height (m) for U_ref


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
    version: str = "1.0"
    foam_version: str = "v2512"
    wall_functions: Dict[str, Dict] = None
    boundary_conditions: Dict[str, Dict[str, Dict]] = None
    
    def __post_init__(self):
        if self.wall_functions is None:
            self.wall_functions = {
                'ground_k': {'type': 'kqRWallFunction', 'value': 0.0},
                'ground_epsilon': {
                    'type': 'atmEpsilonWallFunction',
                    'value': 0.0016
                },
                'ground_nut': {'type': 'atmNutkWallFunction',
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