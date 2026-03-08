# ABL BC Generator

A Python tool that generates OpenFOAM boundary condition files for **Atmospheric Boundary Layer (ABL)** simulations. It reads mesh inlet face data from your case directory and writes ready-to-use `0/U`, `0/k`, `0/epsilon`, and `0/nut` files, together with the non-uniform inlet data files (`inletU`, `inletK`, `inletEpsilon`) and an `initialConditions` include file.

Profiles are based on the Richards & Hoxey (1993) log-law with optional boundary-layer height truncation.

---

## Requirements

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

---

## Case Directory Prerequisites

Before running the generator, your OpenFOAM case directory must contain the inlet face description file at:

```
<case_dir>/
└── 0/
    └── include/
        └── inletFaceInfo.txt   ← required input
```

This is tool was intended to be used with the terrain following mesh tool [terrain_following_mesh_generator](https://github.com/souravsud/terrain_following_mesh_generator). The `inletFaceInfo.txt` is produced by a mesh pre-processing step in the mentioned repo . It contains:

- **Mesh parameters** (`domain_height`, `total_z_cells`, `z_grading`, `avg_inlet_height`, `first_cell_height`, `z0_eff_atInlet`, …)
- **Face data** — one row per inlet column with `block_i, block_j, x_ground, y_ground, z_ground[, z0]`

---

## Usage

### Basic call (all defaults)

```bash
python generateBCs.py /path/to/my/openfoam/case
```

This uses default atmospheric conditions (`u_star = 0.25 m/s`, `wind_dir_met = 225°`, `h_bl = 1500 m`) and writes all boundary condition files into `<case_dir>/0/`.

### Disable profile plot

```bash
python generateBCs.py /path/to/case --no-plot
```

### Enable verbose logging

```bash
python generateBCs.py /path/to/case --verbose
```

### Python API — custom configuration

You can import and call the workflow directly from your own script to customise every parameter:

```python
from generateBCs import generate_inlet_data_workflow
from config import ABLConfig, AtmosphericConfig, TurbulenceConfig, MeshConfig

# Example 1 -- specify u_star directly
config = ABLConfig(
    atmospheric=AtmosphericConfig(
        u_star=0.35,          # friction velocity (m/s)
        z0=0.0,               # use z0 map from inletFaceInfo.txt
        h_bl=1200.0,          # boundary layer height (m)
        wind_dir_met=270.0,   # wind FROM west (meteorological degrees)
    )
)

results = generate_inlet_data_workflow(
    case_dir="/path/to/case",
    config=config,
    plot_profiles=True,
    verbose=False,
)

# Example 2 -- derive u_star from a reference wind speed
config = ABLConfig(
    atmospheric=AtmosphericConfig(
        U_ref=8.0,            # reference wind speed (m/s)
        z_ref=100.0,          # reference height (m)
        wind_dir_met=225.0,
    )
)

results = generate_inlet_data_workflow("/path/to/case", config)
```

`results` is a dictionary containing:

| Key | Description |
|-----|-------------|
| `U_profiles` | `(N, 3)` array of velocity vectors at each inlet face |
| `k_profiles` | `(N,)` array of TKE values |
| `epsilon_profiles` | `(N,)` array of dissipation-rate values |
| `z_coords` | Height coordinates used for the profiles |
| `z0_mean` | Mean surface roughness used for the inlet |
| `config` | The `ABLConfig` object that was applied |

---

## Configuration Reference

### `AtmosphericConfig`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `u_star` | `0.25` | Friction velocity (m/s). Overridden when `U_ref`/`z_ref` are set. |
| `z0` | `0.0` | Surface roughness (m). Set to `0` to use the roughness map from `inletFaceInfo.txt`. |
| `h_bl` | `1500.0` | Boundary layer height (m). Set to `0` for an untruncated Richards & Hoxey log-law. |
| `wind_dir_met` | `225.0` | Meteorological wind direction in degrees (wind *from*: 0 = N, 90 = E, 180 = S, 270 = W). |
| `U_ref` | `None` | Reference wind speed (m/s). If provided together with `z_ref`, `u_star` is derived automatically. |
| `z_ref` | `None` | Reference height (m) corresponding to `U_ref`. |

### `TurbulenceConfig`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Cmu` | `0.033` | Turbulence model constant. |
| `kappa` | `0.40` | Von Kármán constant. |

### `MeshConfig`

Patch names default to `inlet`, `outlet`, `ground`, `sky`, `sides`. Override any name:

```python
from config import MeshConfig
mesh = MeshConfig(patch_names={
    'inlet':  'inletPatch',
    'outlet': 'outletPatch',
    'ground': 'terrainPatch',
    'sky':    'topPatch',
    'sides':  'symmetryPatch',
})
```

---

## Output Files

After a successful run the following files are written inside your case directory:

```
<case_dir>/
└── 0/
    ├── U
    ├── k
    ├── epsilon
    ├── nut
    └── include/
        ├── inletU
        ├── inletK
        ├── inletEpsilon
        └── initialConditions
```

An optional plot (`inlet_profiles.png`) is also saved in `<case_dir>/` when `--plot` is active (the default).

---

## Sample Output Profiles

Untruncated  profiles (h_bl = 0):
<img width="4467" height="1768" alt="inlet_profiles" src="https://github.com/user-attachments/assets/1d6c0d6a-4186-4acc-a986-d333a33f8723" />

Truncated profiles (h_bl = 1500):
<img width="4464" height="1768" alt="inlet_profiles" src="https://github.com/user-attachments/assets/c652587a-fd8d-49f6-8724-8a19c866e26d" />

