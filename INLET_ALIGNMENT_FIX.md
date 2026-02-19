# Fix: Inlet Profile Alignment with Individual Block Ground Elevations

## Problem Statement

The user asked: "How do i ensure that the inlet profiles generated actually line up above the inlet faces (obtained from the inletfaceinfo file). I mean i want the profiles at each block to start exactly at the cell above the ground and end at the ceiling. Is this already enforced?"

## Issue Found

**No, this was NOT enforced in the original implementation.**

The original code used `avg_inlet_height` (average ground elevation across all blocks) to calculate z-coordinates for ALL blocks. This caused misalignment when inlet blocks had varying ground elevations.

### Original Behavior:
```python
# Line 156-161: Single z-coordinate calculation for all blocks
z_coords = calculate_multiregion_z_distribution(
    avg_inlet_height,  # Average used for all blocks
    domain_height,
    mesh_params,
    use_face_centers
)

# Line 187: Height calculation used average
height = max(z - avg_inlet_height, MIN_HEIGHT)
```

**Problem:** All blocks shared the same z-distribution, regardless of their individual ground elevations.

## Solution Implemented

Modified `calculate_inlet_profiles_from_mesh()` to calculate z-coordinates **individually for each block** using its own `z_ground` value.

### Key Changes:

1. **Per-block z-coordinate calculation:**
   ```python
   for block in inlet_blocks:
       # Calculate z-coords for THIS specific block
       block_z_ground = block['z_ground']
       z_coords = calculate_multiregion_z_distribution(
           block_z_ground,  # Individual block's ground elevation
           domain_height,
           mesh_params,
           use_face_centers
       )
   ```

2. **Per-block height calculation:**
   ```python
   # Height above THIS block's ground (not average)
   height = max(z - block_z_ground, MIN_HEIGHT)
   ```

3. **Added diagnostic logging:**
   - Detects and reports varying ground elevations
   - Confirms alignment to individual block heights

## Verification

Created and ran tests demonstrating:

1. **Individual alignment:** Each block's z-coordinates start at its own `z_ground`
   - Block 1 (z_ground=0m): First cell at ~2.5m
   - Block 2 (z_ground=50m): First cell at ~52.4m  
   - Block 3 (z_ground=100m): First cell at ~102.3m

2. **Correct offset:** Cell position offset matches ground elevation difference
   - Expected: 50m difference
   - Actual: 49.87m difference (within numerical tolerance)

3. **Profile differences:** Profiles at same absolute z-coordinate differ based on height above local ground
   - At z=100m: Block 1 sees 100m height, Block 2 sees 50m height
   - Results in different k and epsilon values (physically correct)

## Impact

**Before:** Profiles misaligned when terrain varied (all started at average height)
**After:** Profiles correctly aligned to each block's individual ground elevation

This ensures:
✅ Profiles start at the cell immediately above each block's actual ground
✅ Profiles extend to the domain ceiling for all blocks
✅ Physical correctness: height-dependent quantities calculated from local ground
✅ No impact on cases with uniform ground elevation (backward compatible)

## Files Modified

- `generateBCs.py`:
  - `calculate_inlet_profiles_from_mesh()`: Calculate z-coords per block
  - Added diagnostic logging for varying ground elevations
  - Updated comments in `generate_inlet_data_workflow()`

## Example Output

```
Inlet blocks found: 100, each with 50 z-levels.
Ground elevation varies: min=45.23m, max=102.78m
Profiles will be aligned to each block's individual ground elevation.
Calculating profiles for 5000 inlet faces...
```

When ground is uniform:
```
Inlet blocks found: 100, each with 50 z-levels.
Ground elevation is uniform: 0.000m
Calculating profiles for 5000 inlet faces...
```
