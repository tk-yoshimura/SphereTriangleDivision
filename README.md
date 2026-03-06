# Sphere Triangle Division

A minimal notebook project for dividing a sphere octant (`x, y, z >= 0`) into `N²` spherical triangles.

![octant_n_squared_division](figures/octant_n_squared_division.svg)

## Files

- `sphere_octant_division.ipynb`: main notebook for mesh construction, visualization, and checks.
- `sphere_geometry_util.py`: utility for geodesic arc sampling on the unit sphere.
- `figures/octant_n_squared_division.svg`: generated figure from the notebook example cell (`N=16`).

## How to run

1. Open `sphere_octant_division.ipynb`.
2. Run all cells.
3. The example visualization cell writes:
   - `figures/octant_n_squared_division.svg`

## Current example output

- `N = 16`
- triangle count: `256` (`N^2`)
