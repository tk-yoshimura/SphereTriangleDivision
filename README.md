# Sphere Triangle Division

A minimal notebook project for dividing a sphere octant (`x, y, z >= 0`) into `N²` spherical triangles.

![octant_n_squared_division](figures/octant_n_squared_division.svg)

Area-optimized result (deterministic tension iterator):

![octant_n_squared_division_with_area_optimizer](figures/octant_n_squared_division_with_area_optimizer.svg)

Convergence and final area distribution:

![octant_n_squared_division_with_area_optimizer_history](figures/octant_n_squared_division_with_area_optimizer_history.svg)

Area-colored spherical triangles (`bwr`, blue=smaller area, red=larger area):

Before optimization:

![octant_mesh_before_area_optimizer_16](figures/octant_mesh_before_area_optimizer_16.svg)

After optimization:

![octant_mesh_after_area_optimizer_16](figures/octant_mesh_after_area_optimizer_16.svg)

## Files

- `sphere_octant_division.ipynb`: main notebook for mesh construction, visualization, and checks.
- `sphere_octant_division_with_area_optimizer.ipynb`: deterministic iterative optimizer to reduce spherical area variance.
- `division_result_viewer.ipynb`: loads saved division JSON and visualizes spherical triangles with area-based coloring (`bwr`), including before/after optimization views.
- `sphere_geometry_util.py`: utility for geodesic arc sampling on the unit sphere.
- `sphere_index_util.py`: shared index utilities (`iter_valid_ij`, `k_from_ij`, validation/count helpers).
- `sphere_division_algorithms.py`: shared mesh/area/constraint/optimizer algorithms used by both notebooks.
- `sphere_division_visualization.py`: shared plotting utilities (octant mesh, history/distribution, before/after comparison).
- `figures/octant_n_squared_division.svg`: generated figure from the notebook example cell (`N=16`).
- `figures/octant_n_squared_division_with_area_optimizer.svg`: before/after mesh comparison on the sphere.
- `figures/octant_n_squared_division_with_area_optimizer_history.svg`: convergence history (`std`, `max_rel_dev`) and optimized area histogram.
- `figures/octant_mesh_before_area_optimizer_16.svg`: area-colored (`bwr`) mesh before optimization.
- `figures/octant_mesh_after_area_optimizer_16.svg`: area-colored (`bwr`) mesh loaded from `results/division_result_16.json`.

## Coordinate Data Format

- Point coordinates are represented as a NumPy array with shape `(N+1, N+1, 3)`.
- Use `positions[i, j]` (or `points[i, j]`) to access xyz.
- Index pairs with `i + j > N` are invalid and stored as `NaN`; they are never used in algorithms.
- Triangle vertex keys are `(i, j)` tuples, and `k` is derived as `k = N - i - j` when needed.

## How to run

1. Open `sphere_octant_division.ipynb`.
2. Run all cells.
3. The example visualization cell writes:
   - `figures/octant_n_squared_division.svg`
4. Open `sphere_octant_division_with_area_optimizer.ipynb`.
5. Run all cells to generate:
   - `figures/octant_n_squared_division_with_area_optimizer.svg`
   - `figures/octant_n_squared_division_with_area_optimizer_history.svg`
6. Open `division_result_viewer.ipynb`.
7. Run all cells to generate:
   - `figures/octant_mesh_before_area_optimizer_16.svg`
   - `figures/octant_mesh_after_area_optimizer_16.svg`

## Area optimization notes

- Initial condition: the octant mesh from `sphere_octant_division.ipynb`.
- Objective: make spherical triangle areas as uniform as possible.
- Update rule (deterministic): each triangle proposes vertex movement based on signed area error from the global mean; vertex updates are averaged over incident triangles.
- Constraints: vertices are projected back to the unit sphere, octant (`x,y,z >= 0`), and boundary geodesic arcs (`x=0`, `y=0`, `z=0`) after each update.
- Interpretation of figures:
  - `...with_area_optimizer.svg`: left is before optimization, right is after optimization.
  - `...with_area_optimizer_history.svg`: left plot shows convergence (`std` and `max_rel_dev`), right plot shows final spherical-area distribution.

## Current example output

- `N = 16`
- triangle count: `256` (`N^2`)
