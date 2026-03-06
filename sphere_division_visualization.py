from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from sphere_division_algorithms import build_octant_mesh
from sphere_geometry_util import geodesic_arc


def _add_octant_surface(ax, alpha=0.18, resolution=60):
    th = np.linspace(0.0, np.pi / 2.0, resolution)
    ph = np.linspace(0.0, np.pi / 2.0, resolution)
    th_grid, ph_grid = np.meshgrid(th, ph)
    x = np.sin(th_grid) * np.cos(ph_grid)
    y = np.sin(th_grid) * np.sin(ph_grid)
    z = np.cos(th_grid)
    ax.plot_surface(x, y, z, color="lightsteelblue", alpha=alpha, linewidth=0, antialiased=True)


def _style_octant_axes(ax, title, add_axes=True):
    if add_axes:
        ax.plot([0, 1.1], [0, 0], [0, 0], color="r", linewidth=1.2)
        ax.plot([0, 0], [0, 1.1], [0, 0], color="g", linewidth=1.2)
        ax.plot([0, 0], [0, 0], [0, 1.1], color="b", linewidth=1.2)
        ax.text(1.12, 0, 0, "x")
        ax.text(0, 1.12, 0, "y")
        ax.text(0, 0, 1.12, "z")

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_zlim(0, 1.05)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=28, azim=38)
    ax.set_title(title)


def save_figure(fig, save_path):
    if not save_path:
        return
    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg", bbox_inches="tight")
    print(f"saved figure to: {output.as_posix()}")


def plot_octant_division(n=6, sphere_alpha=0.18, edge_lw=0.8, figsize=(8, 8), save_path=None):
    _, _, tris = build_octant_mesh(n)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    _add_octant_surface(ax, alpha=sphere_alpha, resolution=60)
    poly = Poly3DCollection(tris, facecolors="cornflowerblue", edgecolors="none", alpha=0.25)
    ax.add_collection3d(poly)

    for tri in tris:
        for e0, e1 in ((0, 1), (1, 2), (2, 0)):
            arc = geodesic_arc(tri[e0], tri[e1], samples=20)
            ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color="k", linewidth=edge_lw, alpha=0.75)

    _style_octant_axes(ax, title=f"Octant spherical-triangle division: N={n}, count={len(tris)}", add_axes=True)
    plt.tight_layout()
    save_figure(fig, save_path)
    plt.show()
    return tris


def plot_planar_area_distribution(areas, n, figsize=(7, 4)):
    bins = min(20, max(5, int(np.sqrt(areas.size))))
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(areas, bins=bins, color="steelblue", edgecolor="black", alpha=0.85)
    ax.axvline(areas.mean(), color="crimson", linestyle="--", linewidth=1.5, label=f"mean={areas.mean():.6f}")
    ax.axvline(
        np.median(areas),
        color="darkgreen",
        linestyle=":",
        linewidth=1.5,
        label=f"median={np.median(areas):.6f}",
    )
    ax.set_title(f"Planar Triangle Area Distribution (N={n}, count={areas.size})")
    ax.set_xlabel("Triangle area")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_optimizer_history_and_distribution(hist, areas_eq, n, save_path=None, figsize=(12, 4)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    line_std = ax1.plot(hist[:, 0], hist[:, 2], color="tab:blue", lw=1.8, label="std")[0]
    ax1.set_title("Convergence history")
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("std", color="tab:blue")
    ax1.set_ylim([0, np.max(hist[:, 2]) * 1.1])
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(alpha=0.3)

    ax1r = ax1.twinx()
    line_max = ax1r.plot(hist[:, 0], hist[:, 3], color="tab:red", lw=1.8, label="max_rel_dev")[0]
    ax1r.set_ylabel("max_rel_dev", color="tab:red")
    ax1r.set_ylim([0, np.max(hist[:, 3]) * 1.1])
    ax1r.tick_params(axis="y", labelcolor="tab:red")
    ax1.legend([line_std, line_max], ["std", "max_rel_dev"], loc="upper right")

    bins = min(20, max(5, int(np.sqrt(areas_eq.size))))
    ax2.hist(areas_eq, bins=bins, color="teal", edgecolor="black", alpha=0.85)
    ax2.axvline(areas_eq.mean(), color="crimson", linestyle="--", linewidth=1.5, label=f"mean={areas_eq.mean():.6f}")
    ax2.set_title(f"Spherical area distribution (N={n})")
    ax2.set_xlabel("spherical area")
    ax2.set_ylabel("count")
    ax2.grid(alpha=0.25)
    ax2.legend()

    plt.tight_layout()
    save_figure(fig, save_path)
    plt.show()
    return fig, (ax1, ax2)


def plot_before_after_mesh_comparison(
    triangle_keys,
    positions_before,
    positions_after,
    n,
    save_path=None,
    figsize=(14, 6),
):
    tris_before = [np.array([positions_before[k] for k in tri]) for tri in triangle_keys]
    tris_after = [np.array([positions_after[k] for k in tri]) for tri in triangle_keys]

    fig = plt.figure(figsize=figsize)
    ax_l = fig.add_subplot(121, projection="3d")
    ax_r = fig.add_subplot(122, projection="3d")

    for ax, tris, title in (
        (ax_l, tris_before, f"Before optimization (N={n})"),
        (ax_r, tris_after, f"After optimization (N={n})"),
    ):
        _add_octant_surface(ax, alpha=0.18, resolution=50)
        poly = Poly3DCollection(tris, facecolors="cornflowerblue", edgecolors="none", alpha=0.28)
        ax.add_collection3d(poly)
        for tri in tris:
            for e0, e1 in ((0, 1), (1, 2), (2, 0)):
                arc = geodesic_arc(tri[e0], tri[e1], samples=16)
                ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color="k", linewidth=0.6, alpha=0.7)
        _style_octant_axes(ax, title=title, add_axes=False)

    plt.tight_layout()
    save_figure(fig, save_path)
    plt.show()
    return fig, (ax_l, ax_r)


def plot_octant_mesh_from_positions(
    triangle_keys,
    positions,
    n,
    sphere_alpha=0.18,
    edge_lw=0.8,
    figsize=(8, 8),
    save_path=None,
):
    tris = [np.array([positions[k] for k in tri]) for tri in triangle_keys]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    _add_octant_surface(ax, alpha=sphere_alpha, resolution=60)
    poly = Poly3DCollection(tris, facecolors="cornflowerblue", edgecolors="none", alpha=0.25)
    ax.add_collection3d(poly)

    for tri in tris:
        for e0, e1 in ((0, 1), (1, 2), (2, 0)):
            arc = geodesic_arc(tri[e0], tri[e1], samples=20)
            ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color="k", linewidth=edge_lw, alpha=0.75)

    _style_octant_axes(ax, title=f"Octant spherical-triangle mesh from file: N={n}, count={len(tris)}", add_axes=True)
    plt.tight_layout()
    save_figure(fig, save_path)
    plt.show()
    return fig, ax
