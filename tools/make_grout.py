#!/usr/bin/env python
"""Generate the tiling grout textures used on the floor tiles.

Two maps, both one 1 m tile (the floor material samples them once per metre across a
4 m cell, so a cell reads as a 4×4 grid of tiles):

* ``grout_color.png`` — a base-colour (albedo) grid: a light tile face with darker
  grout lines around the border.  Multiplied by each surface's colour, this makes
  the 1 m grid clearly visible under any lighting.
* ``grout_normal.png`` — a tangent-space normal map: the same border as a recessed
  groove, so the grout also catches the light as a real bump.

This is a committed, reproducible asset pipeline (not a throwaway script).  Re-run to
regenerate both maps into ``src/openglcontext_marble_demo/assets/``.  The maps are
CC0 (procedurally generated, no external inputs).

    python tools/make_grout.py
    python tools/make_grout.py --size 512 --grout 0.08
"""
import argparse
import os

import numpy as np
from PIL import Image

_ASSETS = os.path.join(os.path.dirname(__file__), "..", "src",
                       "openglcontext_marble_demo", "assets")


def _edge_distance(size):
    """For each texel, distance to the nearest tile edge, in [0, 0.5]."""
    coord = np.linspace(0.0, 1.0, size, endpoint=False)
    u, v = np.meshgrid(coord, coord)
    return np.minimum(np.minimum(u, 1.0 - u), np.minimum(v, 1.0 - v))


def grout_mask(size, grout_frac):
    """Smooth 0→1 mask: 0 deep in the grout groove, 1 on the flat tile face."""
    return np.clip(_edge_distance(size) / max(grout_frac, 1e-6), 0.0, 1.0)


def color_map(size, grout_frac, grout_value):
    """Light tile face, darker grout lines (grayscale RGB, sRGB)."""
    mask = grout_mask(size, grout_frac)
    value = grout_value + (1.0 - grout_value) * mask     # grout_value..1
    rgb = np.stack([value] * 3, axis=-1)
    return (rgb * 255.0).clip(0, 255).astype("uint8")


def normal_map(size, grout_frac, depth, strength):
    """Tangent-space normal map from the grout groove height field."""
    height = 1.0 - depth * (1.0 - grout_mask(size, grout_frac))
    dz_dx = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) * 0.5
    dz_dy = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) * 0.5
    nx, ny, nz = -dz_dx * strength, -dz_dy * strength, np.ones_like(height)
    length = np.sqrt(nx * nx + ny * ny + nz * nz)
    rgb = np.stack([nx / length * 0.5 + 0.5,
                    ny / length * 0.5 + 0.5,
                    nz / length * 0.5 + 0.5], axis=-1)
    return (rgb * 255.0).clip(0, 255).astype("uint8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=256, help="tile resolution (px)")
    parser.add_argument("--grout", type=float, default=0.07,
                        help="grout line width as a fraction of the tile")
    parser.add_argument("--grout-value", type=float, default=0.4,
                        help="brightness of the grout lines (0..1)")
    parser.add_argument("--depth", type=float, default=0.8, help="groove depth (0..1)")
    parser.add_argument("--strength", type=float, default=6.0, help="normal strength")
    parser.add_argument("--assets", default=os.path.normpath(_ASSETS))
    args = parser.parse_args(argv)

    os.makedirs(args.assets, exist_ok=True)
    color = color_map(args.size, args.grout, args.grout_value)
    normal = normal_map(args.size, args.grout, args.depth, args.strength)
    Image.fromarray(color, "RGB").save(os.path.join(args.assets, "grout_color.png"))
    Image.fromarray(normal, "RGB").save(os.path.join(args.assets, "grout_normal.png"))
    print(f"wrote grout_color.png + grout_normal.png ({args.size}px) to {args.assets}")


if __name__ == "__main__":
    main()
