"""PBR appearances for the marble, the track surfaces, and the props.

The gameplay modules describe materials as plain data (see :mod:`materials`); this
module turns that data into :class:`~OpenGLContext.scenegraph.pbrmaterial.PBRMaterial`
appearances so the demo renders through OpenGLContext's metallic/roughness PBR pass
under image-based lighting — a steel marble mirrors its surroundings, ice reads as a
smooth dielectric, and so on.

Run the demo with ``OPENGLCONTEXT_RENDERER=pbr`` and ``OPENGLCONTEXT_IBL=full``
(``run.py`` sets these) for the reflections to appear; without them the same nodes
still draw, just without the environment reflection.

Floor surfaces additionally carry a tiling **grout normal map** so each 4 m cell
reads as a 4×4 grid of 1 m tiles (see :func:`grout_normal_texture` and
``tools/make_grout_normal.py``).
"""
import os
from typing import Any

import numpy as np
from OpenGL.GL import GL_REPEAT
from OpenGLContext.scenegraph import basenodes
from OpenGLContext.scenegraph.pbrmaterial import PBRMaterial, PBRTexture
from OpenGLContext.scenegraph.pbrmesh import PBRMesh

# Box faces as (unit u-axis, unit v-axis); the outward normal is u × v, so the
# winding (indices 0,1,2,0,2,3 below) is front-facing.  Each face's UVs are baked
# in metres, so a REPEAT-wrapped texture tiles once per metre with no uv_transform.
_BOX_FACES = (
    ((1, 0, 0), (0, 0, -1)),   # +Y top
    ((1, 0, 0), (0, 0, 1)),    # -Y bottom
    ((0, 0, -1), (0, 1, 0)),   # +X
    ((0, 0, 1), (0, 1, 0)),    # -X
    ((1, 0, 0), (0, 1, 0)),    # +Z
    ((-1, 0, 0), (0, 1, 0)),   # -Z
)

_ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")
_GROUT_NORMAL_PATH = os.path.join(_ASSET_DIR, "grout_normal.png")
_GROUT_COLOR_PATH = os.path.join(_ASSET_DIR, "grout_color.png")

_grout_cache: dict[tuple, object] = {}
_surface_cache: dict[tuple, object] = {}


def marble_material_node(marble: Any) -> Any:
    """A PBRMaterial for a marble: metals reflect; glass is a smooth dielectric.

    Per the plan's resolved decision, glass is approximated as a very smooth
    dielectric (high specular, low roughness) rather than true transmission, which
    keeps a fast-moving marble cheap while still reading as glass under IBL.
    """
    kw = dict(baseColor=tuple(marble.base_color), metallic=marble.metallic,
              roughness=marble.roughness)
    if marble.transmission > 0:
        kw.update(metallic=0.0, roughness=0.04, ior=1.5, specular=1.0)
    return PBRMaterial(**kw)


def marble_appearance(marble: Any) -> Any:
    return basenodes.Appearance(material=marble_material_node(marble))


def surface_appearance(surface: Any, grout: Any=False, uv_scale: Any=4.0) -> Any:
    """A shared PBRMaterial appearance for a track surface.

    Appearances are cached and shared across all tiles of a surface (they are
    immutable), so a large level is not thousands of distinct material nodes.  With
    ``grout`` set and the grout asset present, a tiling normal map is applied at
    ``uv_scale`` repeats per cell (1 m tiles across a 4 m cell).
    """
    key = (surface.name, grout, uv_scale)
    if key in _surface_cache:
        return _surface_cache[key]
    kw = dict(baseColor=tuple(surface.base_color), metallic=surface.metallic,
              roughness=surface.roughness)
    if grout:
        # Tile geometry bakes UVs in metres, so REPEAT-wrapped maps tile at 1 m with
        # no uv_transform.  The colour grid (grout lines darkening baseColor) makes
        # the 1 m tiling clearly visible under any lighting; the normal map adds the
        # groove as a real bump.  baseColor stays the surface tint — the sRGB grid
        # texture multiplies it.
        textures = {}
        color = _grout_texture("grout_color", _GROUT_COLOR_PATH, srgb=True)
        normal = _grout_texture("grout_normal", _GROUT_NORMAL_PATH, srgb=False)
        if color is not None:
            textures["baseColor"] = color
        if normal is not None:
            textures["normal"] = normal
            kw["normalScale"] = 2.5
        if textures:
            kw["textures"] = textures
    appearance = basenodes.Appearance(material=PBRMaterial(**kw))
    _surface_cache[key] = appearance
    return appearance


def tile_mesh(size: Any, meters_per_tile: Any=1.0) -> Any:
    """A box :class:`PBRMesh` with per-metre UVs and tangents, for grouted tiles.

    Unlike the ``Box`` primitive, this supplies the tangent attribute the PBR normal
    mapping needs, and bakes UVs in metres so the grout map tiles once per metre.
    The mesh carries no material; wrap it in a Shape with a surface appearance.
    """
    hx, hy, hz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
    half = np.array([hx, hy, hz])
    positions: list[Any] = []
    normals: list[Any] = []
    texcoords: list[Any] = []
    tangents: list[Any] = []
    indices: list[Any] = []
    for u_dir, v_dir in _BOX_FACES:
        u = np.array(u_dir, dtype="d")
        v = np.array(v_dir, dtype="d")
        normal = np.cross(u, v)
        su = float(abs(np.dot(u, 2 * half)))          # face width along u (metres)
        sv = float(abs(np.dot(v, 2 * half)))          # face height along v
        center = normal * float(abs(np.dot(normal, half)))
        base = len(positions)
        for a, b in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)):
            positions.append(center + a * u * su + b * v * sv)
            normals.append(normal)
            texcoords.append(((a + 0.5) * su / meters_per_tile,
                              (b + 0.5) * sv / meters_per_tile))
            tangents.append((u[0], u[1], u[2], 1.0))
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])
    return PBRMesh(positions=np.array(positions, "f"), normals=np.array(normals, "f"),
                   texcoords=np.array(texcoords, "f"), tangents=np.array(tangents, "f"),
                   indices=np.array(indices, "uint32"))


def color_appearance(color: Any, metallic: Any=0.0, roughness: Any=0.5) -> Any:
    """A one-off PBR appearance for a prop (bumper, ramp, marker)."""
    return basenodes.Appearance(
        material=PBRMaterial(baseColor=tuple(color), metallic=metallic, roughness=roughness))


def _grout_texture(key: Any, path: str, srgb: Any) -> Any:
    """Load a tiling grout map as a REPEAT-wrapped :class:`PBRTexture` (cached).

    The assets are produced by ``tools/make_grout.py``; if they have not been
    generated the floor simply renders without them (no crash).
    """
    if key in _grout_cache:
        return _grout_cache[key]
    try:
        # Pillow is not a dependency: without it the floor renders without grout.
        from PIL import Image  # noqa: PLC0415 Pillow is optional
        image = Image.open(path).convert("RGB")
        texture = PBRTexture(image, srgb=srgb, wrap_s=GL_REPEAT, wrap_t=GL_REPEAT)
    except (ImportError, OSError):
        # No PIL, no GL, or the asset has not been generated: the floor
        # renders without the grout maps.
        texture = None
    _grout_cache[key] = texture
    return texture
