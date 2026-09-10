"""The marble and surface material table — look, feel, and pairwise grip.

Each material carries two coupled descriptions:

* a **look**: PBR metallic / roughness / base colour (and transmission for glass),
  fed to a :class:`~OpenGLContext.scenegraph.pbrmaterial.PBRMaterial`; and
* a **feel**: friction / restitution (and, for marbles, mass), fed to an OMI
  :class:`~omi_physics.model.Material`.

On top of the per-material feel sits the **pairwise friction table**
(:data:`PAIR_FRICTION`): the coefficient that actually governs how much of a
marble's spin turns into motion depends on *which* marble is on *which* surface —
rubber grips stone but slips on ice, independently of steel.  A single
material-combine rule cannot express that, so we register these pairings into the
world's pairwise-friction override (see
:meth:`omi_physics.world.PhysicsWorld.set_pair_friction`).
"""
from dataclasses import dataclass
from typing import Any

from omi_physics import model


@dataclass(frozen=True)
class MarbleMaterial:
    """A marble the player can roll: PBR look + rigid-body feel."""
    name: str
    base_color: tuple            # linear RGB
    metallic: float
    roughness: float
    static_friction: float
    dynamic_friction: float
    restitution: float
    mass: float
    transmission: float = 0.0    # >0 for glass (screen-space transmission)


@dataclass(frozen=True)
class SurfaceMaterial:
    """A track surface: PBR look + rigid-body feel (mass is irrelevant, it's static)."""
    name: str
    base_color: tuple
    metallic: float
    roughness: float
    static_friction: float
    dynamic_friction: float
    restitution: float


MARBLES = {
    "steel":  MarbleMaterial("steel",  (0.56, 0.57, 0.58), 1.0, 0.15,
                             0.40, 0.30, 0.15, mass=3.0),
    "chrome": MarbleMaterial("chrome", (0.55, 0.56, 0.57), 1.0, 0.03,
                             0.35, 0.28, 0.10, mass=3.0),
    "glass":  MarbleMaterial("glass",  (0.85, 0.90, 0.92), 0.0, 0.02,
                             0.30, 0.25, 0.20, mass=1.2, transmission=0.9),
    "rubber": MarbleMaterial("rubber", (0.10, 0.10, 0.12), 0.0, 0.70,
                             1.00, 0.90, 0.80, mass=1.8),
    "wood":   MarbleMaterial("wood",   (0.52, 0.34, 0.18), 0.0, 0.80,
                             0.50, 0.40, 0.20, mass=1.0),
    "ice":    MarbleMaterial("ice",    (0.70, 0.82, 0.90), 0.0, 0.10,
                             0.05, 0.02, 0.05, mass=1.5),
}

SURFACES = {
    "stone":      SurfaceMaterial("stone",      (0.45, 0.44, 0.42), 0.0, 0.85,
                                  0.80, 0.70, 0.10),
    "metal":      SurfaceMaterial("metal",      (0.40, 0.41, 0.43), 0.9, 0.35,
                                  0.45, 0.35, 0.10),
    "rubber_pad": SurfaceMaterial("rubber_pad", (0.15, 0.16, 0.20), 0.0, 0.65,
                                  1.00, 0.90, 0.85),
    "ice_sheet":  SurfaceMaterial("ice_sheet",  (0.72, 0.84, 0.92), 0.0, 0.08,
                                  0.06, 0.03, 0.05),
}

# Dynamic friction coefficient per (marble, surface).  This is the number the
# solver uses at a marble↔surface contact, and thus the dial that makes each
# marble feel distinct on each surface.  Values mirror plans/MARBLE-MADNESS-DEMO.md.
_PAIR_DYNAMIC = {
    #            stone  metal  rubber_pad  ice_sheet
    "steel":   (0.35,  0.30,  0.55,       0.03),
    "chrome":  (0.30,  0.28,  0.50,       0.02),
    "glass":   (0.28,  0.25,  0.45,       0.04),
    "rubber":  (0.90,  0.75,  1.10,       0.15),
    "wood":    (0.50,  0.45,  0.70,       0.08),
    "ice":     (0.10,  0.08,  0.20,       0.02),
}
_SURFACE_ORDER = ("stone", "metal", "rubber_pad", "ice_sheet")

# Static friction sits a little above dynamic (harder to start sliding than to keep
# sliding); the exact gap is not gameplay-critical, so we derive it uniformly.
_STATIC_BONUS = 0.1

PAIR_FRICTION = {
    (marble, surface): (dyn + _STATIC_BONUS, dyn)
    for marble, row in _PAIR_DYNAMIC.items()
    # strict: every row of the grid carries one entry per surface, and a row
    # that lost or gained one would otherwise be silently truncated here.
    for surface, dyn in zip(_SURFACE_ORDER, row, strict=True)
}


def physics_material(material: Any) -> Any:
    """OMI :class:`~omi_physics.model.Material` for a marble/surface."""
    return model.Material(
        staticFriction=material.static_friction,
        dynamicFriction=material.dynamic_friction,
        restitution=material.restitution,
    )


def register_materials(world: Any) -> Any:
    """Add every marble + surface material to ``world``; return ``{name: index}``."""
    index = {}
    for name, material in list(MARBLES.items()) + list(SURFACES.items()):
        index[name] = world.add_material(physics_material(material))
    return index


def apply_pair_frictions(world: Any, index: Any) -> None:
    """Install the pairwise marble↔surface friction overrides into ``world``.

    ``index`` is the ``{name: material_index}`` map from :func:`register_materials`.
    """
    for (marble, surface), (static_mu, dynamic_mu) in PAIR_FRICTION.items():
        world.set_pair_friction(index[marble], index[surface], static_mu, dynamic_mu)
