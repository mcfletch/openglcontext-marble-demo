"""Tests for the marble/surface material table and its physics registration.

A material has two faces the demo must keep consistent: a *look* (PBR
metallic/roughness/colour) and a *feel* (friction/restitution/mass).  On top of
that, how much a marble's spin becomes motion is set by the **pairwise** friction
between a given marble and a given surface, which we register into the world's
pairwise-friction override.
"""
from omi_physics.world import PhysicsWorld

from openglcontext_marble_demo import materials


def test_every_marble_and_surface_is_defined():
    assert set(materials.MARBLES) >= {"steel", "chrome", "glass", "rubber", "wood", "ice"}
    assert set(materials.SURFACES) >= {"stone", "metal", "rubber_pad", "ice_sheet"}


def test_steel_is_metallic_and_heavy_rubber_is_not():
    steel = materials.MARBLES["steel"]
    rubber = materials.MARBLES["rubber"]
    assert steel.metallic == 1.0
    assert rubber.metallic == 0.0
    assert steel.mass > rubber.mass          # steel marble is heavier


def test_glass_has_transmission():
    assert materials.MARBLES["glass"].transmission > 0.0


def test_physics_material_maps_friction_and_restitution():
    mat = materials.physics_material(materials.MARBLES["rubber"])
    assert mat.dynamicFriction == materials.MARBLES["rubber"].dynamic_friction
    assert mat.restitution == materials.MARBLES["rubber"].restitution


def test_register_materials_returns_index_per_name():
    world = PhysicsWorld()
    index = materials.register_materials(world)
    for name in list(materials.MARBLES) + list(materials.SURFACES):
        assert name in index
        assert isinstance(index[name], int)


def test_pairwise_friction_registered_rubber_grips_more_than_steel_on_ice():
    world = PhysicsWorld()
    index = materials.register_materials(world)
    materials.apply_pair_frictions(world, index)

    rubber_ice = world.pair_friction_for(index["rubber"], index["ice_sheet"])
    steel_ice = world.pair_friction_for(index["steel"], index["ice_sheet"])
    assert rubber_ice is not None and steel_ice is not None
    # Rubber grips ice far better than steel does.
    assert rubber_ice[1] > steel_ice[1]


def test_pairwise_friction_rubber_grips_stone_more_than_ice():
    world = PhysicsWorld()
    index = materials.register_materials(world)
    materials.apply_pair_frictions(world, index)
    on_stone = world.pair_friction_for(index["rubber"], index["stone"])
    on_ice = world.pair_friction_for(index["rubber"], index["ice_sheet"])
    assert on_stone[1] > on_ice[1]
