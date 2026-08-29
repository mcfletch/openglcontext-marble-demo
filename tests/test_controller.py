"""Tests for :class:`MarbleController` — steering, grounding, fall/respawn.

These are the game rules that turn the passive physics of a rolling sphere into
Marble Madness: spin-to-steer, keeping track of the last square the marble was
grounded on, telling a *jump* apart from a *fall*, and respawning after a penalty
delay.  Everything is exercised against the real physics world (no window), so the
steering tests prove actual spin→motion, not just that a method was called.
"""
import numpy as np
from omi_physics import model
from omi_physics.world import PhysicsWorld

from openglcontext_marble_demo import materials
from openglcontext_marble_demo.controller import MarbleController
from openglcontext_marble_demo.track import TrackMap


def _world_with_marble(marble="rubber", surface="rubber_pad",
                       cells=None, start=(0.0, 0.6, 0.0), radius=0.5):
    """A world with one grippy marble on a flat track of ``cells``."""
    if cells is None:
        cells = {(0, 0): 0.0, (1, 0): 0.0, (2, 0): 0.0, (3, 0): 0.0}
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)),
                         sleep_enabled=False)
    index = materials.register_materials(world)
    materials.apply_pair_frictions(world, index)

    # A single wide static floor under the cells (physics ground); the TrackMap is
    # the logical track used for grounding/fall decisions.
    ground = world.add_shape(model.Shape.box((40, 1, 40)))
    world.add_body(model.Motion(type=model.STATIC),
                   collider=model.Collider(shape=ground,
                                           physicsMaterial=index[surface]),
                   position=(6, -0.5, 0))
    ball = world.add_shape(model.Shape.sphere(radius))
    i = world.add_body(model.Motion(type=model.DYNAMIC, mass=materials.MARBLES[marble].mass),
                       collider=model.Collider(shape=ball, physicsMaterial=index[marble]),
                       position=start)
    track = TrackMap(cells, cell_size=4.0)
    ctrl = MarbleController(world, i, track, marble_radius=radius,
                            marble_material=marble)
    return world, i, track, ctrl


def _run(world, ctrl, frames=60, dt=1 / 60.0):
    for _ in range(frames):
        world.step(dt)
        ctrl.update(dt)


def _kicks(ctrl, forward, right, n=6):
    for _ in range(n):
        ctrl.kick(forward, right)


# -- steering ------------------------------------------------------------

def test_forward_input_rolls_marble_up_the_slope():
    world, i, track, ctrl = _world_with_marble(
        cells={(c, r): 0.0 for c in range(-2, 3) for r in range(-5, 3)})
    _kicks(ctrl, 1.0, 0.0)
    _run(world, ctrl, frames=90)
    assert world.position[i][2] < -1.0        # forward (up arrow) is -Z, up the slope


def test_right_input_rolls_marble_to_the_right():
    world, i, track, ctrl = _world_with_marble(
        cells={(c, r): 0.0 for c in range(-2, 6) for r in range(-3, 3)})
    _kicks(ctrl, 0.0, 1.0)
    _run(world, ctrl, frames=90)
    assert world.position[i][0] > 1.0         # moved toward +X (right)


def test_single_kick_deflects_the_marble():
    """One arrow click gives a spin that friction turns into a visible deflection.

    It is deliberately *gradual* (pure spin, no linear lurch), so the marble keeps
    its momentum — a click nudges, it does not teleport."""
    world, i, track, ctrl = _world_with_marble(
        cells={(c, r): 0.0 for c in range(-4, 5) for r in range(-4, 5)})
    x0 = float(world.position[i][0])
    ctrl.kick(0.0, 1.0)                        # a single click to the right
    _run(world, ctrl, frames=60)              # ~1 s of the spin converting to motion
    assert world.position[i][0] - x0 > 0.25


def test_a_single_kick_cannot_reverse_a_moving_marble():
    """Momentum matters: one opposing click can't flip a marble's direction."""
    world, i, track, ctrl = _world_with_marble(
        cells={(c, r): 0.0 for c in range(-8, 9) for r in range(-8, 9)})
    world.linear_velocity[i] = (4.0, 0.0, 0.0)   # moving +X at 4 m/s
    ctrl.kick(0.0, -1.0)                          # one click the other way
    _run(world, ctrl, frames=20)
    assert world.linear_velocity[i][0] > 0.5     # still moving +X, not reversed


def test_no_input_leaves_marble_roughly_in_place():
    world, i, track, ctrl = _world_with_marble()
    _run(world, ctrl, frames=60)
    assert abs(world.position[i][0]) < 0.5
    assert abs(world.position[i][2]) < 0.5


def test_pure_spin_converts_to_motion_and_slows():
    """Emergent coupling: a marble given only spin rolls off, and the spin bleeds
    down as it becomes translation (the 'spin slows as it imparts movement' rule)."""
    world, i, track, ctrl = _world_with_marble(
        cells={(c, r): 0.0 for c in range(-4, 5) for r in range(-6, 3)})
    world.angular_velocity[i] = (8.0, 0.0, 0.0)   # pure spin about +X (no linear)
    spin0 = np.linalg.norm(world.angular_velocity[i])
    z0 = float(world.position[i][2])
    _run(world, ctrl, frames=90)
    assert world.position[i][2] != z0                          # it rolled off
    assert np.linalg.norm(world.angular_velocity[i]) < spin0    # spin bled down


# -- grounding & checkpoints --------------------------------------------

def test_marble_on_track_sets_checkpoint_to_current_cell():
    world, i, track, ctrl = _world_with_marble(start=(4.0, 0.6, 0.0))
    ctrl.update(1 / 60.0)
    assert ctrl.checkpoint == (1, 0)


def test_checkpoint_advances_as_marble_rolls_to_a_new_cell():
    """Onto cells the marble is held on, and no others.

    A marble is respawned *at rest* and the board's own lean starts it moving
    again at once, so a checkpoint at the lip of a drop is one the lean carries
    straight back over the same edge: measured on a lane with a bite out of one
    side, a marble shoved into the gap fell seven times, six of them from the
    same lip cell.  A cell counts as somewhere to come back to when every side
    of it is either more floor or a rail.
    """
    world, i, track, ctrl = _world_with_marble(start=(0.0, 0.6, 0.0))
    ctrl.update(1 / 60.0)
    assert ctrl.checkpoint == (0, 0)
    world.position[i] = (8.0, 0.6, 0.0)       # teleport onto cell (2,0)
    ctrl.update(1 / 60.0)
    assert ctrl.checkpoint == (0, 0), \
        'a bare strip holds the marble nowhere, so the start is where it comes back'

    # Widen the strip and the middle of it becomes somewhere to come back to.
    for col in range(4):
        for row in (-1, 1):
            track.cells[(col, row)] = 0.0
    world.position[i] = (8.0, 0.6, 0.0)
    ctrl.update(1 / 60.0)
    assert ctrl.checkpoint == (2, 0), 'floor all round and still not a checkpoint'


def test_a_rail_makes_a_cell_somewhere_to_come_back_to():
    """What a checkpoint needs is that the marble cannot leave by that side, and
    a rail is the board saying so."""
    world, i, track, ctrl = _world_with_marble(start=(0.0, 0.6, 0.0))
    track.rails = frozenset((cell, step)
                             for cell in ((1, 0), (2, 0))
                             for step in ((0, 1), (0, -1)))
    world.position[i] = (8.0, 0.6, 0.0)       # cell (2,0), railed both sides
    ctrl.update(1 / 60.0)
    assert ctrl.checkpoint == (2, 0), \
        'a railed cell is not being counted as somewhere to come back to'


# -- fall vs jump --------------------------------------------------------

def test_falling_off_the_side_triggers_respawn_state():
    world, i, track, ctrl = _world_with_marble()
    ctrl.update(1 / 60.0)                       # grounds, checkpoint (0,0)
    # Marble sails off into the void and drops below the track plane.
    world.position[i] = (0.0, -3.0, 12.0)       # cell (0,3) is not in the map
    ctrl.update(1 / 60.0)
    assert ctrl.state == "fallen"


def test_a_high_jump_over_a_gap_is_not_a_fall():
    # Two platforms with a one-cell gap between them at (2,0).
    cells = {(0, 0): 0.0, (1, 0): 0.0, (3, 0): 0.0, (4, 0): 0.0}
    world, i, track, ctrl = _world_with_marble(cells=cells, start=(4.0, 0.6, 0.0))
    ctrl.update(1 / 60.0)
    # Marble is airborne over the gap cell but ABOVE the track plane (a jump arc).
    world.position[i] = (8.0, 1.2, 0.0)         # over void cell (2,0), but high
    world.linear_velocity[i] = (-6.0, 1.0, 0.0)
    ctrl.update(1 / 60.0)
    assert ctrl.state == "active"


def test_dropping_below_kill_plane_always_falls():
    world, i, track, ctrl = _world_with_marble(start=(0.0, 0.6, 0.0))
    ctrl.update(1 / 60.0)
    world.position[i] = (0.0, -50.0, 0.0)       # still over cell (0,0) but way down
    ctrl.update(1 / 60.0)
    assert ctrl.state == "fallen"


# -- respawn delay -------------------------------------------------------

def test_respawn_waits_the_penalty_delay_then_restores_the_marble():
    world, i, track, ctrl = _world_with_marble(start=(4.0, 0.6, 0.0))
    ctrl.respawn_delay = 2.0
    ctrl.update(1 / 60.0)                        # checkpoint (1,0)
    world.position[i] = (4.0, -8.0, 6.0)         # fell off near cell (1,0)
    ctrl.update(1 / 60.0)
    assert ctrl.state == "fallen"

    # Not yet respawned after 1 s.
    for _ in range(60):
        ctrl.update(1 / 60.0)
    assert ctrl.state == "fallen"

    # After passing 2 s total it respawns at the checkpoint cell, at rest.
    for _ in range(70):
        ctrl.update(1 / 60.0)
    assert ctrl.state == "active"
    assert np.allclose(world.position[i][[0, 2]], (4.0, 0.0), atol=0.3)   # cell (1,0) centre
    assert np.linalg.norm(world.linear_velocity[i]) < 0.1
    assert np.linalg.norm(world.angular_velocity[i]) < 0.1


def test_fall_count_increments_on_each_respawn():
    world, i, track, ctrl = _world_with_marble()
    ctrl.respawn_delay = 0.5
    ctrl.update(1 / 60.0)
    assert ctrl.fall_count == 0
    world.position[i] = (0.0, -20.0, 0.0)
    for _ in range(60):
        ctrl.update(1 / 60.0)
    assert ctrl.fall_count == 1


# -- camera hold ---------------------------------------------------------

class _SpyCamera:
    def __init__(self):
        self.holding = False
        self._target = None

    def target(self, position):
        if not self.holding:
            self._target = np.asarray(position, dtype='d')[:3]

    def hold(self):
        self.holding = True

    def release(self):
        self.holding = False

    @property
    def is_holding(self):
        return self.holding


def test_camera_holds_on_fall_and_releases_on_respawn():
    world, i, track, ctrl = _world_with_marble(start=(4.0, 0.6, 0.0))
    cam = _SpyCamera()
    ctrl.camera = cam
    ctrl.respawn_delay = 0.3
    ctrl.update(1 / 60.0)
    assert not cam.is_holding
    world.position[i] = (4.0, -10.0, 6.0)
    ctrl.update(1 / 60.0)
    assert cam.is_holding                       # froze on the fall
    for _ in range(30):
        ctrl.update(1 / 60.0)
    assert not cam.is_holding                    # released after respawn
