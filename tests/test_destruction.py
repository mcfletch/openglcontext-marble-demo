"""Losing the marble: the four ways it happens, and what each one costs.

Falling off the board respawns the marble after a short wait.  These are the
rules that *destroy* it instead — struck, dropped, crushed, burned — and every
one of them is a threshold, so each is tested from both sides: the blow, the
drop, the gap that the marble survives, and the one just past it that it does
not.  The numbers the physics actually produced go in the assertion messages,
because a threshold nobody can see the margin around is a threshold nobody can
tune.

Everything here runs against the real solver with no window: contacts, closing
speeds and penetration depths are the solver's own, which is what makes the
margins meaningful rather than arithmetic about constants.
"""
import numpy as np
from omi_physics import model
from omi_physics.world import PhysicsWorld

from openglcontext_marble_demo import hud, materials
from openglcontext_marble_demo.controller import (
    ACTIVE,
    CRUSHED,
    DESTROYED,
    DROPPED,
    FALLEN,
    STRUCK,
    MarbleController,
)
from openglcontext_marble_demo.game import LOST, PLAYING, MarbleGame
from openglcontext_marble_demo.level import Finish, Level
from openglcontext_marble_demo.track import TrackMap

DT = 1 / 60.0
RADIUS = 0.5


# -- a bare world, as the impact rules see it ----------------------------------

def _world():
    world = PhysicsWorld(gravity=model.Gravity(gravity=9.81, direction=(0, -1, 0)),
                         sleep_enabled=False)
    index = materials.register_materials(world)
    materials.apply_pair_frictions(world, index)
    ground = world.add_shape(model.Shape.box((200, 1, 200)))
    world.add_body(model.Motion(type=model.STATIC),
                   collider=model.Collider(shape=ground, physicsMaterial=index['stone']),
                   position=(0, -0.5, 0))
    return world, index


def _marble(world, index, position=(0.0, 0.6, 0.0), velocity=(0.0, 0.0, 0.0),
            material='steel'):
    ball = world.add_shape(model.Shape.sphere(RADIUS))
    return world.add_body(
        model.Motion(type=model.DYNAMIC, mass=materials.MARBLES[material].mass,
                     linearVelocity=velocity),
        collider=model.Collider(shape=ball, physicsMaterial=index[material]),
        position=position)


def _controller(world, i):
    cells = {(c, r): 0.0 for c in range(-8, 20) for r in range(-8, 8)}
    return MarbleController(world, i, TrackMap(cells, cell_size=4.0),
                            marble_radius=RADIUS)


def _wall(world, index, x, material='stone'):
    shape = world.add_shape(model.Shape.box((0.5, 6, 20)))
    world.add_body(model.Motion(type=model.STATIC),
                   collider=model.Collider(shape=shape, physicsMaterial=index[material]),
                   position=(x, 3.0, 0))


def _hardest_blow(world, i):
    """The fastest the marble was closing on anything in the step just run."""
    return max((float(c.approach) for c in world.contacts if i in (c.a, c.b)),
               default=0.0)


# -- struck too hard -----------------------------------------------------------

def _slam(speed, surface='stone'):
    """Drive a marble into a wall at ``speed``; the controller and the blow it took."""
    world, index = _world()
    _wall(world, index, 8.0, surface)
    i = _marble(world, index, velocity=(speed, 0.0, 0.0))
    ctrl = _controller(world, i)
    hardest = 0.0
    for _ in range(120):
        world.step(DT)
        hardest = max(hardest, _hardest_blow(world, i))
        if ctrl.update(DT) != ACTIVE:
            break                        # the outcome is settled; stop before it comes back
    return ctrl, hardest


def test_a_wall_hit_faster_than_any_board_delivers_leaves_the_marble_whole():
    """Well above what a boost pad or a downhill run reaches, and it survives."""
    ctrl, blow = _slam(18.0)
    assert ctrl.state != DESTROYED, (
        'a %.1f m/s blow destroyed the marble against a %.1f m/s threshold: nothing '
        'a board can do to a player should' % (blow, ctrl.lethal_impact_speed))
    assert blow < ctrl.lethal_impact_speed, (
        'the survivable case measured %.1f m/s, at or over the %.1f m/s threshold: '
        'it is not testing the margin it claims to' % (blow, ctrl.lethal_impact_speed))


def test_a_wall_hit_hard_enough_destroys_the_marble():
    ctrl, blow = _slam(28.0)
    assert blow > ctrl.lethal_impact_speed, (
        'the fatal case only measured %.1f m/s against a %.1f m/s threshold'
        % (blow, ctrl.lethal_impact_speed))
    assert ctrl.state == DESTROYED and ctrl.last_loss == STRUCK, (
        'a %.1f m/s blow left the marble %s: the threshold is %.1f m/s'
        % (blow, ctrl.state, ctrl.lethal_impact_speed))


def test_a_springy_bumper_is_measured_the_same_as_a_wall():
    """The rule is the closing speed, which a bumper does not change.

    A bumper returns the energy through restitution, so the *impulse* it records
    is half again what stone records for the same hit.  Keying destruction on
    that would make the softest thing on the board the deadliest.
    """
    stone, on_stone = _slam(28.0)
    rubber, on_rubber = _slam(28.0, surface='rubber_pad')
    assert abs(on_stone - on_rubber) < 0.5, (
        'the same hit measured %.1f m/s on stone and %.1f m/s on rubber'
        % (on_stone, on_rubber))
    assert stone.state == rubber.state == DESTROYED


# -- fallen too far ------------------------------------------------------------

def _drop(height):
    """Drop a marble ``height`` metres onto the track; the controller and the fall.

    Stops the moment the marble is lost, so what is reported is the drop that
    took it rather than the clean slate its replacement starts with.
    """
    world, index = _world()
    i = _marble(world, index, position=(0.0, RADIUS + height, 0.0))
    ctrl = _controller(world, i)
    for _ in range(400):
        world.step(DT)
        if ctrl.update(DT) != ACTIVE:
            break
    return ctrl, ctrl.fell_from


def test_a_long_drop_onto_the_board_is_survivable():
    ctrl, fell = _drop(6.0)
    assert ctrl.state == ACTIVE, (
        'a %.1f m drop destroyed the marble against a %.1f m threshold'
        % (fell, ctrl.lethal_fall_height))
    assert fell < ctrl.lethal_fall_height


def test_a_drop_past_the_lethal_height_destroys_the_marble():
    ctrl, fell = _drop(12.0)
    assert fell > ctrl.lethal_fall_height, (
        'the fatal case only fell %.1f m against a %.1f m threshold'
        % (fell, ctrl.lethal_fall_height))
    assert ctrl.state == DESTROYED and ctrl.last_loss == DROPPED, (
        'a %.1f m drop left the marble %s: the threshold is %.1f m'
        % (fell, ctrl.state, ctrl.lethal_fall_height))


def test_dropping_far_is_not_the_same_as_falling_off():
    """Off the edge is a respawn; onto the board from a height is a loss.

    The void has nothing in it to hit, which is the whole difference.
    """
    world, index = _world()
    cells = {(0, 0): 0.0}                       # one square, and nothing around it
    i = _marble(world, index, position=(0.0, 0.6, 0.0))
    ctrl = MarbleController(world, i, TrackMap(cells, cell_size=4.0), marble_radius=RADIUS)
    ctrl.update(DT)
    world.place_body(i, position=(0.0, -30.0, 40.0))     # out over the void
    ctrl.update(DT)
    assert ctrl.state == FALLEN
    assert ctrl.last_loss is None


# -- crushed -------------------------------------------------------------------

def _lid(world, index, y):
    slab = world.add_shape(model.Shape.box((8.0, 1.0, 8.0)))
    return world.add_body(model.Motion(type=model.KINEMATIC),
                          collider=model.Collider(shape=slab, physicsMaterial=index['stone']),
                          position=(0.0, y, 0.0))


def _close_to(gap, hold=2.0):
    """Bring a slab down until the space under it is ``gap`` metres.

    It is held there for ``hold`` seconds and then lifted off again.  Returns the
    controller and the deepest squeeze the solver recorded — how far below its
    diameter the marble was held.
    """
    world, index = _world()
    i = _marble(world, index, position=(0.0, RADIUS, 0.0))
    ctrl = _controller(world, i)
    lid = _lid(world, index, 4.0)
    y, shut = 4.0, gap + 0.5                    # the slab is a metre thick
    deepest, held = 0.0, 0.0
    for _ in range(int(6.0 / DT)):
        if held >= hold:
            y = min(4.0, y + 4.0 * DT)
        else:
            y = max(shut, y - 4.0 * DT)
            held += DT if y <= shut else 0.0
        world.place_body(lid, position=(0.0, y, 0.0))
        world.step(DT)
        deepest = max(deepest, _squeeze(world, i))
        if ctrl.update(DT) != ACTIVE:
            break
    return ctrl, deepest


def _squeeze(world, i):
    """How far below its diameter body ``i`` is held between two opposed surfaces."""
    held = [(np.asarray(c.normal if c.a == i else -c.normal, dtype='d'), float(c.depth))
            for c in world.contacts if i in (c.a, c.b)]
    return max((a[1] + b[1]
                for n, a in enumerate(held) for b in held[n + 1:]
                if float(np.dot(a[0], b[0])) <= -0.5), default=0.0)


def test_a_gap_the_marble_still_fits_under_holds_it_without_crushing_it():
    ctrl, squeeze = _close_to(0.94)
    assert ctrl.state == ACTIVE, (
        'held %.3f m under its diameter and destroyed, against a %.3f m threshold'
        % (squeeze, ctrl.crush_squeeze * 2 * RADIUS))
    assert squeeze < ctrl.crush_squeeze * 2 * RADIUS


def test_a_gap_that_closes_past_the_marble_crushes_it():
    ctrl, squeeze = _close_to(0.86)
    assert squeeze > ctrl.crush_squeeze * 2 * RADIUS, (
        'the fatal case only squeezed %.3f m against a %.3f m threshold'
        % (squeeze, ctrl.crush_squeeze * 2 * RADIUS))
    assert ctrl.state == DESTROYED and ctrl.last_loss == CRUSHED, (
        'squeezed %.3f m and the marble was %s' % (squeeze, ctrl.state))


def test_a_squeeze_let_go_of_in_time_is_survived():
    """The press has to be held: a gap that shuts and opens again is a scare."""
    ctrl, squeeze = _close_to(0.86, hold=0.15)
    assert squeeze > ctrl.crush_squeeze * 2 * RADIUS, (
        'the marble was never squeezed hard enough for the test to mean anything '
        '(%.3f m)' % squeeze)
    assert ctrl.state == ACTIVE, (
        'a squeeze released inside %.2f s still destroyed the marble' % ctrl.crush_time)


# -- what a loss costs the run -------------------------------------------------

def _game(time_limit=60.0):
    # A plate rather than a single row: the board leans, so a marble put back at
    # a checkpoint rolls, and a row one cell wide would have it off the side
    # before the test had finished looking at it.
    cells = {(c, r): 0.0 for c in range(0, 5) for r in range(-3, 4)}
    level = Level(name='loss', cells=cells, start_cell=(0, 0), finish_cell=(4, 0),
                  time_limit=time_limit, features=[Finish((4, 0))])
    return MarbleGame(level)


def test_losing_a_marble_costs_a_chunk_of_the_clock():
    game = _game(time_limit=60.0)
    before = game.time_left
    game.controller.destroy(CRUSHED)
    assert game.time_left == before - game.loss_penalty
    assert game.state == PLAYING            # the run goes on, poorer


def test_a_loss_that_empties_the_clock_ends_the_run_and_says_what_ended_it():
    game = _game(time_limit=60.0)
    game.time_left = 3.0
    game.controller.destroy(CRUSHED)
    assert game.time_left == 0.0
    assert game.state == LOST
    assert game.ended_by == CRUSHED


def test_a_loss_waits_longer_than_a_fall_and_comes_back_at_the_checkpoint():
    game = _game(time_limit=600.0)
    ctrl = game.controller
    assert ctrl.destroy_delay > ctrl.respawn_delay, (
        'being destroyed waits %.1f s and falling off waits %.1f s: the player '
        'cannot tell the two apart' % (ctrl.destroy_delay, ctrl.respawn_delay))
    ctrl.checkpoint = (2, 0)
    ctrl.destroy(CRUSHED)
    for _ in range(int((ctrl.respawn_delay + 0.5) / DT)):
        game.advance(DT)
    assert ctrl.state == DESTROYED, 'a loss came back on the fall delay'
    while ctrl.state != ACTIVE and game.time_left > 500.0:
        game.advance(DT)
    assert ctrl.state == ACTIVE
    assert np.allclose(game.scene.world.position[game.marble.index][[0, 2]],
                       (8.0, 0.0), atol=0.4)


def test_losing_the_marble_does_not_count_as_falling_off():
    game = _game()
    game.controller.destroy(STRUCK)
    assert game.controller.loss_count == 1
    assert game.controller.fall_count == 0


# -- telling the player --------------------------------------------------------

def test_the_status_lines_name_what_the_marble_was_lost_to():
    game = _game()
    assert not any('lost' in line.lower() for line in hud.hud_lines(game))
    game.controller.destroy(CRUSHED)
    line = [line for line in hud.hud_lines(game) if 'LOST' in line]
    assert line and CRUSHED in line[0].lower(), hud.hud_lines(game)


def test_the_banner_says_what_killed_the_marble_while_it_is_remade():
    game = _game(time_limit=600.0)
    assert hud.banner(game) is None
    game.controller.destroy(DROPPED)
    message = hud.banner(game)
    assert message is not None and DROPPED in message.lower(), message
    for _ in range(int((game.controller.destroy_delay + 0.5) / DT)):
        game.advance(DT)
    assert hud.banner(game) is None          # back in play, and the message is gone


def test_the_end_of_run_banner_names_the_loss_that_ended_it():
    game = _game()
    game.time_left = 1.0
    game.controller.destroy(STRUCK)
    assert game.state == LOST
    assert STRUCK in hud.banner(game).lower()


def test_running_out_of_time_on_its_own_still_reads_as_running_out_of_time():
    game = _game(time_limit=0.5)
    for _ in range(60):
        game.advance(DT)
    assert game.state == LOST
    assert game.ended_by is None
    assert 'time' in hud.banner(game).lower()


# -- a restarted run forgets it ------------------------------------------------

def test_a_restarted_run_forgets_the_marble_it_lost():
    game = _game()
    game.controller.destroy(CRUSHED)
    assert game.controller.loss_count == 1
    game.reset()
    assert game.controller.loss_count == 0
    assert game.controller.last_loss is None
    assert game.ended_by is None
    assert game.state == PLAYING
    assert game.controller.state == ACTIVE
    assert not any('LOST' in line for line in hud.hud_lines(game))


def test_a_lethal_blow_in_an_early_step_of_a_slow_frame_still_destroys():
    """At 15 fps a frame is eight physics steps, and the blow lands in one of them.

    Closing speed is read on the step the marble strikes and on no other, so
    the controller has to hear every step rather than look at the last one.
    """
    world, index = _world()
    world.fixed_dt = 1 / 120.0
    _wall(world, index, 8.0)
    i = _marble(world, index, velocity=(30.0, 0.0, 0.0))
    ctrl = _controller(world, i)
    frame = 1 / 15.0
    for _ in range(30):
        world.advance(frame)
        if ctrl.update(frame) != ACTIVE:
            break
    assert ctrl.state == DESTROYED and ctrl.last_loss == STRUCK


def test_blows_taken_while_lost_are_not_read_after_the_respawn():
    """A lost marble is still in the world, and things still hit it.

    It rolls on into the wall again during the destroy delay; the marble set
    down at the checkpoint afterwards has not been struck.
    """
    world, index = _world()
    _wall(world, index, 8.0)
    i = _marble(world, index, velocity=(30.0, 0.0, 0.0))
    ctrl = _controller(world, i)
    for _ in range(120):
        world.step(DT)
        if ctrl.update(DT) != ACTIVE:
            break
    assert ctrl.state == DESTROYED and ctrl.loss_count == 1
    world.position[i] = (5.0, RADIUS, 0.0)
    world.linear_velocity[i] = (30.0, 0.0, 0.0)
    while ctrl.state != ACTIVE:
        world.step(DT)
        ctrl.update(DT)
    world.step(DT)
    assert ctrl.update(DT) == ACTIVE
    assert ctrl.loss_count == 1
