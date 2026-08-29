"""The game orchestrator: level + physics + controller + clock + win/lose.

:class:`MarbleGame` is the seam between the *rules* (all headless and tested) and
the *window* (``run.py``).  It builds a level into a physics scene, spawns the
marble, wires the controller and the fixed-offset camera, runs the countdown, and
decides win (reached the finish) and lose (ran out of time).

The run loop calls, each frame: :meth:`lean` (or :meth:`kick`) with the arrow
input, then :meth:`advance` with the elapsed time.  When the run has ended (won or
lost) both become no-ops, so the final frame stays frozen until the caller resets
or loads the next level.

**Two control models** share this class, so the difference between them can be
felt back to back rather than argued about:

``TILT``
    The player leans the board (:mod:`~openglcontext_marble_demo.tilt`) and
    gravity follows; the marble is never pushed directly.
``SPIN``
    Each arrow press imparts a spin kick through
    :class:`~openglcontext_marble_demo.controller.MarbleController`, and the board
    keeps its constant lean.

Both run on one gravity path: the game owns a
:class:`~openglcontext_marble_demo.tilt.TiltRig` either way, and ``SPIN`` is
simply a rig whose player lean is bounded at zero.
"""
import math

from omi_physics import model
from OpenGLContext.physics.demo import DemoScene
from OpenGLContext.scenegraph import basenodes
from OpenGLContext.scenegraph.transform import Transform

from . import materials, render
from .controller import MarbleController
from .tilt import TiltRig

PLAYING = "playing"
WON = "won"
LOST = "lost"

#: The player leans the board; gravity follows.
TILT = "tilt"
#: The player imparts spin; the board's lean is a constant.
SPIN = "spin"

MARBLE_RADIUS = 0.5
BACKGROUND = (0.16, 0.18, 0.23)

# The whole board leans toward +Z (the finish direction), so — exactly like the
# original Marble Madness — the marble rolls forward on its own if the player does
# nothing, and playing is about directing that momentum rather than creating it.
GRAVITY = 9.81
#: How far the board leans downhill, in radians.
#:
#: The pair of numbers below is the game's whole steering feel, and it was set by
#: one measurement: **how far the marble travels forward while it moves one cell
#: sideways.** A ratio near 1 means a lane change costs a lane, which threads a
#: three-wide lane; much above that and a player is aiming at something they have
#: already passed.
#:
#: At the old 12.4° of board lean and 26° of player lean the ratio was **2.77**,
#: so every fragment about aim was being asked of a control that could not aim.
#: A gentler board and a wider lean bring it to 1.3, at a cruising speed of
#: 3.2 m/s. The board's lean is what was traded: it is pace against being able
#: to put the marble anywhere, and pace is worth nothing if the marble goes
#: where it likes.
BASE_TILT = math.radians(8.0)
#: How far the player may lean it on top of that, on either axis.
PLAYER_TILT = math.radians(40)

#: Linear and angular damping, per second, for everything in the game's world.
#:
#: The scenegraph physics manager defaults to 0.3 and 1.5, which settle a scene
#: of boxes and are a continuous brake on a ball whose whole job is to roll:
#: angular damping at 1.5 takes most of a marble's spin away every second, and
#: rolling couples spin to travel, so it is felt as the marble refusing to get
#: going.  These numbers are most of how fast the game feels — free-rolling
#: pace is about two and a half times what the manager's defaults allow — and
#: enough damping is kept that a marble at rest settles rather than creeping.
ROLL_DAMPING = (0.05, 0.2)

#: Seconds off the clock for a marble lost to a trap, on top of the longer wait
#: for the replacement.
#:
#: The run continues from the last checkpoint rather than ending, because a board
#: that stops at the first mistake is a board played once.  What the loss costs is
#: paid in the currency the player is already watching — a fall off the edge costs
#: the two seconds of its respawn and this costs six times that — so there is one
#: thing to keep an eye on and no second economy to learn.  A player who cannot
#: pay ends the run, and the banner says which trap ended it.
LOSS_PENALTY = 8.0


class MarbleGame:
    def __init__(self, level, marble_material="steel", camera=None,
                 radius=MARBLE_RADIUS, debug_flags=0,
                 steer_forward=None, steer_right=None,
                 control=TILT, base_tilt=BASE_TILT, player_tilt=PLAYER_TILT,
                 damping=ROLL_DAMPING, loss_penalty=LOSS_PENALTY):
        self.level = level
        self.marble_material = marble_material
        self.radius = radius
        self.camera = camera
        self.control = control
        self._steer_forward = steer_forward
        self._steer_right = steer_right

        # One gravity path for both control models: the rig always owns the
        # board's lean, and SPIN is the rig with no player lean available.
        self.tilt = TiltRig(
            base=base_tilt,
            limit=player_tilt if control == TILT else 0.0,
            forward_axis=steer_forward if steer_forward is not None else (0.0, 0.0, -1.0),
            right_axis=steer_right if steer_right is not None else (1.0, 0.0, 0.0))
        self._demand = (0.0, 0.0)

        gravity = model.Gravity(gravity=GRAVITY,
                                direction=tuple(self.tilt.gravity_direction()))
        self.scene = DemoScene(gravity=gravity, debug_flags=debug_flags,
                               background=BACKGROUND,
                               default_linear_damping=damping[0],
                               default_angular_damping=damping[1])
        self.material_index = materials.register_materials(self.scene.world)
        materials.apply_pair_frictions(self.scene.world, self.material_index)

        self.build = level.build_into(self.scene, self.material_index)
        self.marble = self._spawn_marble()
        # Everything the board is made of hangs off one Transform, so the lean
        # can be *drawn*: see :meth:`scene_graph`.
        self.board = Transform(children=list(self.scene.children))
        steer = {}
        if steer_forward is not None:
            steer["steer_forward"] = steer_forward
        if steer_right is not None:
            steer["steer_right"] = steer_right
        self.loss_penalty = loss_penalty
        self.controller = MarbleController(
            self.scene.world, self.marble.index, self.build.track,
            marble_radius=radius, marble_material=marble_material, camera=camera,
            kill_y=level.kill_y, respawn_delay=level.respawn_delay,
            on_lost=self._marble_lost, **steer)

        #: The gates still to be passed.  A run cannot be finished while any
        #: remain, which is what makes a board a route rather than a direction.
        self.gates = set(self.build.gate_bodies)
        # The board's hazards: anything among the animators that can take a
        # marble by holding it rather than by hitting it, and so has to be asked
        # rather than measured from the contacts.
        self.hazards = [a for a in self.build.animators if hasattr(a, 'lost')]
        self._wire_triggers()
        self.time_left = level.time_limit
        self.state = PLAYING
        #: What ended the run, when a lost marble was what emptied the clock.
        self.ended_by = None

    # -- setup ----------------------------------------------------------
    def _spawn_marble(self):
        marble = materials.MARBLES[self.marble_material]
        body = self.scene.add_sphere(
            radius=self.radius, position=self.level.marble_start(self.radius),
            color=marble.base_color, mass=marble.mass,
            material=self.material_index[self.marble_material])
        # Swap the default VRML appearance for a PBR one so the marble reflects.
        body.transform.children[0].appearance = render.marble_appearance(marble)
        return body

    def _wire_triggers(self):
        """One trigger listener drives both the finish (win) and the feature effects.

        The finish body ends the run; every other trigger in ``build.effects`` (ramp
        boosts, spring traps) fires its effect on the marble while it overlaps.
        """
        finish_index = self.build.finish_body.index if self.build.finish_body else None
        marble_index = self.marble.index
        effects = self.build.effects
        world = self.scene.world

        def on_trigger(event_type, trigger_body, other_body):
            if other_body != marble_index or event_type == "exit":
                return
            self.gates.discard(trigger_body)
            if trigger_body == finish_index and self.state == PLAYING \
                    and not self.gates:
                self.state = WON
            effect = effects.get(trigger_body)
            if effect is not None and self.state == PLAYING:
                effect(world, marble_index)

        world.add_trigger_listener(on_trigger)

    # -- per-frame ------------------------------------------------------
    def lean(self, forward, right):
        """Record which way the player is leaning the board, in [-1, 1].

        This is *held* input, not an event: it stands until it is changed, and
        :meth:`advance` is what turns it into board movement over elapsed time.
        A run that has ended levels the board rather than holding the last lean.
        """
        self._demand = (forward, right) if self.state == PLAYING else (0.0, 0.0)

    def kick(self, forward, right):
        """Apply one steering kick (from a key press / repeat) while playing."""
        if self.state == PLAYING:
            self.controller.kick(forward, right)

    def advance(self, dt):
        """Step the world and the clock; return the (possibly new) game state."""
        if self.state != PLAYING:
            return self.state
        # The board leans first, so the step the marble takes this frame is the
        # one the player is asking for now rather than the one they asked for last.
        self.tilt.update(dt, *self._demand)
        self.scene.world.gravity.direction = tuple(self.tilt.gravity_direction())
        for animator in self.build.animators:      # drive elevators/arms this frame
            animator.update(dt)
        self.scene.advance(dt)
        self.controller.update(dt)
        self._check_hazards()
        self._draw_lean()
        # `state` may have flipped to WON inside advance() via the finish trigger.
        if self.state == PLAYING:
            self.time_left -= dt
            if self.time_left <= 0.0:
                self.time_left = 0.0
                self.state = LOST
        return self.state

    # -- losing the marble ----------------------------------------------
    def _check_hazards(self):
        """Ask the board's hazards whether any of them has taken the marble.

        A hazard destroys by holding rather than by striking, so there is no
        contact for the controller to read and the answer has to be asked for.
        """
        for hazard in self.hazards:
            cause = hazard.lost(self.marble.index)
            if cause is not None:
                self.controller.destroy(cause)

    def _marble_lost(self, cause):
        """Charge the run for a destroyed marble; end it if it cannot pay.

        The clock is the only thing a run spends, so this is what a trap costs:
        :data:`LOSS_PENALTY` seconds, and the run carries on from the last
        checkpoint.  A player who does not have the seconds is out of time, and
        ``cause`` is what the banner says took them.
        """
        self.time_left = max(0.0, self.time_left - self.loss_penalty)
        if self.time_left <= 0.0 and self.state == PLAYING:
            self.state = LOST
            self.ended_by = cause

    def reset(self):
        """Abort the run: marble back to the start, clock full, playing again."""
        self.controller.checkpoint = self.level.start_cell
        self.controller._checkpoint_surface = self.level.cells[self.level.start_cell]
        self.controller._respawn()
        self.controller.forget_the_run()
        self.time_left = self.level.time_limit
        self.state = PLAYING
        self.ended_by = None
        # Every gate shut again, and every lever, plug and door put back: a run
        # started over is started over.
        self.gates = set(self.build.gate_bodies)
        for holder in self.build.resettable:
            holder.reset(self.scene.world)
        # A board still leaning from the run just abandoned would start the next
        # one already moving.
        self.tilt.level()
        self._demand = (0.0, 0.0)
        self._draw_lean()

    @property
    def gates_left(self):
        """How many waypoints are still to be reached."""
        return len(self.gates)

    # -- live material change ------------------------------------------
    def set_marble_material(self, name):
        """Swap the marble's material live — both its look and its feel.

        Updates the body's physics material (so the pairwise friction with the
        surface changes), its mass and sphere inertia, and its render colour, all
        in place, so the player can feel the difference mid-run.
        """
        if name not in materials.MARBLES:
            return
        marble = materials.MARBLES[name]
        world, i = self.scene.world, self.marble.index
        world.collider_material[i] = self.material_index[name]
        world.mass[i] = marble.mass
        world.inv_mass[i] = 1.0 / marble.mass
        inertia = 0.4 * marble.mass * self.radius ** 2      # solid sphere
        world.inv_inertia[i] = 1.0 / inertia
        self.marble.transform.children[0].appearance = render.marble_appearance(marble)
        self.marble_material = name
        self.controller.marble_material = name

    # -- rendering ------------------------------------------------------
    def _draw_lean(self):
        """Point the board Transform at the lean, about the marble.

        Rotating about the marble rather than the world origin keeps the ball
        still on screen and leans the world around it — and it keeps the camera
        out of it entirely, since the point it follows is the one point the
        rotation does not move.
        """
        self.board.rotation = self.tilt.board_rotation()
        self.board.center = tuple(self.scene.world.position[self.marble.index])

    def scene_graph(self, extra=()):
        """The full graph: the leaning board, the debug overlay, ``extra``, sky.

        Only the board leans.  The light and the background are outside it, so
        the sun does not swing across the sky every time the player steers.
        """
        children = [self.board, self.scene.debug.root]
        children.extend(extra)
        children.append(basenodes.SimpleBackground(color=BACKGROUND))
        return basenodes.sceneGraph(children=children)
