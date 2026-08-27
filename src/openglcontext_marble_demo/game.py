"""The game orchestrator: level + physics + controller + clock + win/lose.

:class:`MarbleGame` is the seam between the *rules* (all headless and tested) and
the *window* (``run.py``).  It builds a level into a physics scene, spawns the
marble, wires the controller and the fixed-offset camera, runs the countdown, and
decides win (reached the finish) and lose (ran out of time).

The run loop calls, each frame: :meth:`steer` with the arrow input, then
:meth:`advance` with the elapsed time.  When the run has ended (won or lost) both
become no-ops, so the final frame stays frozen until the caller resets or loads the
next level.
"""
from omi_physics import model
from OpenGLContext.physics.demo import DemoScene

from . import materials, render
from .controller import MarbleController

PLAYING = "playing"
WON = "won"
LOST = "lost"

MARBLE_RADIUS = 0.5
BACKGROUND = (0.16, 0.18, 0.23)

# The whole board tilts toward +Z (the finish direction), so — exactly like the
# original Marble Madness — the marble rolls forward on its own if the player does
# nothing, and steering is about fighting and directing that momentum.  A firm tilt
# makes the downhill pull noticeably faster-acting.
GRAVITY = 9.81
BOARD_TILT = 0.22          # +Z component of the (downward) gravity direction


class MarbleGame:
    def __init__(self, level, marble_material="steel", camera=None,
                 radius=MARBLE_RADIUS, debug_flags=0,
                 steer_forward=None, steer_right=None):
        self.level = level
        self.marble_material = marble_material
        self.radius = radius
        self.camera = camera
        self._steer_forward = steer_forward
        self._steer_right = steer_right

        gravity = model.Gravity(gravity=GRAVITY, direction=(0.0, -1.0, BOARD_TILT))
        self.scene = DemoScene(gravity=gravity, debug_flags=debug_flags,
                               background=BACKGROUND)
        self.material_index = materials.register_materials(self.scene.world)
        materials.apply_pair_frictions(self.scene.world, self.material_index)

        self.build = level.build_into(self.scene, self.material_index)
        self.marble = self._spawn_marble()
        steer = {}
        if steer_forward is not None:
            steer["steer_forward"] = steer_forward
        if steer_right is not None:
            steer["steer_right"] = steer_right
        self.controller = MarbleController(
            self.scene.world, self.marble.index, self.build.track,
            marble_radius=radius, marble_material=marble_material, camera=camera,
            kill_y=level.kill_y, respawn_delay=level.respawn_delay, **steer)

        self._wire_triggers()
        self.time_left = level.time_limit
        self.state = PLAYING

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
            if trigger_body == finish_index and self.state == PLAYING:
                self.state = WON
            effect = effects.get(trigger_body)
            if effect is not None and self.state == PLAYING:
                effect(world, marble_index)

        world.add_trigger_listener(on_trigger)

    # -- per-frame ------------------------------------------------------
    def kick(self, forward, right):
        """Apply one steering kick (from a key press / repeat) while playing."""
        if self.state == PLAYING:
            self.controller.kick(forward, right)

    def advance(self, dt):
        """Step the world and the clock; return the (possibly new) game state."""
        if self.state != PLAYING:
            return self.state
        for animator in self.build.animators:      # drive elevators/arms this frame
            animator.update(dt)
        self.scene.advance(dt)
        self.controller.update(dt)
        # `state` may have flipped to WON inside advance() via the finish trigger.
        if self.state == PLAYING:
            self.time_left -= dt
            if self.time_left <= 0.0:
                self.time_left = 0.0
                self.state = LOST
        return self.state

    def reset(self):
        """Abort the run: marble back to the start, clock full, playing again."""
        self.controller.checkpoint = self.level.start_cell
        self.controller._checkpoint_surface = self.level.cells[self.level.start_cell]
        self.controller._respawn()
        self.controller.fall_count = 0
        self.time_left = self.level.time_limit
        self.state = PLAYING

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

    # -- rendering passthrough -----------------------------------------
    def scene_graph(self, extra=()):
        return self.scene.scene_graph(extra=extra)
