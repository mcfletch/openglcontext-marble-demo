"""Run loop and window for the marble demo.

``run.py`` is the *window* around the tested, headless game: it builds a
:class:`~openglcontext_marble_demo.game.MarbleGame` from a generated level, drives
the near-isometric follow camera, samples the keyboard, steps the game once per
frame, and paints the HUD.

The interesting rules — the board's lean, fall/respawn, the timer, win/lose, the
mechanisms — all live in the game/tilt/controller/level modules and are unit
tested; this file is the shape a programmer copies to drive OpenGLContext's
physics + PBR from their own input.

**Input is sampled, not reacted to.** The engine's
:class:`~OpenGLContext.events.inputstate.InputState` already accumulates every key
transition (``ViewPlatformMixin.ProcessEvent`` feeds it, whatever movement manager
is bound), so a frame asks what is held rather than counting key-repeat events.
That is what makes a lean an angle held over time instead of a stutter of kicks
arriving at whatever rate the platform repeats at.
"""
import argparse
import math
import os
import time
from typing import Any

# Core profile + GLFW + the PBR renderer with image-based lighting give reflective
# marbles and shadows; set before OpenGLContext imports a backend so it takes effect.
os.environ.setdefault("OPENGLCONTEXT_BACKEND", "glfw")
os.environ.setdefault("OPENGLCONTEXT_RENDERER", "pbr")
os.environ.setdefault("OPENGLCONTEXT_IBL", "full")

import numpy as np
from OpenGLContext import testingcontext
from OpenGLContext.move.followcam import FollowCamera
from OpenGLContext.physics.demo import disable_vsync
from OpenGLContext.scenegraph import basenodes

from . import generator, levelfile, materials
from .game import PLAYING, ROLL_DAMPING, SPIN, TILT, MarbleGame
from .hud import HUD

# Annotated Any: the base class is chosen at runtime by the backend the
# environment above selects, so it is not a name a checker can resolve.
BaseContext: Any = testingcontext.getInteractive()

# Isometric-style camera: 30° elevation above the ground *and* 30° of yaw (looking
# in from a corner) so the boxes show two faces and the scene reads as 3D — not the
# flat, face-on look a straight-down-the-axis view gives.  It sits on the downhill
# (+Z) side so the marble falls toward the viewer; a narrow FOV keeps perspective
# near-orthographic.  Follows the marble only; no camera controls.
CAMERA_ELEVATION = math.radians(30)
CAMERA_YAW = math.radians(30)
#: Far enough back to see the board rather than the square the marble is on:
#: with 4-metre cells and a 30° field of view this frames about a dozen of them.
CAMERA_DISTANCE = 48.0
CAMERA_FOV = math.radians(30)
#: How much further back the camera eases at speed, and the speed that reaches it.
#: Seeing further is worth most exactly when there is least time to react.
CAMERA_PULL_BACK = 0.30
CAMERA_PULL_BACK_SPEED = 9.0


def camera_offset(distance=CAMERA_DISTANCE, elevation=CAMERA_ELEVATION,
                  yaw=CAMERA_YAW):
    """The camera's fixed offset from the marble, for a distance and two angles."""
    ground = distance * math.cos(elevation)
    return (ground * math.sin(yaw), distance * math.sin(elevation),
            ground * math.cos(yaw))


def steering_frame(offset):
    """Ground-plane forward/right axes aligned with the camera's screen.

    "Forward" (up arrow) is up-screen — the horizontal direction from the camera
    toward the marble; "right" is screen-right.  Computing them from the camera
    offset keeps the lean intuitive under the yawed isometric view: pushing right
    leans the board toward the right of the screen, whatever the yaw.
    """
    horizontal = np.array([offset[0], 0.0, offset[2]])
    forward = -horizontal / (np.linalg.norm(horizontal) or 1.0)
    right = np.cross(forward, (0.0, 1.0, 0.0))
    right = right / (np.linalg.norm(right) or 1.0)
    return forward, right


# What is held decides the lean; several names per direction so the arrows and
# WASD both work.  The arrow keys arrive under their bracketed special-key names.
FORWARD_KEYS = ("<up>", "w")
BACKWARD_KEYS = ("<down>", "s")
RIGHT_KEYS = ("<right>", "d")
LEFT_KEYS = ("<left>", "a")

#: Under the spin model each press and key-repeat is one kick, in these directions.
ARROW_DIRECTION = {
    "<up>": (1.0, 0.0), "<down>": (-1.0, 0.0),
    "<right>": (0.0, 1.0), "<left>": (0.0, -1.0),
}

MARBLE_CYCLE = ["steel", "chrome", "glass", "rubber", "wood", "ice"]

CONTROLS = """\
Marble — arrows or WASD lean the board; the marble rolls downhill on its own.
  R  restart the run      N  next board      M  cycle the marble material
  C  switch between leaning the board and kicking spin       Esc  quit
"""


class MarbleContext(BaseContext):
    """An OpenGLContext window that plays a generated marble level."""

    # Class attributes set by main() from the command line.
    marble_name = "steel"
    seed = 1
    difficulty = 2
    #: A board file to play instead of a generated one, as the editor's "Play
    #: it" passes.  N generates the next seed, which is how a player gets back
    #: to the endless boards from an authored one.
    board_path = None
    control = TILT
    base_tilt = math.degrees(math.atan(0.22))
    player_tilt = 26.0
    camera_distance = CAMERA_DISTANCE
    damping = ROLL_DAMPING

    def OnInit(self):
        disable_vsync()

        # The follow camera is the sole driver of context.platform, so unbind the
        # default free-fly movement manager or the two fight over the camera.
        if self.movementManager is not None:
            self.movementManager.unbind(self)
            self.movementManager = None
        self.view_offset = camera_offset(self.camera_distance)
        self.camera = FollowCamera(self.platform, offset=self.view_offset,
                                   pull_back=CAMERA_PULL_BACK,
                                   pull_back_speed=CAMERA_PULL_BACK_SPEED)
        # A tight near/far around this small scene keeps depth precision high; the
        # default far (50000) wastes the depth buffer and z-fights coincident tiles.
        self.platform.setFrustum(fieldOfView=CAMERA_FOV, near=1.0,
                                 far=8.0 * self.camera_distance)

        self.level_number = 1
        self._marble_i = MARBLE_CYCLE.index(self.marble_name) \
            if self.marble_name in MARBLE_CYCLE else 0
        self._build_game()

        # Under the spin model a press or key-repeat is one kick, so those still
        # need handlers.  The lean needs none: the engine samples every key into
        # the input state already, and OnIdle asks it what is held.
        for key in ARROW_DIRECTION:
            self.addEventHandler("keyboard", name=key, state=1, function=self._on_arrow)
        self.addEventHandler("keypress", name="r", function=self._on_reset)
        self.addEventHandler("keypress", name="n", function=self._on_next)
        self.addEventHandler("keypress", name="m", function=self._on_cycle_material)
        self.addEventHandler("keypress", name="c", function=self._on_cycle_control)
        self.keyRepeatDelay = 0.1
        self._last = time.time()
        print(CONTROLS)

    # -- game/scene -----------------------------------------------------
    def _build_game(self):
        level = self._level()
        forward, right = steering_frame(self.view_offset)
        self.game = MarbleGame(level, marble_material=self.marble_name,
                               camera=self.camera, control=self.control,
                               base_tilt=math.radians(self.base_tilt),
                               player_tilt=math.radians(self.player_tilt),
                               damping=tuple(self.damping),
                               steer_forward=forward, steer_right=right)
        self.hud = HUD(self.game)
        self.camera.release()
        self.camera.target(self.game.scene.world.position[self.game.marble.index])
        self.camera.apply()

        sun = basenodes.DirectionalLight(direction=(-0.4, -1, -0.5),
                                         color=(1, 0.98, 0.9), intensity=0.9)
        self.game.scene.advance(0.0)
        self.sg = self.game.scene_graph(extra=[sun])
        print(f"Level {self.level_number} ({level.name}): {len(level.cells)} tiles, "
              f"{level.time_limit:.0f}s limit, {self.control} control")

    def _level(self):
        """The board to play: the file that was named, else a generated one."""
        if self.board_path:
            return levelfile.load(self.board_path)
        return generator.generate(seed=self.seed, difficulty=self.difficulty)

    # -- overlay hook (called by the FlatPass after the scene draws) ----
    def renderShaderOverlay(self, flatpass):
        self.hud.render(flatpass, self)

    # -- input ----------------------------------------------------------
    def _lean_demand(self):
        """What the player is asking of the board right now, in [-1, 1] per axis.

        Read from the engine's sampled input state rather than from key events, so
        holding two directions leans the board diagonally and a lost window drops
        the lean instead of leaving it stuck on.
        """
        state = self.getInputState()
        return (state.axis(FORWARD_KEYS, BACKWARD_KEYS),
                state.axis(RIGHT_KEYS, LEFT_KEYS))

    def _on_arrow(self, event):
        """One spin kick — the spin model only; the lean is sampled, not evented."""
        if self.game.control == SPIN:
            self.game.kick(*ARROW_DIRECTION[event.name])
            self.triggerRedraw(1)

    def _on_reset(self, event):
        self.game.reset()
        self.triggerRedraw(1)

    def _on_next(self, event):
        """Advance to the next generated level, leaving any named board behind."""
        self.board_path = None
        self.seed += 1
        self.level_number += 1
        self._build_game()
        self.triggerRedraw(1)

    def _on_cycle_material(self, event):
        self._marble_i = (self._marble_i + 1) % len(MARBLE_CYCLE)
        self.marble_name = MARBLE_CYCLE[self._marble_i]
        self.game.set_marble_material(self.marble_name)
        self.triggerRedraw(1)

    def _on_cycle_control(self, event):
        """Switch control model on the spot, so the two can be felt back to back.

        Bounding the rig's player lean at zero *is* the spin model, so the switch
        is that bound and nothing else; the board levels itself on the way.
        """
        self.control = SPIN if self.game.control == TILT else TILT
        self.game.control = self.control
        self.game.tilt.limit = math.radians(self.player_tilt) \
            if self.control == TILT else 0.0
        self.game.tilt.level()
        self.game.lean(0.0, 0.0)
        print(f"control: {self.control}")
        self.triggerRedraw(1)

    # -- loop -----------------------------------------------------------
    def OnIdle(self, *args):
        now = time.time()
        dt = min(now - self._last, 0.05)
        self._last = now

        was_playing = self.game.state == PLAYING
        self.game.lean(*self._lean_demand())
        self.game.advance(dt)
        self.camera.advance(dt, self.game.controller.speed)
        self.camera.apply()
        if was_playing and self.game.state != PLAYING:
            print(f"Run {self.game.state}! time left {self.game.time_left:.1f}s, "
                  f"falls {self.game.controller.fall_count}")
        self.triggerRedraw(1)      # gravity always acts, so always redraw
        return 1


def build_parser():
    parser = argparse.ArgumentParser(description="Marble Madness demo for OpenGLContext")
    parser.add_argument("--marble", default="steel", choices=sorted(materials.MARBLES),
                        help="marble material to start with")
    parser.add_argument("--board", default=None, dest="board_path",
                        metavar="FILE",
                        help="play a board file (see oglc-marble-editor) "
                             "rather than a generated one")
    parser.add_argument("--seed", type=int, default=1, help="level seed")
    parser.add_argument("--difficulty", type=int, default=2,
                        help="level difficulty (longer, steeper tracks)")
    parser.add_argument("--control", default=TILT, choices=(TILT, SPIN),
                        help="lean the board, or impart spin with each press "
                             "(C switches while playing)")
    parser.add_argument("--tilt", type=float, default=math.degrees(math.atan(0.22)),
                        dest="base_tilt", metavar="DEGREES",
                        help="how far the board leans downhill; sets the pace")
    parser.add_argument("--lean", type=float, default=26.0, dest="player_tilt",
                        metavar="DEGREES",
                        help="how far the player may lean it on top of that")
    parser.add_argument("--camera-distance", type=float, default=CAMERA_DISTANCE,
                        metavar="METRES", help="how far back the camera sits")
    parser.add_argument("--damping", nargs=2, type=float, default=ROLL_DAMPING,
                        metavar=("LINEAR", "ANGULAR"),
                        help="how freely the marble rolls, per second; the "
                             "scenegraph manager's own defaults are 0.3 1.5")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    MarbleContext.marble_name = args.marble
    MarbleContext.board_path = args.board_path
    MarbleContext.seed = args.seed
    MarbleContext.difficulty = args.difficulty
    MarbleContext.control = args.control
    MarbleContext.base_tilt = args.base_tilt
    MarbleContext.player_tilt = args.player_tilt
    MarbleContext.camera_distance = args.camera_distance
    MarbleContext.damping = tuple(args.damping)
    MarbleContext.ContextMainLoop()


if __name__ == "__main__":
    main()
