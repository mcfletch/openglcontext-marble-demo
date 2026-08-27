"""Run loop and window for the marble demo.

``run.py`` is the *window* around the tested, headless game: it builds a
:class:`~openglcontext_marble_demo.game.MarbleGame` from a generated level, drives
the near-isometric follow camera, turns arrow keys into steering, steps the game
once per frame, and paints the HUD.

The interesting rules (steering, fall/respawn, timer, win/lose, mechanisms) all live
in the game/controller/level modules and are unit tested; this file is the shape a
programmer copies to drive OpenGLContext's physics + PBR from their own input.
"""
import argparse
import math
import os
import time
from typing import Any

import numpy as np

# Core profile + GLFW + the PBR renderer with image-based lighting give reflective
# marbles and shadows; set before OpenGLContext imports a backend so it takes effect.
os.environ.setdefault("OPENGLCONTEXT_BACKEND", "glfw")
os.environ.setdefault("OPENGLCONTEXT_RENDERER", "pbr")
os.environ.setdefault("OPENGLCONTEXT_IBL", "full")

from OpenGLContext import testingcontext
from OpenGLContext.move.followcam import FollowCamera
from OpenGLContext.physics.demo import disable_vsync
from OpenGLContext.scenegraph import basenodes

from . import generator, materials
from .game import PLAYING, MarbleGame
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
CAMERA_DISTANCE = 24.0
_ground = CAMERA_DISTANCE * math.cos(CAMERA_ELEVATION)
CAMERA_OFFSET = (_ground * math.sin(CAMERA_YAW),
                 CAMERA_DISTANCE * math.sin(CAMERA_ELEVATION),
                 _ground * math.cos(CAMERA_YAW))
CAMERA_FOV = math.radians(30)


def _steering_frame(offset):
    """Ground-plane forward/right axes aligned with the camera's screen.

    "Forward" (up arrow) is up-screen — the horizontal direction from the camera
    toward the marble; "right" is screen-right.  Computing them from the camera
    offset keeps steering intuitive under the yawed isometric view.
    """
    horizontal = np.array([offset[0], 0.0, offset[2]])
    forward = -horizontal / (np.linalg.norm(horizontal) or 1.0)
    right = np.cross(forward, (0.0, 1.0, 0.0))
    right = right / (np.linalg.norm(right) or 1.0)
    return forward, right

# Each arrow key maps to a (forward, right) steering direction; a press or key-repeat
# fires one steering kick.  Up is "up the slope" (away from the viewer).  The arrow
# keys arrive under their bracketed special-key names (see glfwevents keyboardMapping).
ARROW_DIRECTION = {
    "<up>": (1.0, 0.0), "<down>": (-1.0, 0.0),
    "<right>": (0.0, 1.0), "<left>": (0.0, -1.0),
}

MARBLE_CYCLE = ["steel", "chrome", "glass", "rubber", "wood", "ice"]


class MarbleContext(BaseContext):
    """An OpenGLContext window that plays a generated marble level."""

    # Class attributes set by main() from the command line.
    marble_name = "steel"
    seed = 1
    difficulty = 2

    def OnInit(self):
        disable_vsync()

        # The follow camera is the sole driver of context.platform, so unbind the
        # default free-fly movement manager or the two fight over the camera.
        if self.movementManager is not None:
            self.movementManager.unbind(self)
            self.movementManager = None
        self.camera = FollowCamera(self.platform, offset=CAMERA_OFFSET)
        # A tight near/far around this small scene keeps depth precision high; the
        # default far (50000) wastes the depth buffer and z-fights coincident tiles.
        self.platform.setFrustum(fieldOfView=CAMERA_FOV, near=1.0, far=300.0)

        self.level_number = 1
        self._marble_i = MARBLE_CYCLE.index(self.marble_name) \
            if self.marble_name in MARBLE_CYCLE else 0
        self._build_game()

        # Steering is discrete: each arrow press/repeat fires one kick (key-repeat
        # gives sustained steering).  Bind on 'keyboard' state=1, which the backend
        # re-emits on repeat.
        for key in ARROW_DIRECTION:
            self.addEventHandler("keyboard", name=key, state=1, function=self._on_arrow)
        self.addEventHandler("keypress", name="r", function=self._on_reset)
        self.addEventHandler("keypress", name="n", function=self._on_next)
        self.addEventHandler("keypress", name="m", function=self._on_cycle_material)
        self.keyRepeatDelay = 0.1
        self._last = time.time()
        print(__doc__)

    # -- game/scene -----------------------------------------------------
    def _build_game(self):
        level = generator.generate(seed=self.seed, difficulty=self.difficulty)
        steer_forward, steer_right = _steering_frame(CAMERA_OFFSET)
        self.game = MarbleGame(level, marble_material=self.marble_name,
                               camera=self.camera, steer_forward=steer_forward,
                               steer_right=steer_right)
        self.hud = HUD(self.game)
        self.camera.release()
        self.camera.target(self.game.scene.world.position[self.game.marble.index])
        self.camera.apply()

        sun = basenodes.DirectionalLight(direction=(-0.4, -1, -0.5),
                                         color=(1, 0.98, 0.9), intensity=0.9)
        self.game.scene.advance(0.0)
        self.sg = self.game.scene_graph(extra=[sun])
        print(f"Level {self.level_number} ({level.name}): {len(level.cells)} tiles, "
              f"{level.time_limit:.0f}s limit")

    # -- overlay hook (called by the FlatPass after the scene draws) ----
    def renderShaderOverlay(self, flatpass):
        self.hud.render(flatpass, self)

    # -- input ----------------------------------------------------------
    def _on_arrow(self, event):
        forward, right = ARROW_DIRECTION[event.name]
        self.game.kick(forward, right)
        self.triggerRedraw(1)

    def _on_reset(self, event):
        self.game.reset()
        self.triggerRedraw(1)

    def _on_next(self, event):
        """Advance to the next generated level."""
        self.seed += 1
        self.level_number += 1
        self._build_game()
        self.triggerRedraw(1)

    def _on_cycle_material(self, event):
        self._marble_i = (self._marble_i + 1) % len(MARBLE_CYCLE)
        self.marble_name = MARBLE_CYCLE[self._marble_i]
        self.game.set_marble_material(self.marble_name)
        self.triggerRedraw(1)

    # -- loop -----------------------------------------------------------
    def OnIdle(self, *args):
        now = time.time()
        dt = min(now - self._last, 0.05)
        self._last = now

        was_playing = self.game.state == PLAYING
        self.game.advance(dt)
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
    parser.add_argument("--seed", type=int, default=1, help="level seed")
    parser.add_argument("--difficulty", type=int, default=2,
                        help="level difficulty (longer, steeper tracks)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    MarbleContext.marble_name = args.marble
    MarbleContext.seed = args.seed
    MarbleContext.difficulty = args.difficulty
    MarbleContext.ContextMainLoop()


if __name__ == "__main__":
    main()
