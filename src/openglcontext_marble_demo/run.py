"""Run loop and window for the marble demo.

``run.py`` is the *window* around the tested, headless game: it builds a
:class:`~openglcontext_marble_demo.game.MarbleGame` from a generated level, drives
the near-isometric follow camera, samples the keyboard, steps the game once per
frame, and paints the HUD.

The interesting rules — the board's lean, fall/respawn, the timer, win/lose, the
mechanisms — all live in the game/tilt/controller/level modules and are unit
tested; this file is the shape a programmer copies to drive OpenGLContext's
physics + PBR from their own input.

**It plays itself when asked.** ``--demo`` hands the board to
:class:`~openglcontext_marble_demo.pilot.Autopilot`, which leans it through
exactly the controls a player holds -- so an attract mode, and a recorded video,
are the game rather than a scripted animation of it.

**It records.** ``--record run.mp4`` writes the frames straight from the
framebuffer to the GPU's video encoder, and ``--telemetry run.jsonl`` writes down
every input against the frame that acted on it, so a run can be read back and run
again (``--replay run.jsonl``).

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
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, ClassVar

# Core profile + GLFW + the PBR renderer with image-based lighting give reflective
# marbles and shadows; set before OpenGLContext imports a backend so it takes effect.
os.environ.setdefault("OPENGLCONTEXT_BACKEND", "glfw")
os.environ.setdefault("OPENGLCONTEXT_RENDERER", "pbr")
os.environ.setdefault("OPENGLCONTEXT_IBL", "full")
# Every session is recorded unless the caller says otherwise.  A gameplay fault
# is a thing that happened at a moment -- the marble was told it had gone over an
# edge it was visibly standing on -- and a journal is the only way to say which
# moment and what the game thought was true at it.  ``1`` means a dated file
# under the user's application-data directory; the path is printed at startup so
# it can be quoted in a report.  ``--no-telemetry`` turns it off.
os.environ.setdefault("OPENGLCONTEXT_TELEMETRY", "1")

import numpy as np
from OpenGLContext import testingcontext
from OpenGLContext.events.systemtime import systemTime
from OpenGLContext.move.followcam import FollowCamera
from OpenGLContext.physics.demo import disable_vsync
from OpenGLContext.scenegraph import basenodes
from OpenGLContext.video.recorder import RecordingMixin

from . import generator, levelfile, materials, pilot
from .game import BASE_TILT, PLAYER_TILT, PLAYING, ROLL_DAMPING, SPIN, TILT, MarbleGame
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


def _say_where_the_journal_is(context: Any) -> None:
    """Print the session journal's path, if this session is being recorded.

    Printed rather than logged: it is the one thing a player needs to be able to
    quote when something goes wrong, and a warning in a log they are not
    watching is a path they will not have.
    """
    journal = getattr(getattr(context, 'telemetry', None), 'journal', None)
    path = getattr(journal, 'path', None)
    if path is not None and not getattr(journal, 'disabled', False):
        print('Recording this session to %s' % (path,))


def camera_offset(distance: Any=CAMERA_DISTANCE, elevation: Any=CAMERA_ELEVATION,
                  yaw: Any=CAMERA_YAW) -> Any:
    """The camera's fixed offset from the marble, for a distance and two angles."""
    ground = distance * math.cos(elevation)
    return (ground * math.sin(yaw), distance * math.sin(elevation),
            ground * math.cos(yaw))


def steering_frame(offset: Any) -> Any:
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

#: How long the demo holds on a finished run before starting the next board.
#: Long enough to read what happened, short enough that a recording is mostly
#: play.
DEMO_PAUSE = 2.0

CONTROLS = """\
Marble — arrows or WASD lean the board; the board is level until you lean it.
  R  restart the run      N  next board      M  cycle the marble material
  C  lean the board / kick spin     D  let the autopilot play      Esc  quit
"""


class MarbleContext(RecordingMixin, BaseContext):
    """An OpenGLContext window that plays a generated marble level."""

    # Class attributes set by main() from the command line.
    marble_name = "steel"
    seed = 1
    difficulty = 2
    #: A board file to play instead of a generated one, as the editor's "Play
    #: it" passes.  N generates the next seed, which is how a player gets back
    #: to the endless boards from an authored one.
    board_path = None
    #: Let the autopilot play, which is what an attract mode and a recording
    #: want.  D switches it on and off mid-run.
    demo = False
    #: Where to record the run to, and how; None for an ordinary run.
    record_path = None
    record_options: ClassVar[Mapping[str, Any]] = MappingProxyType({})
    control = TILT
    # From the game's own constants rather than restated here.  Restated, they
    # went stale: the game was retuned to an 8-degree lean and 40 degrees of
    # player tilt -- which is what makes the board aimable, and what every
    # measurement in the test suite is taken against -- and the program went on
    # running the 12.4 and 26 it had been given, so the one configuration nobody
    # was testing was the one people played.
    base_tilt = math.degrees(BASE_TILT)
    player_tilt = math.degrees(PLAYER_TILT)
    camera_distance = CAMERA_DISTANCE
    damping = ROLL_DAMPING

    def OnInit(self) -> None:
        disable_vsync()
        _say_where_the_journal_is(self)

        # The follow camera is the sole driver of context.platform, so unbind the
        # default free-fly movement manager or the two fight over the camera.
        # Read through the class this mixes with rather than off `self`:
        # a checker settles an attribute's type from the first assignment
        # it sees, and the one below is `None`.
        manager: Any = getattr(self, 'movementManager', None)
        if manager is not None:
            manager.unbind(self)
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
        self.addEventHandler("keypress", name="d", function=self._on_toggle_demo)
        self.keyRepeatDelay = 0.1
        #: How many times the marble had gone over the edge when last looked at,
        #: so a fall is marked once rather than every frame after it.
        self._falls = 0
        #: When the run ended, so the demo can hold on the result before moving
        #: on; None while one is being played.
        #: When the run ended, or None while it is being played.
        self._ended_at: float | None = None
        self._last = systemTime()
        if self.record_path:
            self.setupRecording(self.record_path, **self.record_options)
        print(CONTROLS)

    # -- game/scene -----------------------------------------------------
    def _build_game(self) -> None:
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

        # A pilot for every board, whether or not it is driving: switching the
        # demo on mid-run should not have to build one first.
        self.pilot = pilot.Autopilot(level,
                                     forward_axis=self.game.tilt.forward_axis,
                                     right_axis=self.game.tilt.right_axis)

        sun = basenodes.DirectionalLight(direction=(-0.4, -1, -0.5),
                                         color=(1, 0.98, 0.9), intensity=0.9)
        self.game.scene.advance(0.0)
        self.sg = self.game.scene_graph(extra=[sun])
        # What the engine cannot know: which board this is.  A mark is the line
        # a reader looks for first in a journal, and a replay checks each of
        # them against the one recorded in its place.
        self.mark('board-loaded', name=level.name, tiles=len(level.cells),
                  seconds=level.time_limit, control=self.control,
                  demo=bool(self.demo))
        print(f"Level {self.level_number} ({level.name}): {len(level.cells)} tiles, "
              f"{level.time_limit:.0f}s limit, {self.control} control")

    def _level(self) -> Any:
        """The board to play: the file that was named, else a generated one."""
        if self.board_path:
            return levelfile.load(self.board_path)
        return generator.generate(seed=self.seed, difficulty=self.difficulty)

    # -- overlay hook (called by the FlatPass after the scene draws) ----
    def renderShaderOverlay(self, flatpass: Any) -> None:
        self.hud.render(flatpass, self)

    # -- input ----------------------------------------------------------
    def lean_demand(self) -> Any:
        """What the player is asking of the board right now, in [-1, 1] per axis.

        Read from the engine's sampled input state rather than from key events, so
        holding two directions leans the board diagonally and a lost window drops
        the lean instead of leaving it stuck on.
        """
        state = self.getInputState()
        return (state.axis(FORWARD_KEYS, BACKWARD_KEYS),
                state.axis(RIGHT_KEYS, LEFT_KEYS))

    def _on_toggle_demo(self, event: Any) -> None:  # noqa: ARG002 OpenGLContext key-event callback signature
        """Hand the board to the autopilot, or take it back."""
        self.demo = not self.demo
        print('demo: %s' % ('on' if self.demo else 'off'))
        self.mark('demo', on=bool(self.demo))
        self.triggerRedraw(1)

    def _on_arrow(self, event: Any) -> None:
        """One spin kick — the spin model only; the lean is sampled, not evented."""
        if self.game.control == SPIN:
            self.game.kick(*ARROW_DIRECTION[event.name])
            self.triggerRedraw(1)

    def _demand(self) -> Any:
        """Who is steering: the autopilot, or whoever is holding the keys."""
        if not self.demo:
            return self.lean_demand()
        world = self.game.scene.world
        index = self.game.marble.index
        return self.pilot.lean(world.position[index],
                               world.linear_velocity[index])

    def _on_reset(self, event: Any) -> None:  # noqa: ARG002 OpenGLContext key-event callback signature
        self.game.reset()
        self.triggerRedraw(1)

    def _on_next(self, event: Any) -> None:  # noqa: ARG002 OpenGLContext key-event callback signature
        """Advance to the next generated level, leaving any named board behind."""
        self.board_path = None
        self.seed += 1
        self.level_number += 1
        self._ended_at = None
        self._build_game()
        self.triggerRedraw(1)

    def _on_cycle_material(self, event: Any) -> None:  # noqa: ARG002 OpenGLContext key-event callback signature
        self._marble_i = (self._marble_i + 1) % len(MARBLE_CYCLE)
        self.marble_name = MARBLE_CYCLE[self._marble_i]
        self.game.set_marble_material(self.marble_name)
        self.triggerRedraw(1)

    def _on_cycle_control(self, event: Any) -> None:  # noqa: ARG002 OpenGLContext key-event callback signature
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

    def presentFrame(self) -> Any:
        """Present the frame, and give it to the recording first.

        The back buffer holds the finished frame only until it is swapped away,
        so a recording takes it here rather than on a clock.
        """
        if self.recording:
            self.tickRecording()
        return super().presentFrame()

    # -- loop -----------------------------------------------------------
    def OnIdle(self, *args: Any) -> Any:  # noqa: ARG002 OpenGLContext OnIdle callback signature
        # The engine's clock, not `time.time()`.  It is the same wall clock for
        # an ordinary run, and it is the *recorded* one under a replay and a
        # fixed step per frame under a recording -- so a run that is a function
        # of its input and its clock runs again as it ran, and a video advances
        # the world by exactly one frame of time per frame it keeps.
        now = systemTime()
        dt = min(now - self._last, 0.05)
        self._last = now

        was_playing = self.game.state == PLAYING
        self.game.lean(*self._demand())
        self.game.advance(dt)
        self.camera.advance(dt, self.game.controller.speed)
        self.camera.apply()
        if was_playing and self.game.state != PLAYING:
            print(f"Run {self.game.state}! time left {self.game.time_left:.1f}s, "
                  f"falls {self.game.controller.fall_count}")
            # Outcome and falls, and deliberately not the clock: a mark is
            # what a replay is checked against, and a float that accumulates a
            # hundredth of a second over six hundred frames makes a check that
            # always fails.  What has to reproduce is what happened.
            self.mark('run-ended', outcome=self.game.state,
                      falls=self.game.controller.fall_count)
            self._ended_at = now
        falls = self.game.controller.fall_count
        if falls != self._falls:
            self._falls = falls
            self.mark('fell', count=falls)
        # An attract mode that stopped on the first finish would be a frozen
        # screen for the rest of the recording, so the demo goes round: a beat
        # on the result, then the next board.
        if self.demo and self._ended_at is not None \
                and now - self._ended_at >= DEMO_PAUSE:
            self._ended_at = None
            self._on_next(None)
        self.triggerRedraw(1)      # gravity always acts, so always redraw
        return 1


def build_parser() -> Any:
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
    parser.add_argument("--tilt", type=float, default=math.degrees(BASE_TILT),
                        dest="base_tilt", metavar="DEGREES",
                        help="how far the board leans downhill; sets the pace")
    parser.add_argument("--lean", type=float, default=math.degrees(PLAYER_TILT),
                        dest="player_tilt", metavar="DEGREES",
                        help="how far the player may lean it on top of that")
    parser.add_argument("--camera-distance", type=float, default=CAMERA_DISTANCE,
                        metavar="METRES", help="how far back the camera sits")
    parser.add_argument("--damping", nargs=2, type=float, default=ROLL_DAMPING,
                        metavar=("LINEAR", "ANGULAR"),
                        help="how freely the marble rolls, per second; the "
                             "scenegraph manager's own defaults are 0.3 1.5")
    parser.add_argument("--size", nargs=2, type=int, default=None,
                        metavar=("WIDTH", "HEIGHT"),
                        help="the window's size in pixels, which is a "
                             "recording's size too")
    parser.add_argument("--demo", action="store_true",
                        help="let the autopilot play (D switches it mid-run)")

    recording = parser.add_argument_group(
        "recording", "writing a run down, as video or as a session journal")
    recording.add_argument("--record", metavar="PATH", dest="record_path",
                           help="record the run to PATH (an .mp4) and quit when "
                                "the recording is done")
    recording.add_argument("--record-seconds", type=float, default=20.0,
                           metavar="SECONDS", help="how long a recording runs "
                                                   "for (default: %(default)s)")
    recording.add_argument("--record-fps", type=int, default=60, metavar="FPS",
                           help="frames a second in the recording "
                                "(default: %(default)s)")
    recording.add_argument("--record-delay", type=float, default=1.0,
                           metavar="SECONDS",
                           help="let the scene settle for this long before the "
                                "recording starts (default: %(default)s)")
    recording.add_argument("--record-bitrate", type=int, default=0,
                           metavar="BITS",
                           help="bits a second; 0 lets the encoder choose from "
                                "the frame size and rate")
    recording.add_argument("--telemetry", metavar="PATH",
                           help="record the session -- every input against the "
                                "frame that acted on it -- to PATH (a .jsonl). "
                                "Recording is on by default, to a dated file "
                                "under the application-data directory")
    recording.add_argument("--no-telemetry", action="store_true",
                           help="do not record this session")
    recording.add_argument("--replay", metavar="PATH",
                           help="run a recorded session again, with the same "
                                "input on the same frames")
    return parser


def main(argv: Any=None) -> None:
    args = build_parser().parse_args(argv)
    # Telemetry is read from the environment by the context before OnInit, so
    # the flags are the environment: one way in, whichever the caller used.
    if args.telemetry:
        os.environ["OPENGLCONTEXT_TELEMETRY"] = args.telemetry
    if args.no_telemetry:
        os.environ.pop("OPENGLCONTEXT_TELEMETRY", None)
    if args.replay:
        os.environ["OPENGLCONTEXT_TELEMETRY_REPLAY"] = args.replay
    MarbleContext.marble_name = args.marble
    MarbleContext.board_path = args.board_path
    MarbleContext.seed = args.seed
    MarbleContext.difficulty = args.difficulty
    MarbleContext.control = args.control
    MarbleContext.base_tilt = args.base_tilt
    MarbleContext.player_tilt = args.player_tilt
    MarbleContext.camera_distance = args.camera_distance
    MarbleContext.damping = tuple(args.damping)
    MarbleContext.demo = args.demo
    MarbleContext.record_path = args.record_path
    MarbleContext.record_options = {
        "fps": args.record_fps, "seconds": args.record_seconds,
        "start_after": args.record_delay,
        **({"bitrate": args.record_bitrate} if args.record_bitrate else {}),
    }
    if args.size:
        MarbleContext.ContextMainLoop(size=tuple(args.size))
    else:
        MarbleContext.ContextMainLoop()


if __name__ == "__main__":
    main()
