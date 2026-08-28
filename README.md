# openglcontext-marble-demo

A modern re-interpretation of the 1984 Atari/Amiga classic **Marble Madness**,
built on [OpenGLContext](http://pyopengl.sourceforge.net/context/): roll a
reflective PBR marble down a **procedurally generated**, gently-tilted track and
race a countdown to the finish. Steer by imparting spin with the arrow keys, ride
ramps and elevators, bounce off rubber bumpers, and try not to fall off — a fall
costs you a respawn penalty.

It is a **scene package**: the reusable simulation and rendering live in
`OpenGLContext` (PBR/IBL renderer, follow camera, scenegraph) and `omi_physics`
(the rigid-body engine), and this package supplies the *game* — the level
generator, the marble rules, the HUD, and the run loop. It doubles as a worked
example of driving that physics and PBR from your own input.

## Run it

From a checkout of the [OpenGL-dev](https://github.com/mcfletch/OpenGL-dev)
workspace, where this is a submodule and `uv sync` installs the whole stack
editable:

```bash
uv sync                          # at the workspace root
oglc-marble                      # or: python -m openglcontext_marble_demo
oglc-marble --seed 7 --difficulty 3 --marble chrome
```

Standalone, against the development branch of the engine:

```bash
pip install --pre "PyOpenGL @ git+https://github.com/mcfletch/pyopengl@develop" \
                  "OpenGLContext @ git+https://github.com/mcfletch/openglcontext@develop"
pip install --pre -e .
```

`--help` lists the options: `--seed` and `--difficulty` pick the generated
board, `--marble` the starting material, and `--board FILE` plays a board drawn
in [the editor](https://github.com/mcfletch/marble-editor) instead of a
generated one.

```bash
oglc-marble --board spiral.marble
```

## Controls

| Key | Action |
|-----|--------|
| **↑ / ↓** | steer up-slope (away) / down-slope (toward you) |
| **← / →** | steer left / right |
| **R** | reset the run (marble back to the start, clock refilled) |
| **N** | next level (a new generated seed) |
| **M** | cycle the marble material (steel → chrome → glass → rubber → wood → ice) |
| **D** | let the autopilot play |
| **Esc** | quit |

Each arrow **press imparts a spin kick**; holding a key re-fires it (key-repeat) for
sustained steering. The whole board tilts toward the camera, so — like the original
— the marble rolls toward you on its own and steering is about *directing that
momentum*, not driving from a standstill. How much a kick actually moves the marble
depends on the marble↔surface friction: a rubber marble bites and turns sharply, a
steel marble on ice barely responds and drifts.

## Watching it, and reading a run back

The game plays itself when asked, records what it draws, and writes a session
down so a run can be read back and run again.

```bash
oglc-marble --demo                          # the autopilot plays; D toggles it
oglc-marble --demo --record run.mp4 --record-seconds 30 --size 1280 720
oglc-marble --telemetry run.jsonl           # write the session down
oglc-marble --replay run.jsonl              # run it again, same input, same frames
python -m OpenGLContext.telemetry run.jsonl # read it
```

The **autopilot** (`pilot.py`) reads the board, works out a line across it, and
leans the board along that line — through exactly the controls a player holds,
bounded to what a held key gives. So a recording of it is a recording of the
game rather than of an animation, and an attract mode is the game playing.

**Recording** goes straight from the framebuffer to the GPU's H.264 encoder
(`OpenGLContext[video]`), and installs a clock that advances by exactly one
frame's worth per frame kept — so a video is smooth however fast the machine
drew it.

**Telemetry** records every input against the frame that acted on it, and the
game marks what the engine cannot know: which board loaded, when one was fallen
off, how a run ended. A replay answers each mark with the one recorded in its
place and says how the two accounts compared:

```
replay of run.jsonl: 5 marks, all as recorded
```

That check is why the run loop reads the engine's clock
(`OpenGLContext.events.systemtime`) rather than `time.time()`: a game that reads
the wall clock itself replays approximately, and this one replays exactly.

## How it plays

- **Timer.** Each level has a countdown (top-right HUD). Reach the green finish pad
  before it hits zero. Time out and you lose; press **R** to retry.
- **Falling off.** Leave the track and you drop into the void; the camera holds on
  the last square while a short **respawn penalty** elapses, then you restart there.
  The delay is deliberate — you can't dump unwanted speed by driving off the edge.
- **Jumps.** Hit a launch ramp fast and the marble sails in a real ballistic arc
  over the tiles ahead. That is *not* a fall — the game tells a jump apart from
  going over the edge.
- **Speed.** Ramps boost you; hitting a wall or landing hard from a height scrubs
  your speed; rubber bumpers give it back.
- **Mechanisms.** Elevators carry you up and down, rotating arms sweep you
  sideways, spring traps fling you. Sand takes eight times as long to churn
  through as plain floor, unless you are launched over it. A lever opens its
  door when struck at 4 m/s or harder — and the blow stops you, so a run at one
  costs a pass. Water sinks you at a third of the speed of falling, and its plug
  gives when you reach it, which you cannot know until you have.

## How the code is organized

```
marble-demo/
  src/openglcontext_marble_demo/
    run.py          # the window: context, follow camera, input, per-frame loop
    game.py         # MarbleGame: ties level + physics + controller + clock + win/lose
    controller.py   # MarbleController: spin-to-steer, grounding, fall/respawn
    generator.py    # procedural level generator (seeded)
    level.py        # Level data + features (Floor/Ramp/Wall/Bumper/Spring/Elevator/Arm/Finish)
    track.py        # TrackMap: the grid of occupied cells the controller reasons about
    materials.py    # marble + surface material table, incl. the pairwise friction grid
    render.py       # PBR appearances + the grouted tile mesh
    hud.py          # the countdown / speed / material HUD (drawn via the overlay hook)
    assets/         # generated stand-in assets (grout color + normal maps)
  tools/            # committed asset & capture pipelines (make_grout.py, capture.py)
  tests/            # the test suite (headless; steering/physics exercised for real)
```

The split is deliberate: everything except `run.py` (and the GL bits of `render.py`
and `hud.py`) is **headless and unit-tested** — the steering, fall/respawn, timer,
generator, and mechanisms are all exercised against the real physics world with no
window. `run.py` is just the glue that turns key presses into game calls and paints
frames.

### The engine side

The demo leans on machinery in `OpenGLContext` itself, some of it added for this
demo (all generic and reusable):

- `omi_physics` — the rigid-body engine that does gravity, sphere collision,
  friction/restitution, rolling, and ballistic jumps, reached through
  `OpenGLContext.physics`, which ties it to the scenegraph. The demo drives
  `apply_impulse`/`apply_angular_impulse` (steering, boosts, traps), the
  **pairwise friction override** (`set_pair_friction`, the marble↔surface grip
  grid), and `KinematicAnimator` (elevators, arms).
- `OpenGLContext.move.followcam` — the fixed-offset follow camera with hold/release.
- The PBR renderer (`OPENGLCONTEXT_RENDERER=pbr`) + image-based lighting
  (`OPENGLCONTEXT_IBL=full`), which `run.py` enables, give the reflective marbles.
- A small generic `renderShaderOverlay(pass)` hook on the FlatPass lets `hud.py`
  paint the HUD over the finished frame.

The original design and phase-by-phase build log live in the engine repo at
`plans/MARBLE-MADNESS-DEMO.md`. Where the game goes next — the control model, the
board generator and the screens around the run — is
[`plans/STANDALONE-GAME.md`](plans/STANDALONE-GAME.md).

## Working with the code

Run the tests. The render smoke test opens a real window and skips itself where
there is no display; everything else is headless:

```bash
cd marble-demo
python -m pytest
```

Lint and types are gates in CI, configured in `pyproject.toml`:

```bash
ruff check .
mypy src/openglcontext_marble_demo
```

`.github/workflows/test.yml` runs all three on 3.10 through 3.13, against the
**development branch of the engine** rather than its last PyPI release — the game
uses engine features that went in for it, and tracking the tip is what reports a
moved module on the day it moves. `release.yml` runs the same checks on a push to
`main` and publishes to PyPI when `__version__` names a release that is not there
yet.

Regenerate the grout textures (a committed, reproducible asset pipeline):

```bash
python tools/make_grout.py            # -> src/.../assets/grout_{color,normal}.png
```

Capture a screenshot of a generated level:

```bash
python tools/capture.py --seed 7 --difficulty 3 --out shot.png
```

### Extending it

- **A new mechanism** is a new module in `mechanisms/`, holding a dataclass with
  `owned_cells()` and `build(scene, level, index, result)` and decorated with
  `@mechanism('name')`. Registering is also what teaches the file format to read
  and write it, so there is no way to add one that no board can save. Record a
  trigger effect in `result.effects`, a `KinematicAnimator` in
  `result.animators`, anything holding state a restarted run must forget in
  `result.resettable`, and anything that has to find another mechanism in
  `result.channels`.
- **A new story fragment** is a new module in `fragments/`, decorated with
  `@fragment(name=..., tags=..., variants=...)`. Both packages are found by
  scanning their own directory, so a new file is a new entry and no shared file
  records it — which is what lets several be built at once without conflicting.
- **A new marble/surface** is a row in `materials.MARBLES` / `SURFACES` plus a
  column in the pairwise friction grid.
- **A new generator style** replaces `generator._carve_path` / `_decorate`; keep the
  cells connected and the tests (determinism, solvability) will hold.

## Licensing

The source code is MIT (`LICENSE`). Art assets and their licenses are listed in
[`ASSET-LICENSES.md`](ASSET-LICENSES.md); the shipped assets are procedurally
generated (CC0) stand-ins — see that file for the higher-quality finals worth
sourcing.
