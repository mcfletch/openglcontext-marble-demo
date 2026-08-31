#!/usr/bin/env python
"""Render a generated level headlessly and save a screenshot.

A committed capture pipeline (used for README screenshots and quick visual checks),
not a throwaway script.  It drives the real demo context under OpenGLContext's
auto-exit capture harness, so what it saves is exactly what the game renders.

    python tools/capture.py --seed 7 --difficulty 3 --out shot.png
    python tools/capture.py --marble chrome --frames 90 --out chrome.png
    python tools/capture.py --size 1600x900 --out wide.png
"""
import argparse
import os


def window_size(text):
    """``WIDTHxHEIGHT``, for a picture wider than the window the game opens."""
    try:
        width, height = (int(part) for part in text.lower().split("x", 1))
    except ValueError:
        raise argparse.ArgumentTypeError(
            "expected WIDTHxHEIGHT, e.g. 1600x900, not %r" % (text,)) from None
    if width < 1 or height < 1:
        raise argparse.ArgumentTypeError("a window has to have some size in it")
    return (width, height)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--difficulty", type=int, default=3)
    parser.add_argument("--marble", default="steel")
    parser.add_argument("--frames", type=int, default=60,
                        help="frames to simulate before capturing")
    parser.add_argument("--control", default=None, choices=("tilt", "spin"))
    parser.add_argument("--camera-distance", type=float, default=None,
                        metavar="METRES", help="how far back the camera sits")
    parser.add_argument("--lean", nargs=2, type=float, default=None,
                        metavar=("FORWARD", "RIGHT"),
                        help="hold this lean, in [-1, 1] per axis, while the "
                             "frames run -- for picturing the board leaning")
    parser.add_argument("--size", type=window_size, default=None,
                        metavar="WIDTHxHEIGHT",
                        help="render at this size rather than the game's default")
    parser.add_argument("--out", default="marble.png")
    args = parser.parse_args(argv)

    # The auto-exit harness writes "<name>.png" into the capture dir during the run
    # (the loop exits the process on capture), so point it straight at the target.
    out = os.path.abspath(args.out)
    name = os.path.basename(out)
    if name.endswith(".png"):
        name = name[:-4]
    os.environ.update(
        OPENGLCONTEXT_PROFILE="core", OPENGLCONTEXT_BACKEND="glfw",
        OPENGLCONTEXT_RENDERER="pbr", OPENGLCONTEXT_IBL="full",
        OPENGLCONTEXT_DISABLE_FPS_DISPLAY="1",
        OPENGLCONTEXT_AUTO_EXIT_FRAMES=str(args.frames),
        OPENGLCONTEXT_AUTO_EXIT_CAPTURE_DIR=os.path.dirname(out) or ".",
        OPENGLCONTEXT_AUTO_EXIT_CAPTURE_NAME=name,
    )
    print(f"capturing seed={args.seed} difficulty={args.difficulty} "
          f"marble={args.marble} -> {out}")

    # Import after the env is set so the backend/renderer choice takes effect.
    from openglcontext_marble_demo.run import MarbleContext
    MarbleContext.seed = args.seed
    MarbleContext.difficulty = args.difficulty
    MarbleContext.marble_name = args.marble
    if args.control is not None:
        MarbleContext.control = args.control
    if args.camera_distance is not None:
        MarbleContext.camera_distance = args.camera_distance
    if args.lean is not None:
        # Stand in for the player's hands: the frames run with this held, so a
        # picture can show the board leaning rather than only sitting level.
        held = tuple(args.lean)
        MarbleContext._lean_demand = lambda self, held=held: held
    if args.size is not None:
        MarbleContext.ContextMainLoop(size=args.size)
    else:
        MarbleContext.ContextMainLoop()


if __name__ == "__main__":
    main()
