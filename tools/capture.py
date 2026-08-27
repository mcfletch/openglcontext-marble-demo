#!/usr/bin/env python
"""Render a generated level headlessly and save a screenshot.

A committed capture pipeline (used for README screenshots and quick visual checks),
not a throwaway script.  It drives the real demo context under OpenGLContext's
auto-exit capture harness, so what it saves is exactly what the game renders.

    python tools/capture.py --seed 7 --difficulty 3 --out shot.png
    python tools/capture.py --marble chrome --frames 90 --out chrome.png
"""
import argparse
import os


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--difficulty", type=int, default=3)
    parser.add_argument("--marble", default="steel")
    parser.add_argument("--frames", type=int, default=60,
                        help="frames to simulate before capturing")
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
    MarbleContext.ContextMainLoop()


if __name__ == "__main__":
    main()
