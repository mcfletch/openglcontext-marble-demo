"""End-to-end render smoke test: the demo actually draws a non-blank frame.

Every other test is headless game logic; this one drives the *real* window under
OpenGLContext's auto-exit capture harness (in a subprocess, so its GL context can't
pollute the rest of the suite) and asserts the captured frame has real content — the
level rendered, not just a cleared background.

Skipped automatically where no GL target is available (so it never fails a headless
CI that lacks a display); on this dev container's GLFW backend it runs.
"""
import os
import subprocess
import sys

import pytest

pytest.importorskip("glfw")
pytest.importorskip("PIL")
import numpy as np
from PIL import Image


def _has_gl():
    return bool(os.environ.get("WAYLAND_DISPLAY") or os.environ.get("DISPLAY"))


@pytest.mark.skipif(not _has_gl(), reason="no GL display available")
def test_demo_renders_a_non_blank_frame(tmp_path):
    env = dict(os.environ)
    env.update(
        OPENGLCONTEXT_PROFILE="core",
        OPENGLCONTEXT_RENDERER="pbr", OPENGLCONTEXT_IBL="full",
        OPENGLCONTEXT_DISABLE_FPS_DISPLAY="1",
        OPENGLCONTEXT_AUTO_EXIT_FRAMES="40",
        OPENGLCONTEXT_AUTO_EXIT_CAPTURE_DIR=str(tmp_path),
        OPENGLCONTEXT_AUTO_EXIT_CAPTURE_NAME="smoke",
    )
    result = subprocess.run(
        [sys.executable, "-m", "openglcontext_marble_demo",
         "--seed", "3", "--difficulty", "1"],
        env=env, capture_output=True, text=True, timeout=180)

    shot = tmp_path / "smoke.png"
    assert shot.exists(), f"no frame captured; stderr:\n{result.stderr[-2000:]}"

    pixels = np.asarray(Image.open(shot).convert("RGB"), dtype=float)
    # A blank frame is a single flat colour; a rendered level has real variance
    # (tiles, marble, HUD) — assert the frame is not near-uniform.
    assert pixels.std() > 8.0, "frame looks blank (nothing rendered)"
    # And it is not all one channel / all black.
    assert pixels.mean() > 5.0
