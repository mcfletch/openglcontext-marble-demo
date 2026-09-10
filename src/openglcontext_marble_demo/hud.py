"""The heads-up display: countdown timer, speed, material, and win/lose banner.

The HUD is split into a *content* half (pure functions that format the game state
into strings — unit tested) and a *drawing* half (:class:`HUD`, which paints those
strings over the finished frame through the engine's ``renderShaderOverlay`` hook
and shader text renderer).  Keeping the content pure means the interesting logic is
tested without a window.
"""
import math
from typing import Any

from OpenGLContext.scenegraph.text.shadertext import get_text_renderer

from .game import LOST, PLAYING, WON

URGENT_TIME = 4.0            # seconds remaining below which the clock turns red


def hud_lines(game: Any) -> Any:
    """The lines drawn in the top-right corner.

    The clock, how fast the marble is going, what it is made of, how many times
    it has gone over the edge — and how far the board is leaning, with the
    control model driving it, so what the controls are doing is on the screen
    rather than inferred from how the marble moves.
    """
    lean = math.degrees(math.hypot(game.tilt.pitch, game.tilt.roll))
    lines = [
        f"TIME {game.time_left:4.1f}",
        f"SPEED {game.controller.speed:4.1f}",
        f"MARBLE {game.marble_material}",
        f"FALLS {game.controller.fall_count}",
        f"LEAN {lean:4.1f} {game.control}",
    ]
    # Only where a board has them: a read-out that always says zero is one
    # nobody reads.
    if game.gates_left:
        lines.append(f"GATES {game.gates_left}")
    # How many marbles this run has cost and what took the last one, named, so
    # the trap that keeps winning can be recognised and steered around.
    if game.controller.last_loss:
        lines.append(f"LOST {game.controller.loss_count} {game.controller.last_loss}")
    return lines


def time_is_urgent(game: Any) -> Any:
    return game.state == PLAYING and game.time_left <= URGENT_TIME


def banner(game: Any) -> Any:
    """A big center message: the end of the run, or the marble just lost.

    A trap the player cannot name is a trap they cannot learn, so a destroyed
    marble puts what took it — and what it cost the clock — in the middle of the
    screen for the seconds before its replacement arrives.
    """
    if game.state == WON:
        return "FINISH!  press N for next"
    if game.state == LOST:
        if game.ended_by:
            return f"OUT OF TIME — {game.ended_by}  press R to retry"
        return "OUT OF TIME  press R to retry"
    if game.controller.is_lost:
        return f"{game.controller.last_loss.upper()}  -{game.loss_penalty:.0f}s"
    return None


class HUD:
    """Draws the HUD over the frame via the pass's shader text renderer.

    Instances are cheap; the text renderer is cached per size by the engine.  Call
    :meth:`render` from a context's ``renderShaderOverlay(flatpass)`` hook.
    """

    def __init__(self, game: Any, font_size: Any=20, banner_size: Any=34) -> None:
        self.game = game
        self._text = get_text_renderer(font_size)
        self._banner = get_text_renderer(banner_size)
        self.margin = 14

    def render(self, flatpass: Any, context: Any) -> None:
        width, height = context.getViewPort()
        if not width or not height:
            return
        shader = flatpass.shader_program
        self._draw_status(shader, width, height)
        self._draw_banner(shader, width, height)

    def _draw_status(self, shader: Any, width: Any, height: Any) -> None:
        lines = hud_lines(self.game)
        line_h = self._text.char_height * 1.2
        # Top-right, stacked downward, inset by the width of the longest line
        # rather than a fixed figure: the lines are monospace, so this is exact,
        # and adding a line no longer risks it running off the edge.
        longest = max((len(line) for line in lines), default=0)
        x = max(self.margin, width - self.margin - longest * self._text.char_width)
        y = height - self.margin - self._text.char_height
        clock_color = (1.0, 0.3, 0.3, 1.0) if time_is_urgent(self.game) else (0.9, 1.0, 0.9, 1.0)
        for i, line in enumerate(lines):
            color = clock_color if i == 0 else (0.85, 0.9, 1.0, 1.0)
            self._text.render_text(line, x=x, y=int(y - i * line_h),
                                   shader_program=shader,
                                   viewport_width=width, viewport_height=height,
                                   color=color, background_color=(0.05, 0.06, 0.08, 0.6))

    def _draw_banner(self, shader: Any, width: Any, height: Any) -> None:
        message = banner(self.game)
        if message is None:
            return
        x = max(10, width // 2 - len(message) * self._banner.char_width // 2)
        y = height // 2
        self._banner.render_text(message, x=int(x), y=int(y), shader_program=shader,
                                 viewport_width=width, viewport_height=height,
                                 color=(1.0, 0.95, 0.4, 1.0),
                                 background_color=(0.05, 0.06, 0.08, 0.8))
