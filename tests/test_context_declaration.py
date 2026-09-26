"""What the game's window is: GLFW, and no navigation beside the follow camera."""
from OpenGLContext.context import Context

from openglcontext_marble_demo.run import MarbleContext


def test_it_is_a_context():
    assert issubclass(MarbleContext, Context)


def test_it_opens_a_glfw_window():
    assert MarbleContext.windowSystemName == 'glfw'


def test_the_follow_camera_is_all_that_moves_the_view():
    """No movement mode, and no free-fly camera to fight the follow camera."""
    assert not MarbleContext.resolveDefinition().navigation


def test_a_window_size_given_at_start_keeps_the_declaration():
    resolved = MarbleContext.resolveDefinition(size=(640, 480))
    assert tuple(resolved.size) == (640, 480)
    assert not resolved.navigation
