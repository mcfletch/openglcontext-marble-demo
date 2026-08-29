"""A magnet: a pull sideways, that bends the line without touching the marble.

Every other mechanism in the library acts on a marble that has arrived --
:class:`~openglcontext_marble_demo.mechanisms.sand.Sand` drags on it, a
:class:`~openglcontext_marble_demo.mechanisms.lever.Lever` waits to be struck,
a :class:`~openglcontext_marble_demo.mechanisms.burner.Burner` counts the time it
stands there.  A magnet acts on one that is still free, and what it changes is
where the marble is going.

**The pull acts along one axis, and never along the lane.** A pull towards the
post in both directions is a well: whatever falls in stops there, and the piece
becomes a trap rather than a corner.  So a magnet carries the ``axis`` its force
acts along, and pulls the marble only towards its own position on that axis --
across the lane, for a post standing beside one.  The marble keeps every bit of
the speed it arrived with and only its heading changes, which is what "bends the
line" has to mean if the line is still to arrive somewhere.

The pull is horizontal for the same reason.  A magnet that pulled downward as
well would press the marble into the floor, and what a player would feel is a
patch of drag they could not see the reason for.

**It falls off with distance, and it is capped.** The pull is
``strength`` metres per second squared at the post itself, easing to nothing at
:attr:`Magnet.radius`, so the field has an edge a player can learn rather than a
reach that goes on forever.  The cap is the same number: an inverse-square pull
is unbounded at the centre and would fling a marble that got close, which is a
different mechanism and a worse one.

**Repelling is the same mechanism with the sign turned round.** A negative
``strength`` pushes instead, which is a post to be threaded between rather than
one to lean away from.

:class:`MagnetPull` rides in ``BuildResult.animators``, the list the game updates
once per frame before it steps the world.  It finds bodies by looking rather than
by being registered, which is what lets it work on a marble that did not exist
when the level was built.

    >>> from openglcontext_marble_demo import mechanisms
    >>> mechanisms.registry()['magnet'] is Magnet
    True
"""
import math
from dataclasses import dataclass

import numpy as np

from .. import render
from . import mechanism

__all__ = ['MAGNET_COLOR', 'Magnet', 'MagnetPull']

#: Linear RGB for the post: dark, polished iron, so it reads as something with a
#: force about it rather than as another piece of the scenery.
MAGNET_COLOR = (0.16, 0.17, 0.20)

#: How tall the post is drawn and how wide, in metres.  Tall enough to be seen
#: from across the board, narrow enough that hitting one is a mistake rather than
#: an inevitability -- the pull is the mechanism, and the post is only where it
#: comes from.
POST = (0.5, 2.4)


class MagnetPull:
    """Pulls every dynamic body within reach of a set of posts towards them.

    Built with the world it acts on and the posts as ``[(x, z, strength,
    radius)]``.  :meth:`update` is called once per frame before the world steps.

    The whole set of posts is one animator rather than one each, because the
    work is a pass over the body arrays and doing it once for four posts is the
    same pass done once.
    """

    def __init__(self, world, posts):
        self.world = world
        self.posts = [(float(x), float(z), float(strength), float(radius),
                       (float(axis[0]), float(axis[1])))
                      for x, z, strength, radius, axis in posts]

    def pull_at(self, point):
        """The acceleration a body at world ``point`` feels, as ``(ax, az)``.

        A pure function of the posts and the point, so the field can be measured
        without stepping anything.
        """
        ax = az = 0.0
        for x, z, strength, radius, axis in self.posts:
            dx, dz = x - float(point[0]), z - float(point[2])
            distance = math.hypot(dx, dz)
            if distance >= radius:
                continue
            # How far off the post's own line the body is, measured along the
            # axis the post acts on.  Its sign is the direction the pull points,
            # and a negative strength turns that round into a push.
            offset = dx * axis[0] + dz * axis[1]
            if abs(offset) < 1e-6:
                continue
            # Linear falloff over the *round* field, so the post has an edge a
            # player can learn, and full strength on its own line.
            amount = math.copysign(strength * (1.0 - distance / radius), offset)
            ax += amount * axis[0]
            az += amount * axis[1]
        return ax, az

    def bodies_in_reach(self):
        """The indices of the dynamic bodies any post can act on.

        One vector pass over the body arrays rather than a loop over bodies: the
        animator runs every frame of every board that has a magnet on it, and a
        Python loop over every dynamic body is a cost that grows with the board
        rather than with the mechanism.
        """
        world = self.world
        position = world.position
        # ``inv_mass`` is non-zero for exactly the bodies an impulse moves.
        near = world.inv_mass > 0.0
        x, z = position[:, 0], position[:, 2]
        within = np.zeros(len(near), dtype=bool)
        for post_x, post_z, _strength, radius, _axis in self.posts:
            within |= ((x - post_x) ** 2 + (z - post_z) ** 2) < radius * radius
        return [int(index) for index in np.flatnonzero(near & within)]

    def update(self, dt=0.0):
        """Apply one frame of pull to everything in reach."""
        if not (dt and self.posts):
            return
        world = self.world
        for index in self.bodies_in_reach():
            ax, az = self.pull_at(world.position[index])
            if ax or az:
                mass = world.mass[index]
                world.apply_impulse(index, (ax * mass * dt, 0.0, az * mass * dt))


@mechanism('magnet')
@dataclass
class Magnet:
    """A post at ``cell`` that pulls a marble towards it as it passes.

    ``strength`` is the pull on the post's own line in metres per second squared,
    negative to push instead, and ``radius`` how far it reaches in metres.
    ``axis`` is the grid direction the pull acts along -- ``(1, 0)`` for a post
    that draws a marble across a lane running north, and never the direction the
    lane itself runs, which would make the post a brake.  ``height`` is where the
    post stands relative to the cell's surface, which is what keeps it out of the
    way of a marble arcing over.

    The post is drawn but takes no part in the physics: a marble that runs into
    one passes through it.  What a magnet is for is the line it bends, and a
    solid post would make it a wall with an interesting approach.
    """
    cell: tuple[int, int]
    strength: float = 9.0
    radius: float = 12.0
    axis: tuple[int, int] = (1, 0)
    height: float = 0.0

    def __post_init__(self):
        self.cell = (int(self.cell[0]), int(self.cell[1]))
        self.axis = (int(self.axis[0]), int(self.axis[1]))

    def owned_cells(self):
        """None: a magnet stands on the floor the piece around it laid."""
        return set()

    def build(self, scene, level, index, result):
        if self.cell not in level.cells:
            return
        x, z = level.cell_center(self.cell)
        base = level.cells[self.cell] + self.height
        width, tall = POST
        post = scene.add_box(size=(width, tall, width),
                             position=(x, base + tall / 2.0, z),
                             color=MAGNET_COLOR, dynamic=False,
                             material=index[level.surface])
        post.transform.children[0].appearance = render.color_appearance(
            MAGNET_COLOR, metallic=0.9, roughness=0.25)
        result.feature_bodies.append(post)
        result.animators.append(MagnetPull(
            scene.world, [(x, z, self.strength, self.radius, self.axis)]))
