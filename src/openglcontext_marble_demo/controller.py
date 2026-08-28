"""The marble controller: spin-to-steer, grounding, and fall/respawn.

This is the heart of the game feel.  It owns no rendering and no window — it reads
and writes a :class:`~omi_physics.world.PhysicsWorld` body and a
:class:`~openglcontext_marble_demo.track.TrackMap`, so every rule here is unit
tested against the real physics.

Two responsibilities:

**Steering.**  Holding an arrow imparts *spin*, not a shove.  A ball rolling in a
horizontal direction ``d`` spins about the axis ``up × d`` (rolling without
slipping), so :meth:`steer` applies an angular impulse about that axis plus a
small linear assist.  How much of that spin becomes motion is left to the physics
— the pairwise marble↔surface friction (see :mod:`materials`) does the rest, which
is why a rubber marble bites and a steel marble on ice slips.

**Track presence.**  Each :meth:`update` classifies the marble as *grounded* (over
a track cell, near its surface — this becomes the checkpoint and camera target),
*jumping* (airborne but still above the track plane — left alone), or *fallen*
(out over the void and dropped below the plane, or past the kill plane).  A fall
freezes the camera on the last square and, after a penalty delay, respawns the
marble there at rest.  The delay is deliberate: an instant respawn would let a
player dump unwanted speed by driving off the edge.
"""
import numpy as np

ACTIVE = "active"
FALLEN = "fallen"

# Steering is expressed in the camera's fixed frame.  The board tilts toward +Z, so
# the marble rolls +Z downhill *toward the viewer* (the camera sits on the +Z side,
# looking back up the slope).  Screen-up is therefore -Z (up the slope, away from the
# viewer) and screen-right is +X: so +forward (the up arrow) is -Z and +right is +X.
FORWARD_AXIS = np.array([0.0, 0.0, -1.0])
RIGHT_AXIS = np.array([1.0, 0.0, 0.0])
UP = np.array([0.0, 1.0, 0.0])


class MarbleController:
    #: How far above the surface the marble has to have been for the contact
    #: that follows to be a landing rather than a roll.  Comfortably more than
    #: a terrace step, which a marble crossing at speed hops off every time.
    FALL_HEIGHT = 1.5
    #: Frames after leaving the air during which a floor contact is that landing.
    LANDING_GRACE = 10

    def __init__(self, world, index, track, marble_radius=0.5, marble_material="steel",
                 camera=None, kill_y=-8.0, respawn_delay=2.0, kick_speed=1.2,
                 linear_fraction=0.6, steer_forward=FORWARD_AXIS, steer_right=RIGHT_AXIS):
        self.world = world
        self.index = index
        self.track = track
        self.radius = marble_radius
        self.marble_material = marble_material
        self.camera = camera
        self.kill_y = kill_y
        self.respawn_delay = respawn_delay
        # Steering frame (ground-plane axes for "forward"/"right").  Set from the
        # camera so steering is screen-relative even when the view is yawed for the
        # isometric angle; defaults to the world -Z / +X axes.
        self.forward_axis = np.asarray(steer_forward, dtype='d')
        self.right_axis = np.asarray(steer_right, dtype='d')
        # Each arrow "click" steers with target speed ``kick_speed`` (m/s).  Most of
        # it is a spin kick (angular momentum, the classic feel), with a fraction
        # ``linear_fraction`` applied as *direct* lateral velocity for immediate,
        # friction-independent authority.  The linear part is deliberately small: it
        # gives responsive per-square steering against the board tilt, but is far
        # below the marble's momentum, so one click cannot reverse a moving marble
        # (that still takes sustained held steering).
        self.kick_speed = kick_speed
        self.linear_fraction = linear_fraction

        # How close (marble bottom to cell surface) still counts as grounded, and
        # how far below the launch surface a void crossing must drop to be a fall.
        self.ground_tolerance = 0.35
        self.fall_margin = 1.0

        # Impact speed-kill tuning.  ``*_impact_speed`` are the contact impulses
        # (per unit mass, i.e. an effective approach speed) above which a hit
        # counts as a crash; ``*_retain`` is the fraction of horizontal speed kept
        # afterward.  A surface whose restitution is at least ``elastic_restitution``
        # is exempt — its bounciness returns the energy through the physics instead.
        self.wall_impact_speed = 3.0
        self.hard_landing_speed = 5.0
        self.wall_retain = 0.1
        self.landing_retain = 0.2
        self.elastic_restitution = 0.5

        self.state = ACTIVE
        self.respawn_timer = 0.0
        self.fall_count = 0
        # Whether the marble has *fallen*, which is what the hard-landing rule
        # keys on.  Not how hard the floor pushed back: a marble rolling down a
        # slope onto a flatter part pushes back exactly as hard as one that fell
        # there, and scrubbing it makes a dip impossible to carry speed through,
        # which is the whole of what a kicker asks of a player.
        #
        # What is remembered is *how far it fell*, which is what the rule has
        # always said in words: landing hard from a height costs you.  Two
        # things had to be got right to measure that.  A marble crossing a
        # terrace at speed hops off every lip, so being off the ground is not
        # enough -- it has to have been high.  And one falling fast crosses the
        # grounded tolerance a frame or two before it touches, so the height has
        # to be remembered for a moment after it is back down.
        self._peak_clearance = 0.0
        self._fell_from = 0.0
        self._since_air = self.LANDING_GRACE + 1
        start_cell = track.cell_of(world.position[index][0], world.position[index][2])
        self.checkpoint = start_cell
        self._checkpoint_surface = track.cells.get(start_cell, 0.0)

    # -- steering -------------------------------------------------------
    def kick(self, forward, right):
        """Impart one steering kick in the camera-relative direction.

        ``forward``/``right`` are in [-1, 1].  The kick is mostly **spin** (angular
        momentum about ``up × travel`` — the classic Marble-Madness feel, and what
        the marble↔surface grip converts to motion over time) plus a small *direct*
        lateral velocity (``linear_fraction``) for immediate authority against the
        board tilt.  The linear part is small relative to the marble's momentum, so
        one click steers but cannot reverse a moving marble — sustained held steering
        does.  One call is one "click"; key-repeat re-fires it.
        """
        raw = forward * self.forward_axis + right * self.right_axis
        magnitude = np.linalg.norm(raw)
        if magnitude < 1e-9:
            return
        travel = raw / magnitude
        speed = self.kick_speed * min(magnitude, 1.0)
        mass = self.world.mass[self.index]
        radius = self.radius
        spin_axis = np.cross(UP, travel)
        # Spin: angular impulse L = I·Δω with Δω = speed / radius, I = ⅖·m·r² (sphere).
        self.world.apply_angular_impulse(
            self.index, spin_axis * (0.4 * mass * radius * speed))
        # A small immediate lateral nudge so steering bites within a square.
        self.world.apply_impulse(
            self.index, travel * (mass * speed * self.linear_fraction))

    # -- per-frame update ----------------------------------------------
    def update(self, dt):
        """Advance the track-presence state machine one frame; drive the camera."""
        if self.state == FALLEN:
            self._update_respawn(dt)
            return self.state

        position = self.world.position[self.index]
        cell = self.track.cell_of(position[0], position[2])
        on_track = cell in self.track.cells

        grounded = on_track and self._is_grounded(position, self.track.cells[cell])
        if grounded:
            self.checkpoint = cell
            self._checkpoint_surface = self.track.cells[cell]
        surface = self.track.cells[cell] if on_track else self._checkpoint_surface
        self._watch_the_air(grounded, position[1] - self.radius - surface)

        if self.camera is not None:
            self.camera.target(position)

        self._apply_impact_rules(airborne=self.airborne)

        if self._has_fallen(position, on_track):
            self._begin_fall()
        return self.state

    def _watch_the_air(self, grounded, clearance):
        """Remember how far above the surface the marble has been, and when.

        The height is latched on the way *down* -- the frame the marble is back
        on the ground -- and left alone after that, so a second grounded frame
        does not wipe what the first one recorded.
        """
        if not grounded:
            self._peak_clearance = max(self._peak_clearance, float(clearance))
            self._since_air = 0
            return
        if self._peak_clearance:
            self._fell_from = self._peak_clearance
            self._peak_clearance = 0.0
        self._since_air += 1

    @property
    def airborne(self):
        """Whether the marble fell far enough, recently enough, that a floor
        contact now is a landing rather than a roll."""
        return (self._since_air <= self.LANDING_GRACE
                and max(self._fell_from, self._peak_clearance) >= self.FALL_HEIGHT)

    # -- impact speed kills ---------------------------------------------
    def _apply_impact_rules(self, airborne=True):
        """Scrub speed on a hard wall hit or landing (non-elastic surfaces only).

        Reads this step's contacts: a large normal impulse per unit mass is an
        effective approach speed, and the contact normal's verticality tells a
        floor landing from a wall.  Resting contacts carry only the tiny
        weight-support impulse, well below the thresholds, so a marble simply
        sitting or rolling is never affected.

        ``airborne`` says whether the marble had left the ground.  A landing is
        only a landing if it fell: rolling fast down a slope onto a flatter part
        pushes the floor exactly as hard, and scrubbing that makes a dip
        impossible to carry speed through.  A wall is a wall either way.
        """
        world, i = self.world, self.index
        mass = max(world.mass[i], 1e-6)
        for contact in world.contacts:
            if i not in (contact.a, contact.b):
                continue
            other = contact.b if contact.a == i else contact.a
            impact_speed = contact.normal_impulse / mass
            if impact_speed < self.wall_impact_speed:
                continue
            if world.material_for(world.collider_material[other]).restitution \
                    >= self.elastic_restitution:
                continue                         # springy surface: let physics keep it
            if abs(contact.normal[1]) > 0.7:     # floor/ceiling contact
                if airborne and impact_speed >= self.hard_landing_speed:
                    self._scale_horizontal_speed(self.landing_retain)
            else:                                # wall contact
                self._scale_horizontal_speed(self.wall_retain)

    def _scale_horizontal_speed(self, retain):
        # Scrub the horizontal velocity *and* the spin — otherwise the spin the
        # marble built up re-accelerates it a frame later and the crash wouldn't
        # bite (a real crash kills both).
        v = self.world.linear_velocity[self.index]
        v[0] *= retain
        v[2] *= retain
        self.world.angular_velocity[self.index] *= retain

    def _is_grounded(self, position, surface):
        return (position[1] - self.radius) <= surface + self.ground_tolerance

    def _has_fallen(self, position, on_track):
        if position[1] < self.kill_y:
            return True
        # Over the void and dropped below the plane it launched from → a fall, not
        # a jump.  A jump arc stays at or above the plane until it lands.
        return (not on_track
                and position[1] < self._checkpoint_surface - self.fall_margin)

    # -- fall / respawn -------------------------------------------------
    def _begin_fall(self):
        self.state = FALLEN
        self.respawn_timer = 0.0
        if self.camera is not None:
            center_x, center_z = self.track.cell_center(*self.checkpoint)
            self.camera.target((center_x, self._checkpoint_surface, center_z))
            self.camera.hold()

    def _update_respawn(self, dt):
        self.respawn_timer += dt
        if self.respawn_timer >= self.respawn_delay:
            self._respawn()

    def _respawn(self):
        center_x, center_z = self.track.cell_center(*self.checkpoint)
        rest_y = self._checkpoint_surface + self.radius + 0.05
        self.world.position[self.index] = (center_x, rest_y, center_z)
        self.world.linear_velocity[self.index] = (0.0, 0.0, 0.0)
        self.world.angular_velocity[self.index] = (0.0, 0.0, 0.0)
        self.world.wake(self.index)
        self.state = ACTIVE
        self.respawn_timer = 0.0
        self.fall_count += 1
        if self.camera is not None:
            self.camera.release()

    # -- convenience ----------------------------------------------------
    @property
    def is_respawning(self):
        return self.state == FALLEN

    @property
    def speed(self):
        """Horizontal speed of the marble (for the HUD)."""
        v = self.world.linear_velocity[self.index]
        return float(np.hypot(v[0], v[2]))
