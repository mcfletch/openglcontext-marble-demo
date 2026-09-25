"""The marble controller: spin-to-steer, grounding, fall/respawn, and destruction.

This is the heart of the game feel.  It owns no rendering and no window — it reads
and writes a :class:`~omi_physics.world.PhysicsWorld` body and a
:class:`~openglcontext_marble_demo.track.TrackMap`, so every rule here is unit
tested against the real physics.

Three responsibilities:

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

**Destruction.**  A marble can also be *lost*, which is what gives the board's
traps teeth: hit hard enough (:attr:`~MarbleController.lethal_impact_speed`),
dropped far enough onto the board (:attr:`~MarbleController.lethal_fall_height`),
held between two surfaces closer together than it is wide
(:attr:`~MarbleController.crush_squeeze`), or destroyed by a hazard that calls
:meth:`destroy` itself.  Falling off the board is the cheap ending; being lost
costs the longer :attr:`~MarbleController.destroy_delay` and, through
:class:`~openglcontext_marble_demo.game.MarbleGame`, a chunk of the clock.  What
it was lost to is kept in :attr:`~MarbleController.last_loss` so the player can
be told which trap took the marble.

A hazard on the board — a burner, anything spread over cells rather than met at a
contact — destroys a marble by calling :meth:`destroy` with its own cause.  The
game finds those hazards among the build's animators: anything answering
``lost(body)`` with a cause or ``None`` is asked once a frame.
"""
from typing import Any, NamedTuple

import numpy as np

ACTIVE = "active"
FALLEN = "fallen"
DESTROYED = "destroyed"

#: What a marble can be lost to.  These are the words the player is shown: a
#: trap nobody can name is a trap nobody learns to avoid.
STRUCK = "struck"
DROPPED = "dropped"
CRUSHED = "crushed"
BURNED = "burned"

#: How far past a right angle two of the marble's contact normals have to point
#: before it counts as being *between* those surfaces rather than tucked into the
#: corner where they meet.  A right-angled corner reads 0; two flat faces facing
#: each other read -1.
OPPOSED = -0.5

# Steering is expressed in the camera's fixed frame.  The board tilts toward +Z, so
# the marble rolls +Z downhill *toward the viewer* (the camera sits on the +Z side,
# looking back up the slope).  Screen-up is therefore -Z (up the slope, away from the
# viewer) and screen-right is +X: so +forward (the up arrow) is -Z and +right is +X.
FORWARD_AXIS = np.array([0.0, 0.0, -1.0])
RIGHT_AXIS = np.array([1.0, 0.0, 0.0])
UP = np.array([0.0, 1.0, 0.0])


class Touch(NamedTuple):
    """Something the marble touched this frame, read from the marble's side.

    The world's normal runs from body ``a`` toward body ``b``, so which way it
    points depends on which of the pair the marble happens to be.  Flipped
    once here, every rule below can read it as *the way out of the marble*.

    A frame is several physics steps.  ``approach`` and ``impulse`` are the
    hardest of the frame's steps, since a blow is measured on the step it
    lands and on no other; ``normal`` and ``depth`` are the latest.
    """
    #: The body on the other side of the contact.
    other: int
    #: Unit normal pointing out of the marble, toward what it is touching.
    normal: np.ndarray
    #: How fast the two were closing along that normal when they met, in m/s.
    #: The solver records this before it resolves anything, which is the only
    #: moment it can be read.
    approach: float
    #: The impulse the solver applied, per unit of the marble's mass.  A marble
    #: is a sphere, so it meets anything at one point and this is that point's.
    impulse: float
    #: How far the two overlap, in metres.
    depth: float


class MarbleController:
    #: How far above the surface the marble has to have been for the contact
    #: that follows to be a landing rather than a roll.  Comfortably more than
    #: a terrace step, which a marble crossing at speed hops off every time.
    FALL_HEIGHT = 1.5
    #: Frames after leaving the air during which a floor contact is that landing.
    LANDING_GRACE = 10

    def __init__(self, world: Any, index: Any, track: Any, marble_radius: Any=0.5, marble_material: Any="steel",
                 camera: Any=None, kill_y: Any=-8.0, respawn_delay: Any=2.0, destroy_delay: Any=None,
                 on_lost: Any=None, kick_speed: Any=1.2,
                 linear_fraction: Any=0.6, steer_forward: Any=FORWARD_AXIS, steer_right: Any=RIGHT_AXIS) -> None:
        self.world = world
        self.index = index
        self.track = track
        self.radius = marble_radius
        self.marble_material = marble_material
        self.camera = camera
        self.kill_y = kill_y
        self.respawn_delay = respawn_delay
        # A lost marble is remade at the checkpoint like a fallen one, but the
        # wait is twice as long by default: the difference between a slip and a
        # death has to be felt before the clock comes into it at all.
        self.destroy_delay = (2.0 * respawn_delay if destroy_delay is None
                              else destroy_delay)
        #: Called with the cause the moment a marble is lost, for whoever is
        #: charging for it -- the game docks the clock and names it on screen.
        self.on_lost = on_lost
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

        # Destruction thresholds.  Each is set well clear of what ordinary play
        # produces, because a rule that fires on a good run is a rule the player
        # reads as the game breaking rather than as a trap they walked into.
        #
        # ``lethal_impact_speed`` is a *closing* speed in m/s -- how fast the two
        # were coming together when they met, which the solver records before it
        # resolves anything.  The impulse it then applies is (1 + restitution)
        # times as large, so a springy bumper books half again what stone books
        # for the same hit; measuring the closing speed is what stops the softest
        # thing on the board being the deadliest.  A marble free-rolling down a
        # leaned board reaches a wall at about 12 m/s and a boost pad caps at 8.
        self.lethal_impact_speed = 20.0
        # How far a marble may fall *onto* the board and roll on.  Terraces step
        # 0.9 m and an elevator travels 3 m, so this is a tower a board has to be
        # built to have.  Falling off the *edge* is a different thing and stays
        # cheap: there is nothing out there to hit.
        self.lethal_fall_height = 8.0
        # Held between two opposed surfaces closer together than the marble is
        # wide by this fraction of its diameter, for ``crush_time`` seconds.  A
        # fraction rather than a distance, so it means the same for any marble.
        self.crush_squeeze = 0.1
        self.crush_time = 0.25

        self.state = ACTIVE
        self.respawn_timer = 0.0
        #: What the marble has touched since the last update, by the other body.
        self._touched: dict[int, Touch] = {}
        #: What the last update read, and the step it was read after.
        self._last_touches: list[Touch] = []
        self._read_at = -1
        self._listen(world)
        self.fall_count = 0
        #: How many marbles this run has lost, and what the last one was lost
        #: to -- one of :data:`STRUCK`, :data:`DROPPED`, :data:`CRUSHED`,
        #: :data:`BURNED`, or whatever cause a hazard passed to :meth:`destroy`.
        self.loss_count = 0
        self.last_loss = None
        # Seconds the marble has been held between two surfaces, so far.
        self._squeezed_for = 0.0
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
        #: The height of the last ground the marble stood on, whether or not it
        #: was somewhere worth coming back to.  Kept apart from the checkpoint's
        #: height because the two answer different questions: this one is *how
        #: far below the board am I*, and the checkpoint's is *where do I come
        #: back to*.  Sharing one number meant that holding the checkpoint back
        #: at the top of a descent also held the falling-off test up there, and
        #: a marble rolling normally down a piece was counted as having fallen.
        self._ground_surface = self._checkpoint_surface

    # -- where a fallen marble comes back ---------------------------------
    #: The four cells a checkpoint has to have floor in.
    _AROUND = ((1, 0), (-1, 0), (0, 1), (0, -1))

    def _somewhere_to_come_back_to(self, cell: Any) -> Any:
        """Is ``cell`` somewhere a marble put down at rest will still be a moment later?

        Floor on every side, or a rail where the floor runs out.  A marble is
        respawned **at rest**, and the board's own lean starts it moving again
        immediately -- so a checkpoint at the lip of a drop is one the lean
        carries straight back over the same edge, and the player watches the same
        two seconds of penalty over and over with nothing they can do about it.
        A cell that is held on every side is the margin that gives them somewhere
        to steer from.

        Measured on a lane with a bite out of one side: a marble shoved into the
        gap fell seven times, six of them from the same lip cell, and every one
        of the seven checkpoints had void beside it.  Held instead, the same
        shove costs one fall.
        """
        cells = self.track.cells
        return all((cell[0] + dcol, cell[1] + drow) in cells
                   or self._railed(cell, (dcol, drow))
                   for dcol, drow in self._AROUND)

    def _railed(self, cell: Any, step: Any) -> Any:
        """Is there a wall on the ``step`` side of ``cell``?

        A plank with rails is as good to come back to as open floor: what a
        checkpoint needs is that the marble cannot leave by that side, and a rail
        is the board saying so.
        """
        railed = getattr(self.track, 'railed', None)
        return bool(railed and railed(cell, step))

    # -- steering -------------------------------------------------------
    def kick(self, forward: Any, right: Any) -> None:
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
    def update(self, dt: float) -> Any:
        """Advance the track-presence state machine one frame; drive the camera."""
        if self.state != ACTIVE:
            self._update_respawn(dt)
            return self.state

        position = self.world.position[self.index]
        cell = self.track.cell_of(position[0], position[2])
        on_track = cell in self.track.cells

        grounded = on_track and self._is_grounded(position, self.track.cells[cell])
        if grounded and self._somewhere_to_come_back_to(cell):
            self.checkpoint = cell
            self._checkpoint_surface = self.track.cells[cell]
        if grounded:
            self._ground_surface = self.track.cells[cell]
        surface = self.track.cells[cell] if on_track else self._ground_surface
        landed_from = self._watch_the_air(grounded, position[1] - self.radius - surface)

        if self.camera is not None:
            self.camera.target(position)

        # Everything the contacts have to say, read once.  The order is the
        # order the player would name it in: how far it fell, then how hard it
        # was hit, then whether something is closing on it.
        touches = self._touches()
        if landed_from >= self.lethal_fall_height:
            return self.destroy(DROPPED)
        if max((touch.approach for touch in touches), default=0.0) \
                >= self.lethal_impact_speed:
            return self.destroy(STRUCK)
        if self._held_too_long(touches, dt):
            return self.destroy(CRUSHED)

        self._apply_impact_rules(touches, airborne=self.airborne)

        if self._has_fallen(position, on_track):
            self._begin_fall()
        return self.state

    def _listen(self, world: Any) -> None:
        """Have the world report every step of the marble's contacts to :meth:`_touch`.

        Only the marble's pairs are recorded, and each of them on every step
        it lasts: a squeeze is a pair of contacts held, not a pair that began.
        """
        world.report_contacts(self.index)
        if world.contact_reporting == 'off':
            world.contact_reporting = 'flagged'
        world.report_persist = True
        world.add_contact_listener(self._touch)

    def _touch(self, event: Any) -> None:
        """Keep one step's contact on the marble, the hardest blow of the frame kept.

        Only while the marble is in play: a lost marble is still in the world
        and can still be struck, and nothing reads those blows.
        """
        i = self.index
        if (self.state != ACTIVE or event.phase == 'end'
                or i not in (event.a.index, event.b.index)):
            return
        mine = event.a.index == i
        other = event.b.index if mine else event.a.index
        approach = event.approach
        impulse = event.impulse / max(float(self.world.mass[i]), 1e-6)
        earlier = self._touched.get(other)
        if earlier is not None:
            approach = max(approach, earlier.approach)
            impulse = max(impulse, earlier.impulse)
        self._touched[other] = Touch(other, event.normal if mine else -event.normal,
                                     approach, impulse, event.depth)

    def _touches(self) -> Any:
        """What the marble touched over the steps since the last update.

        An update with no step since the one before sees what that one saw:
        nothing has moved, so the marble is held exactly as it was.
        """
        if self.world.step_count == self._read_at:
            return self._last_touches
        self._read_at = self.world.step_count
        self._last_touches = list(self._touched.values())
        self._touched.clear()
        return self._last_touches

    def _watch_the_air(self, grounded: Any, clearance: Any) -> Any:
        """Remember how far above the surface the marble has been, and when.

        The height is latched on the way *down* -- the frame the marble is back
        on the ground -- and left alone after that, so a second grounded frame
        does not wipe what the first one recorded.

        Returns that height on the one frame the marble lands, and zero on every
        other frame, which is what tells the drop that destroys a marble from
        the same drop still being remembered a moment later.
        """
        if not grounded:
            self._peak_clearance = max(self._peak_clearance, float(clearance))
            self._since_air = 0
            return 0.0
        self._since_air += 1
        if not self._peak_clearance:
            return 0.0
        self._fell_from = self._peak_clearance
        self._peak_clearance = 0.0
        return self._fell_from

    @property
    def airborne(self) -> Any:
        """Whether the marble fell far enough, recently enough, that a floor
        contact now is a landing rather than a roll."""
        return (self._since_air <= self.LANDING_GRACE
                and max(self._fell_from, self._peak_clearance) >= self.FALL_HEIGHT)

    # -- impact speed kills ---------------------------------------------
    def _apply_impact_rules(self, touches: Any, airborne: Any=True) -> None:
        """Scrub speed on a hard wall hit or landing (non-elastic surfaces only).

        A large normal impulse per unit mass is an effective approach speed, and
        the contact normal's verticality tells a floor landing from a wall.
        Resting contacts carry only the tiny weight-support impulse, well below
        the thresholds, so a marble simply sitting or rolling is never affected.

        ``airborne`` says whether the marble had left the ground.  A landing is
        only a landing if it fell: rolling fast down a slope onto a flatter part
        pushes the floor exactly as hard, and scrubbing that makes a dip
        impossible to carry speed through.  A wall is a wall either way.
        """
        world = self.world
        for touch in touches:
            if touch.impulse < self.wall_impact_speed:
                continue
            if world.material_for(world.collider_material[touch.other]).restitution \
                    >= self.elastic_restitution:
                continue                         # springy surface: let physics keep it
            if abs(touch.normal[1]) > 0.7:       # floor/ceiling contact
                if airborne and touch.impulse >= self.hard_landing_speed:
                    self._scale_horizontal_speed(self.landing_retain)
            else:                                # wall contact
                self._scale_horizontal_speed(self.wall_retain)

    # -- being crushed ---------------------------------------------------
    def _held_too_long(self, touches: Any, dt: float) -> Any:
        """Whether the marble has been squeezed past bearing, for long enough.

        The press has to be *held*.  A gap that shuts on the marble and opens
        again is a scare rather than a death, which is what ``crush_time`` buys
        the player -- and what keeps a marble bouncing off a ceiling on its way
        through a tunnel out of it.
        """
        if self._squeeze(touches) < self.crush_squeeze * 2.0 * self.radius:
            self._squeezed_for = 0.0
            return False
        self._squeezed_for += dt
        return self._squeezed_for >= self.crush_time

    @staticmethod
    def _squeeze(touches: Any) -> Any:
        """How far below its diameter the marble is being held, in metres.

        Two contacts whose outward normals oppose one another are the two sides
        of a grip, and the depths they overlap by sum to exactly how far the pair
        of surfaces has closed past the marble's width.
        """
        return max((first.depth + second.depth
                    for n, first in enumerate(touches) for second in touches[n + 1:]
                    if float(np.dot(first.normal, second.normal)) <= OPPOSED),
                   default=0.0)

    def _scale_horizontal_speed(self, retain: Any) -> None:
        # Scrub the horizontal velocity *and* the spin — otherwise the spin the
        # marble built up re-accelerates it a frame later and the crash wouldn't
        # bite (a real crash kills both).
        v = self.world.linear_velocity[self.index]
        v[0] *= retain
        v[2] *= retain
        self.world.angular_velocity[self.index] *= retain

    def _is_grounded(self, position: Any, surface: Any) -> Any:
        return (position[1] - self.radius) <= surface + self.ground_tolerance

    def _has_fallen(self, position: Any, on_track: Any) -> Any:
        if position[1] < self.kill_y:
            return True
        # Over the void and dropped below the plane it launched from → a fall, not
        # a jump.  A jump arc stays at or above the plane until it lands.
        return (not on_track
                and position[1] < self._ground_surface - self.fall_margin)

    # -- fall / destruction / respawn -----------------------------------
    def destroy(self, cause: Any) -> Any:
        """Lose the marble to ``cause``; a new one arrives at the last checkpoint.

        Public because the board reaches in through it: three of the four ways to
        lose a marble are measured here from the contacts, and a hazard spread
        over cells — a burner, anything with a dwell rather than an impact —
        calls this with its own cause instead.

        Losing an already-lost marble is nothing, so a second cause arriving in
        the same moment cannot overwrite the one the player is being shown.
        """
        if self.state == DESTROYED:
            return self.state
        self.state = DESTROYED
        self.respawn_timer = 0.0
        self._squeezed_for = 0.0
        self.last_loss = cause
        self.loss_count += 1
        self._hold_camera()
        if self.on_lost is not None:
            self.on_lost(cause)
        return self.state

    def forget_the_run(self) -> None:
        """Clear what the marble has been through, for a run started over."""
        self.fall_count = 0
        self.loss_count = 0
        self.last_loss = None

    def _begin_fall(self) -> None:
        self.state = FALLEN
        self.respawn_timer = 0.0
        self._hold_camera()

    def _hold_camera(self) -> None:
        """Freeze the view on the square the marble was last safely on."""
        if self.camera is None:
            return
        center_x, center_z = self.track.cell_center(*self.checkpoint)
        self.camera.target((center_x, self._checkpoint_surface, center_z))
        self.camera.hold()

    @property
    def wait_to_return(self) -> Any:
        """Seconds before the marble comes back, which is longer for a loss."""
        return self.destroy_delay if self.state == DESTROYED else self.respawn_delay

    def _update_respawn(self, dt: float) -> None:
        self.respawn_timer += dt
        if self.respawn_timer >= self.wait_to_return:
            self._respawn()

    def _respawn(self) -> None:
        was_lost = self.state == DESTROYED
        center_x, center_z = self.track.cell_center(*self.checkpoint)
        self._ground_surface = self._checkpoint_surface
        rest_y = self._checkpoint_surface + self.radius + 0.05
        self.world.position[self.index] = (center_x, rest_y, center_z)
        self.world.linear_velocity[self.index] = (0.0, 0.0, 0.0)
        self.world.angular_velocity[self.index] = (0.0, 0.0, 0.0)
        self.world.wake(self.index)
        # A marble set down at a checkpoint has not fallen there.  Without this
        # the drop it just took is still on the books, and the first frame back
        # on the ground reads as a landing from the bottom of the void.
        self._peak_clearance = 0.0
        self._fell_from = 0.0
        self._since_air = self.LANDING_GRACE + 1
        self._squeezed_for = 0.0
        # Nor has it been struck there: whatever was recorded before it was
        # lost belongs to the marble that was.
        self._touched.clear()
        self._last_touches = []
        self._read_at = self.world.step_count
        self.state = ACTIVE
        self.respawn_timer = 0.0
        if not was_lost:
            self.fall_count += 1
        if self.camera is not None:
            self.camera.release()

    # -- convenience ----------------------------------------------------
    @property
    def is_respawning(self) -> Any:
        return self.state in (FALLEN, DESTROYED)

    @property
    def is_lost(self) -> Any:
        """Whether a destroyed marble is still waiting to be replaced."""
        return self.state == DESTROYED

    @property
    def speed(self) -> Any:
        """Horizontal speed of the marble (for the HUD)."""
        v = self.world.linear_velocity[self.index]
        return float(np.hypot(v[0], v[2]))
