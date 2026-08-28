# Marble: a stand-alone game

A proposal, for review. Two pieces of work that want doing together: the game
grows the screens that make it something you can start with no arguments and
play, and the driving model changes so that a run is fast, readable, and has
more than one way through it.

Nothing here is built yet. The numbers quoted are measured from the game as it
stands today, headless, at `seed=7 difficulty=3` and `seed=1 difficulty=2`.

---

## 1. Where the pace and the control go

### The board is a one-cell corridor

`generator._carve_path` random-walks a path that never crosses itself, and every
cell of the level *is* that path. A generated board is a snake one cell wide:

```
seed 7, difficulty 3 -- 25 cells, 63.8 s limit, 5 columns x 14 rows
 .  .  .  S  .
 .  .  .  #  X       S start   F finish   # plain tile
 .  .  ^  #  #       ^ ramp    o bumper   s spring
 .  .  #  .  .       X rotating arm       | wall
 .  .  ^  .  .
 .  #  ^  .  .
 .  #  o  #  .
 .  .  .  #  #
 .  .  .  .  ^
 .  .  .  .  #
 .  .  .  s  #
 .  .  ^  #  .
 .  .  #  .  .
 .  F  #  ^  .
```

Three consequences, and they are the three complaints:

- **There is exactly one route.** Not one good route among several — one route.
- **Every obstacle is across the only lane.** A rotating arm on a one-wide path
  is a gate, not a hazard to steer around; the choice it offers is "time it or
  wait".
- **A third of the cells are transition ramps** (6 of 25 here, 9 of 20 on seed 1)
  because the walk steps height on most cells, so the run is mostly slope
  furniture rather than track.

### Forward is slow and sideways is a teleport

Measured on the physics as it is, free-rolling with no input:

| | |
|---|---|
| speed after 6 s | 2.1 m/s |
| distance in 6 s | 8.6 m = **2.1 cells** |
| effective pace | **0.35 cells/second** |

At that rate the 25-cell board above is 70 s of rolling, and the 63.8 s limit is
roughly "keep going and don't fall". Now the same measurement with the right
arrow held (`run.py` sets `keyRepeatDelay = 0.1`, so ten kicks a second):

| marble | lateral travel, 2 s held |
|---|---|
| steel | 12.5 m |
| rubber | 12.5 m |
| ice | 12.6 m |

**6 m/s sideways against 1 m/s forward, in a lane 4 m wide.** A tap is a 0.72 m/s
velocity jolt; a hold is a stutter of ten of them a second, and the amount it
moves you depends on the platform's key-repeat rate. There is no small
correction available — the smallest input the game accepts already crosses a
sixth of the lane, and a held one leaves the board.

That is the whole of "easy to see what to do, hard to accomplish". The player is
asked to thread a 4 m corridor with a control that only knows one size of nudge
and delivers it at a rate nobody chose.

The identical figures across steel, rubber and ice are the second half of it:
`linear_fraction = 0.6` of each kick is applied as direct velocity, which the
friction table never sees, and it swamps the spin component entirely. The marble
materials change how the ball looks and nothing about how it drives.

### Falling and hitting stop the game dead

- A fall freezes the camera for `respawn_delay = 2.0` s before returning control.
- A wall contact above `wall_impact_speed = 3.0` scales horizontal velocity *and*
  spin to `wall_retain = 0.1` — a glancing touch of a rail at speed leaves the
  marble at a tenth of its speed with its spin gone.

Both were reasoned as costs. On a board where reaching 2 m/s takes six seconds,
they are the majority of the run.

### You cannot see the board

`CAMERA_DISTANCE = 24.0` with a 30° field of view over 4 m cells puts about two
cells on screen. Planning a line needs seeing the line.

![The view at seed 7: about two cells of a 25-cell board.](images/current-view-seed7.png)

---

## 2. What is proposed

### 2.1 Tilt the board — the control model

The player's input becomes a **board tilt**, and gravity follows it. Held keys
(or a stick) ramp a tilt vector toward the held direction and spring it back to
neutral when released; each frame writes `world.gravity.direction` from a fixed
**base tilt toward the finish** plus the **player tilt** on top. The marble
always rolls forward on its own — the thing the original is built on — and
steering is leaning the whole board, which is the control every player already
has an intuition for.

Gravity is re-read from `world.gravity` on every step
(`omi_physics.backend.NumpyBackend.integrate_forces` → `resolve_gravity`), so
this costs one vector write per frame and no engine change at all.

The scene root gets that same tilt as a Transform, so the board visibly leans.
The lean is the feedback that tells you how hard you are pushing, and it costs
one node.

Prototyped headlessly against the real physics, base tilt 12°, from a standstill:

| input | result after 1 s |
|---|---|
| 15° right | 0.72 m lateral (0.18 cells) |
| 25° right | 1.13 m lateral (0.28 cells) |
| 35° right | 1.45 m lateral (0.36 cells) |

The pace with no input is unchanged by this, and the first draft of this document
said otherwise: the board's existing lean **already is** 12.4° (`atan(0.22)`), so
leaning it is a change to *steering*, not to speed. What the pace actually costs
is measured in §9 — it is the physics manager's damping defaults, and they are
worth more than the angle.

What this buys, in order of how much it matters:

- **Every input size is available.** A 30 ms tap leans the board a little; a hold
  leans it fully. Acceleration, not teleportation, so the response is
  proportional to how long you asked for it and small corrections exist.
- **Control stops depending on the key-repeat rate.** The tilt integrates over
  real time.
- **Authority no longer depends on grip.** Gravity acts whatever the marble is
  made of, so a player never loses the ability to steer. Friction goes back to
  being flavour — how much the ball rolls versus slides, how long it takes to
  answer — which is where the material table wants to be. In the prototype ice
  drifts 18% further than steel on the same input: felt, not disabling.
- **It scales to an analog stick** the day someone plugs one in, with no second
  control scheme.

**Tuning knobs, all data:** base tilt (pace), maximum player tilt (authority),
tilt ramp-in and spring-back rates (how sharp the board feels). Every one of them
is a number in a dataclass a test can drive.

### 2.2 Hop — one button, and most of the second solution

**Space hops.** A short upward impulse, available when grounded, on a ~0.6 s
cooldown. It is the cheapest possible source of alternative routes: on a board
that has any shape at all, a hop clears a rail, skips the outside of a corner,
crosses a gap, or reaches a terrace one step up. Every board gets an expert line
for free, and the generator does not have to author a single one of them.

It is also the answer to a blocked lane that is not "wait for the arm".

### 2.3 Braided boards, two to three cells wide

The generator changes from carving *the* path to carving a **region**:

1. Walk a spine start → finish as now.
2. **Widen** it, so the corridor is 2–3 cells across most of the way and opens
   into 4–5 cell plazas at intervals.
3. **Braid** it: add 1–3 alternate strands that leave the spine and rejoin it —
   loops, which is what makes a maze offer a choice rather than a solution.
4. Decorate *strands*, not cells: one strand of a braid gets the hazards and is
   short, the other is clear and long. That is a real decision, made at speed,
   about what you are good at.
5. Keep the connectivity and determinism tests. They get stronger, not weaker:
   with braids there is a shortest route and a safest route, and both are
   assertable.

This is what fixes obstacles-across-the-only-passage without removing obstacles.
A bumper in a 3-wide lane is something to dodge or bounce off deliberately. A
rotating arm across a plaza is a hazard with room to read it.

**Boards get shorter as they get wider** — 12–18 cells of spine — because the
interesting length is the number of decisions, not the number of tiles.

### 2.4 Pace

- **Base tilt 12–15°** instead of the current 0.22 gravity-Z component, for
  0.8–1.0 cells/second free-rolling.
- **Time limit ~1.1 s per spine cell + a small constant**, in place of today's
  2.4 s per cell. A run becomes 25–40 s: short enough that Retry is instant and
  cheap, which is what makes a fast game fast.
- **Falls cost clock, not control.** Respawn immediately at the checkpoint,
  charge 3 s on the timer, flash it on the HUD. The cost is legible and the game
  never stops.
- **Rails deflect instead of arresting.** `wall_retain` 0.1 → ~0.6, and the
  speed-kill reserved for a genuinely head-on hit — scale the penalty by how much
  of the approach was into the wall, so a graze along a rail keeps your line and
  a full-speed nose-in still costs you.
- **Camera back to ~45–55 units** and easing outward with speed, so you see the
  board and see further when you are moving. Speed-scaled distance belongs in
  `OpenGLContext.move.followcam` as an engine feature — every following camera in
  a game wants it, and glisteel's would use it too.

---

## 3. Other mechanics, with a recommendation on each

**Recommended, in the first pass**

- **Boost pads / speed strips.** Directional strips laid along a strand: choose
  the fast line and you are committed to it. Uses the ramp effect already
  written.
- **Time crystals.** Pickups worth 2–3 s, placed on the *longer* braid strand.
  This is what makes the choice between strands a real one rather than a
  preference — the safe route buys back its own cost. It gives the timer a second
  job besides threatening you.
- **A finish that is a ramp.** End the board by launching into the goal rather
  than rolling onto a pad. Costs nothing and ends a run on the high note.

**Recommended, second pass**

- **Marble as a choice with consequences.** Once friction drives feel rather than
  authority, pick the marble at the start screen and mean it: rubber grips and
  hops well, ice is fast and loose, glass is light and gets flung further by
  springs. Right now the choice is cosmetic.
- **Ghost of your best run.** A translucent marble replaying the best time on
  this board. Well-matched to a short-run, best-time game, and it makes a second
  attempt worth making. Needs a recorded position track — small, and reusable.
- **Combo/flow scoring.** Time without touching a wall or falling builds a
  multiplier shown on the HUD. Rewards the clean line without punishing the messy
  one.

**Worth discussing, not recommended yet**

- **Flippers.** Fun, and genuinely off-genre: a flipper is furniture the marble
  must be delivered *to*, which is a pinball problem, and the game we are keeping
  is a navigation problem. If they come in, they come as a *board feature* — a
  pair of paddles at a wide junction, fired with a context key when the marble is
  in the zone, that fling you up to an otherwise unreachable strand. That is a
  route, which is what we want, rather than a control scheme, which we already
  have.
- **Sticky/magnet marble.** Hold a key to grip harder (higher friction) at the
  cost of speed. Cheap to build on the pairwise-friction override and it is a
  real risk/reward dial. It may be one verb too many alongside tilt and hop.
- **Moving-platform bridges.** Elevators exist; using them as the *only* way
  across a gap on one strand makes the braid choice a timing choice. Reads well;
  wants the braid work done first.

**Against**

- **A second controllable object, or any weapon.** Out of genre and it does not
  answer any of the complaints.
- **Procedural boards as the only source.** Not a mechanic, but see §4: a game
  you can hand someone needs boards with names.

---

## 4. The stand-alone shape

Follow `glisteel` exactly, because it has already answered these questions and a
second answer would be worse and different.

```
marble/
  menu.py       main_menu / board_screen / finish_screen -- plain functions
                returning OpenGLContext.ui Panels, no GL, fully tested
  boards.py     the library: named, curated (seed, difficulty, rules) boards
                plus the player's directory
  records.py    best time per board, JSON under the player's directory
  session.py    a run, headless: board + physics + controller + clock + phase
  run.py        the window: overlays, input, camera, frames
```

- **Main menu** — Play, Boards, Settings, Quit; **Resume** first when a run is in
  progress, and Escape then closes the menu rather than throwing the run away.
- **Boards** — a `Carousel` of boards, each with its own picture and its own best
  time, exactly as glisteel shows tracks. Boards are **named curated seeds**, not
  numbers to type: `--seed 7` stays as a flag for people who want it, but
  "choose a different board" has to mean picking a thing with a name and a face.
  Board pictures come from `tools/capture.py`, which already renders a board
  headlessly.
- **Finish** — the time, where it came in this board's table, and: Retry, Another
  board, Quit.
- **Settings** — tilt sensitivity, invert, camera distance, marble. `KeyCapture`
  is in the widget set, so rebinding is available when it is wanted.
- **Escape opens the menu.** R still retries and N still skips, as accelerators.

Everything above `run.py` is headless and testable, which is the split the demo
already has and the reason its rules are tested today.

---

## 5. Where each piece lives

Per the workspace rule that features live in the engine:

**`OpenGLContext`**

- **Speed-scaled follow distance** on `move.followcam.FollowCamera`, with easing.
  Every chase camera wants it.
- **A tilt-input helper** — the held-keys-to-a-ramped-vector integrator is not
  marble-specific; it is the same object a flight demo or a boat would want. It
  takes a time step and an input state and returns a vector; no GL.
- Anything the braided generator needs that turns out to be general graph work.

**`marble`** — the board generator, the tilt-to-gravity mapping, hop, crystals,
scoring, the screens, the records.

Nothing proposed here needs a change to `omi_physics`.

---

## 6. Housekeeping this depends on

- **`marble-demo/` is not a git repository** and is untracked in the workspace.
  Per the workspace rule for new projects it wants: its own repo, ruff + mypy
  configured in `pyproject.toml`, GitHub CI running the suite, and registration as
  a submodule installed editable by `uv sync`. Worth doing before the rewrite, so
  the rewrite has history.
- **The demo had drifted from the engine.** `model`, `mathutil` and `kinematic`
  moved from `OpenGLContext.physics` to `omi_physics`; four import lines in the
  package and five in the tests now point at the new homes, and the suite is
  green at 77 passed. CI would have caught it the day it happened.
- **A rename.** If it stops being a demo and becomes a game, `marble-demo` and
  `openglcontext_marble_demo` want to be the game's name. It is still an
  advertisement for the engine either way.

---

## 7. Order of work

Each phase is playable at its end, so the feel can be judged before the next one
is built.

1. **Tilt control + camera.** Tilt integrator, gravity mapping, visible board
   lean, camera pulled back and speed-scaled. Nothing else changes. This alone is
   most of the "controls should help you" fix, and it is where the tuning
   conversation happens.
2. **Pace.** Base tilt, time budget, instant respawn with a clock charge,
   deflecting rails.
3. **Hop.**
4. **Braided boards.** Widen, braid, decorate by strand; boost strips and time
   crystals; connectivity and two-route tests.
5. **The screens.** `menu.py`, `boards.py`, `records.py`, the session split,
   board pictures, Escape.
6. **The rest**, by whatever the review of this document decides: marble
   consequences, ghost, flow scoring, flippers.

Phase 1 is the one that answers the question this document was written about, and
it is small. Worth building before agreeing the rest.

---

## 8. Open questions for the review

1. **Tilt versus keeping spin-to-steer.** The proposal replaces the kick model.
   The alternative that keeps it is to make each kick continuous — an acceleration
   while held rather than a jolt on repeat — and raise the spin share so friction
   matters. That is a smaller change and keeps the marble identity; it does not
   give the board lean, the analog path, or the same intuitiveness. Which?
2. **How wide.** 2–3 cells is a corridor you can dodge in; 4–6 is a plane with
   hazards on it, which is closer to the original and further from a maze. The
   brief says keep maze navigation, so the proposal takes the narrower reading.
3. **Curated boards, generated boards, or both.** The proposal is a named library
   of curated seeds with generation behind it. A "Random board" entry is one line
   if it is wanted alongside.
4. **Does the timer stay the failure condition?** A short run with a tight clock
   and a best-time table might want the clock to be the *score* and falling off
   to be the only failure. Time crystals push in that direction.
5. **Flippers**: in, out, or as the junction feature described in §3?

---

## 9. Phase 1, built: `spike/tilt-camera`

Phase 1 of §7 exists and is playable. Every number below is measured against the
real physics, headless; the suite is 127 tests, ruff and mypy clean.

```bash
oglc-marble                          # leaning the board, the camera back at 48 m
oglc-marble --control spin           # the model it replaces, to compare
oglc-marble --tilt 18 --lean 32      # the two angles, in degrees
oglc-marble --damping 0.3 1.5        # the physics manager's own damping, to compare
```

![The board level, and the same board leaning left; the camera at 48 m, and the
marble holding its place on screen while the world tips around it.](images/spike-level.png)

![](images/spike-leaning.png)

**C switches control model mid-run**, so the two are felt back to back rather
than argued about, and the HUD carries the board's lean and which model is
driving it.

### What it does

`tilt.TiltRig` is the whole control model, and nothing in it knows about a
marble, a window or a clock: held input in, a gravity direction and a board
rotation out. `MarbleGame` writes that direction onto `world.gravity` each frame
— the physics re-reads it every step, so no engine change was needed for the
control — and points one `Transform` at the board rotation, centred on the
marble, so the world leans **around the ball** and the ball holds its place on
screen. The camera needs to know nothing about any of it, because the point it
follows is the one point the rotation does not move.

Both models run on one gravity path: `SPIN` is the same rig with the player's
lean bounded at zero. That is the whole difference, which is why the switch is
instant and why neither model can drift away from the other.

Input is **sampled, not evented**: the engine's `InputState` already accumulates
every key transition, so a frame asks what is held. Holding two directions leans
the board diagonally, and losing the window drops the lean rather than leaving
it stuck on — both for free, and no held-key bookkeeping was written.

### Steering: what changed

Lateral travel holding *right* on a flat board, steel marble:

| held | as it was | the spike |
|---|---|---|
| 0.10 s | 0.07 m | 0.01 m |
| 0.25 s | 0.33 m | 0.06 m |
| 0.50 s | 1.10 m | 0.27 m |
| 1.00 s | 3.73 m | 1.10 m |
| 2.00 s | **11.93 m** | 3.87 m |

The old smallest input was a **0.72 m/s velocity jolt**, arriving ten a second
while a key was held; there was nothing smaller and nothing in between. The new
smallest is one frame of lean — 2.7° at 60 Hz, worth 0.46 m/s² of pull — and
every size above it exists. Two seconds of full lean now moves one lane rather
than three.

**The marble materials mean something now.** Two seconds of held right:

| | as it was | the spike |
|---|---|---|
| steel | 11.93 m | 3.87 m |
| rubber | 11.96 m | 3.93 m |
| ice | 12.17 m | 5.17 m |

Under the old model the three are within 2% of each other, because the linear
part of each kick bypassed the friction table and swamped the spin. Under the
lean, ice carries a third further than steel — a difference the player feels,
without any marble losing the ability to steer, since gravity acts whatever the
ball is made of.

### Three things the measurements turned up

**The pace was never the board's lean; it was the damping.** The scenegraph
physics manager defaults to 0.3 linear and 1.5 angular damping — right for
settling a scene of boxes, and a continuous brake on a ball whose whole job is to
roll, since rolling couples spin to travel. Free-rolling pace against those two
numbers alone:

| linear, angular | free-rolling | top speed |
|---|---|---|
| 0.3, 1.5 (the manager's) | 0.43 cells/s | 2.3 m/s |
| 0.05, 0.2 (the game's now) | 0.94 cells/s | 6.8 m/s |
| 0.0, 0.0 | 1.12 cells/s | 9.0 m/s |

The game names its own (`ROLL_DAMPING`), keeping enough that a marble at rest
settles. Board for board, that is most of the pace §2.4 asks for:

| board | as it was | the spike |
|---|---|---|
| seed 7 d3 | 0.34 cells/s | 0.57 |
| seed 1 d2 | 0.71 | 0.99 |
| seed 5 d2 | 0.71 | 1.02 |
| seed 3 d1 | 0.05 | 0.09 |
| seed 11 d2 | 0.05 | 0.09 |

**Two of those boards go nowhere, and it is not the controls.** The board's
constant lean points +Z, and the generated path leaves the start *sideways* on
seed 3 and seed 11 — so a player who touches nothing is pushed straight over the
edge and falls twice in ten seconds. The lean and the path disagree. Either the
base lean follows the carved route rather than a fixed axis, or the boards get
wide enough that the disagreement does not reach an edge; §2.3 wants the second
anyway, and the first is a few lines. This is the clearest single argument in the
document for doing the generator work.

**Raising the lean limit buys much less than it looks like it should.** Seconds
of full lean to cross one 4 m lane from rest: 2.38 s at 20°, 2.05 s at 26°,
1.67 s at 40°, 1.45 s at 50°. A rolling sphere accelerates at `(5/7)·g·sin θ`,
and `sin` flattens out; the honest lever on how sharply the game turns is the
damping above, not the angle. 26° is where the spike sits.

### Also landed

`FollowCamera` grew `pull_back` / `pull_back_speed` / `pull_back_rate` and an
`advance(dt, speed)` in the **engine** (`spike/followcam-pullback` on
OpenGLContext) — a chase camera that eases back as its target speeds up, seeing
further exactly when there is least time to react. The easing is exponential in
elapsed time rather than in frames, so the same run frames the same way on a fast
machine and a slow one. `pull_back` defaults to zero, which is the fixed-offset
camera unchanged, so no existing caller is affected. The demo sits at 48 m with
0.30 of pull-back, against the 24 m that showed two cells.

### What is deliberately not in it

- **`TiltRig` is in the demo, not the engine.** The generic part is real — held
  input to a smoothed, clamped, frame-rate-independent axis pair — but its final
  shape depends on questions the spike is meant to answer: whether it needs a
  third axis, a deadzone for an analog stick, per-axis rates. Deciding its home
  after the shape settles is cheaper than guessing now; the move is small.
- **No hop, no braiding, no screens.** Phases 3–5.
- **The impact rules and the respawn freeze are untouched** (§2.4). They are
  pace, and they belong with the rest of the pace work, where they can be judged
  against a board that is worth crossing quickly.

### One bug worth recording

The first draft of `board_rotation` built the axis as `up × gradient`, which
draws the board tipping **up** into the direction the marble accelerates — the
world leaning the wrong way. The test that should have caught it asserted the
axis was parallel to the forward axis *up to a sign*, which is exactly the thing
that was wrong. Replacing it with an assertion about what the rotation *does* —
push the board's up-vector through the engine's own `transformMatrix` and check
which way it tips — catches it, and cannot be satisfied by the wrong convention.

---

## 10. Phase 4, built: braided boards

Section 2.3 exists. `boards.py` builds a **region** in four steps that each
answer one question — where the route goes, how much room there is, how many
ways round there are, and how far the marble has come — and hands back the
**strands** that decoration hangs on, because "the short way is the dangerous
one" is a statement about strands and cannot be said about loose cells.

`generator.py` decorates by strand: hazards go on the strands that *save cells*,
the long way round stays clear, and every hazard is checked against the board as
it is placed — one that would leave no clean route from start to finish is not
placed at all.

### What holds now, over twenty-four seeds each

| | |
|---|---|
| no board is a one-cell corridor | every cell has two ways off it |
| two routes share nothing but their ends | vertex-disjoint, every board |
| a braid is a different way round | not the next lane: 13 of 13 scenic strands cost ≥ 3 cells |
| a hazard-free route always exists | at difficulty 1, 3 and 5 |
| the quick line is the one that asks something | the clean route is never shorter |
| every route cell has board downhill of it | 0 exceptions in 72 boards |

That last row is the defect §9 turned up, closed. The old generator's constant
+Z lean pushed a player who touched nothing straight over the edge on two of
five sampled boards; there is now not one route cell on any board with nothing
in front of it.

### Two things that had to change to get there

**Room is a radius, not a width.** Widening only sideways leaves the places the
route goes *sideways* one cell tall — a corridor turned ninety degrees, with a
cell in it every route has to pass through. Three of the first six boards had
exactly that pinch. A radius has no direction to get wrong.

**Height is a function of rank alone.** It gives the terraced look, makes
downhill and forward the same direction, and holds the slope budget by
construction rather than by a check afterwards: neighbours are one rank apart at
most, so they are one terrace apart at most.

### Difficulty reversed

It shortens the route and thins the plazas rather than adding tiles — a long easy
board is only a long one. Hazards go from 0.9 a board to 2.1, and the clock lands
between 15 and 45 seconds, which is short enough that another go is cheap. The
old test asserting that a harder board had *more* cells is now a test asserting
the opposite, which is a deliberate change of contract rather than a loosened
assertion.

---

## 11. The editor

`marble-editor` is a sibling project — a game does not link an editor, exactly as
glisteel and glisteel-editor are separate.

```bash
oglc-marble-editor                    # a blank board to draw on
oglc-marble-editor spiral.marble      # carry on with one
oglc-marble-editor --from-seed 7      # start from a generated board
oglc-marble --board spiral.marble     # play one
```

![The editor: the board as a map, the tools down the left, the read-outs beside
them.](images/editor.png)

Six tools — tiles, height, surface, pieces, markers, pan — and two conventions
that run through all of them: **left does and right undoes**, and **a drag is one
step**. Ramps, launch ramps, rails, bumpers, springs, elevators and rotating arms
all place with a click, because the board works out which way a ramp points and
which side a rail faces.

### Where the parts are

`board.py` is the only thing that changes a level, so every rule about what a
board may be is a question a test can ask, and the tools are left holding nothing
but gestures. Undo is a **snapshot**: a board is small enough that keeping the
whole of one costs less than the bookkeeping for inverting each kind of change,
and a snapshot cannot get an inverse wrong.

Nothing stops a designer making a board that is half-drawn — that is a normal
thing to be looking at, and an editor that refused would be fighting them for the
whole middle of the work. The read-out says what would stop it being *played*
instead, most usefully that the finish cannot be reached from the start.

### What it is built on, and what it is not

The authoring toolkit in **OpenGLContext itself**: `edit` for the plan view and
the tool modes that give the tool in force first refusal of the pointer, `ui` for
the palette, the menus and the read-outs. The same foundation glisteel-editor
uses.

**`OpenGLContext-editor` is deliberately not a dependency.** That package is the
world-authoring half — DEM terrain, road alignment, 3D Tiles baking — and a board
is a grid of tiles. Depending on it would buy nothing and cost every editor user
a terrain toolchain.

The **board format belongs to the game** (`levelfile.py`), which is what keeps
the editor and the game from ever disagreeing about it. A mechanism writes itself
out of its own dataclass fields, and a test holds the registry to every authorable
class in `level.py`, so a mechanism added without a name there fails rather than
silently not saving.

### One thing the pointer bridge duplicates

`marble_editor/controls.py` turns pointer events into a
`~OpenGLContext.edit.tools.Pointer` and routes them, and glisteel-editor's
`MapControls` does the same job. The engine has the tool modes and the map view
but not the bridge between them. It is a candidate for `OpenGLContext.edit`, once
the second implementation has shown which parts of it are actually general — the
marble one wants no height function and no grab reach, and the glisteel one wants
both.

### A defect this turned up

A board carrying an **elevator or a rotating arm** logs one
`Failure in Box render: MissingVertexInput: the shader reads aColor, aJoints,
aTangent, aTexCoord1, aWeights` per run, and that one node does not draw. It
reproduces on generated boards (`--seed 0 --difficulty 4`) as readily as on
authored ones, so it predates this work. Giving the kinematic body a PBR
appearance like every other body in the game — which it should have had anyway,
and now has — did not cure it. The pass catches it, so the rest of the frame
renders. Worth chasing: it is the engine's instanced/kinematic draw path meeting
a plain `Box`, and nothing in the game can work around it.

---

## 12. What the game is missing, and what to build

Watching a recorded run says two things the tests could not.

### 12.1 The board does not feel like an object

It reads as a pinball flipper rather than a table: a huge surface thrown about
with no weight in it, and the world heaving around a ball that stays put.
Measured over one autopilot run (seed 1, difficulty 2, 771 frames):

| | |
|---|---|
| drawn lean, mean | **16°** |
| drawn lean, 90th percentile | **26°** |
| drawn lean, peak | **35°** |
| frames past 25° | **30%** |
| frames where the pilot's demand is at the stop | **34%** |
| how far a far corner of a 36 m board swings at peak | **20 m** |

Four separate causes, and they compound:

1. **The limit is per axis, so a diagonal exceeds it.** `pitch` and `roll` are
   each bounded at 26°, and the board is drawn at `atan(|gradient|)` — the two
   combined. A full diagonal draws **35°**, not 26°. The limit does not limit.
2. **The drawn lean is the whole lean.** `visual_gain` is 1.0, so every degree
   the physics uses is a degree the world turns through.
3. **There is no mass.** The lean moves at 160°/s: full deflection in 0.16 s.
   A table with a marble on it does not do that, and neither do hands.
4. **The pilot lives at the stop.** Its demand saturates on a third of frames,
   so what a viewer sees is mostly the extreme.

**What it should feel like.** A heavy table, tilted by hand: it takes an
appreciable fraction of a second to reach a lean, it eases in and out rather
than snapping, and the useful range is a handful of degrees rather than a
quarter turn. The *physics* lean and the *drawn* lean are already separate
knobs; the drawn one should be a hint of the real one, because a small visible
tilt over a large surface already reads as a large tilt.

**Requirements**

- The combined lean is bounded, not each axis separately.
- The board accelerates into a lean and eases out of it, rather than moving at a
  constant rate — second order, so there is something to feel.
- The drawn lean is a fraction of the physical one, and the fraction is a
  tunable with a default that keeps a far corner's swing under a cell or two.
- The pilot asks for a lean proportional to what it needs, and reaches the stop
  when it is in trouble rather than as a matter of course.
- The ball stays in frame: whatever the board does, the camera holds the marble.

### 12.2 There is one level, and it is a floor

Every board is the same board: a wide, gently terraced plane with a few things
scattered on it. There is nowhere you *must* reach, nothing to overcome, and no
reason to be anywhere in particular. Rolling across it is rolling across a
slightly bumpy floor.

What is wanted instead is **a series of stories**: a board that is a chain of
places, each with a thing to do, joined by transitions that are themselves the
challenge. Something like —

> On the first plateau you avoid the bumpers, because they give you too much
> energy and you will miss the turn-off; so you sneak round by bouncing off one
> wall, which lines you up for the ramp. On the ramp you have to backspin hard to
> get round the corner, which drops you into the bowl with enough speed to come
> out the far side and hit the lever that opens the door to the final room. Miss
> the turn-off and you are dropped onto the middle ramp — the forest of banking
> mushrooms — where you must hit a bumper hard enough to get through the passage
> of pinball bunkers that fling you about, and then over a narrow bridge to the
> ramp down.

Every noun in that is a piece, and every verb is a transition.

**Requirements**

**Pieces with ports.** A board is built from named pieces, each of which knows
where it is entered and where it is left — a cell, a facing and a height. A
piece places its own cells, walls and mechanisms relative to its entry, and
answers with its exit. Chaining them is what makes a board.

**Transitions are functions, and they are the challenge.** A joiner is a piece
whose whole purpose is getting from one plateau to the next, and what makes it
worth playing is the *rule* it imposes:

- **The kicker.** A dip you must enter fast enough to climb the far side. Too
  slow and you roll back down and have to try again.
- **The spillway.** A ramp with no wall at the bottom end: control the descent
  or overshoot into the void.
- **The hairpin.** A right-angle turn onto a short way round. Take it at speed
  and you miss it and are committed to the long way, which is passable and slow.
- **The narrow bridge.** A one-cell span with nothing either side.
- **The scatter.** A field of bumpers that flings you about: getting through is
  partly luck and entirely about arriving with the right speed.

**Waypoints.** A board has places you must reach before the finish will take
you, so a route is a route rather than a direction. A gate is a trigger like the
finish, and the finish refuses until every gate has been passed.

**Themes per area.** A piece names a theme, and a theme says what its floors and
walls are made of — which is a look *and* a feel, since the material is the grip
— and, later, what they sound like. A board that changes underfoot as you move
through it is a board you can navigate by.

**Stories compose pieces.** A story is a list of piece-builders, so the same
kicker used in three stories is the same function, and a new challenge is a new
function rather than a new generator. Written as functions, not generated by a
model: a model in the build path is a dependency, a cost and a source of boards
nobody can reproduce, and the vocabulary here is small enough to write down.

### 12.3 Where the experiments are

One branch each, so each can be judged on its own:

| Branch | What it tries |
|---|---|
| `spike/tilt-feel` | §12.1 — the board as something with mass |
| `spike/sections` | pieces with ports, and the joiners that are the challenges |
| `spike/waypoints` | gates the finish waits for |
| `spike/theming` | per-area materials, and where sound would hook in |
| `spike/stories` | chains of pieces, and the stories built from them |
