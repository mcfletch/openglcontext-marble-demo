"""Generating a story out of the library.

The editor can add a chapter at a time; this is the other half of the goal — a
whole board, assembled, that is worth playing. What makes one worth playing is
not randomness. A board that picked eight fragments out of a hat would be as
samey as one shape repeated, because the thing a player notices is *rhythm*: a
place to gather yourself, then something that asks a question, then somewhere to
land.

So the properties here are about shape rather than content, and they are what
separates an assembled board from a sprinkled one.
"""

from openglcontext_marble_demo import fragments, storygen


def _story(seed=1, **named):
    return storygen.compose(seed, **named)


# -- what comes out -------------------------------------------------------------

def test_a_composed_story_is_a_story_that_lays_out():
    built = _story().build(1)
    assert built.cells
    assert built.placed


def test_a_story_has_the_number_of_chapters_it_was_asked_for():
    for wanted in (4, 6, 10):
        assert len(_story(chapters=wanted).chapters) >= wanted


def test_a_composed_story_is_the_same_story_for_the_same_seed():
    assert _story(3).chapters.keys() == _story(3).chapters.keys()
    assert [c.fragment for c in _story(3).chapters.values()] == \
        [c.fragment for c in _story(3).chapters.values()]


def test_different_seeds_compose_different_stories():
    runs = {tuple(c.fragment for c in _story(seed).chapters.values())
            for seed in range(8)}
    assert len(runs) > 1


def test_every_chapter_names_something_the_library_holds():
    for chapter in _story().chapters.values():
        assert chapter.fragment in fragments.library()


def test_every_chapter_names_a_variant_its_fragment_offers():
    library = fragments.library()
    for chapter in _story().chapters.values():
        assert chapter.variant in library[chapter.fragment].variants


# -- rhythm ---------------------------------------------------------------------

def test_a_board_alternates_places_and_things_that_ask_something():
    """Somewhere to gather yourself, then a question, then somewhere to land.
    Two challenges back to back is where a board stops being readable."""
    for seed in range(6):
        kinds = _kinds(_story(seed, chapters=8))
        for before, after in zip(kinds, kinds[1:], strict=False):
            assert not (before == 'joiner' and after == 'joiner'), seed


def test_a_board_begins_somewhere_safe():
    """A challenge on the first chapter is one a player meets at a standstill."""
    library = fragments.library()
    for seed in range(6):
        story = _story(seed)
        assert not library[story.chapters[story.start].fragment].rule, seed


def test_a_board_ends_somewhere_safe():
    """The end is the chapter nothing leads out of, which on a board with a way
    round something is not the last one written down."""
    library = fragments.library()
    for seed in range(6):
        story = _story(seed)
        ends = [c for c in story.chapters.values() if not c.exits]
        assert ends, seed
        for chapter in ends:
            assert not library[chapter.fragment].rule, (seed, chapter.fragment)


def test_the_same_challenge_does_not_come_twice_in_a_row():
    for seed in range(6):
        asks = [c.fragment for c in _story(seed, chapters=10).chapters.values()]
        for before, after in zip(asks, asks[1:], strict=False):
            assert before != after or before == 'plateau', seed


def test_a_long_board_asks_more_than_one_kind_of_question():
    """Eight chapters of the same tag is one idea stretched out."""
    library = fragments.library()
    for seed in range(6):
        tags = set()
        for chapter in _story(seed, chapters=10).chapters.values():
            tags |= set(library[chapter.fragment].tags) - {'place'}
        assert len(tags) >= 2, seed


# -- difficulty -----------------------------------------------------------------

def test_a_board_gets_harder_as_it_goes():
    """What a player meets first should cost less than what they meet last."""
    library = fragments.library()
    for seed in range(6):
        costs = [library[c.fragment].cost
                 for c in _story(seed, chapters=10).chapters.values()]
        early = sum(costs[:len(costs) // 2])
        late = sum(costs[len(costs) // 2:])
        assert late >= early, seed


def test_asking_for_an_easier_board_gets_one():
    library = fragments.library()

    def total(difficulty):
        return sum(library[c.fragment].cost
                   for c in _story(2, chapters=10, difficulty=difficulty)
                   .chapters.values())
    assert total(1) < total(5)


# -- the way round ----------------------------------------------------------------

def test_a_board_offers_a_way_round_at_least_one_thing():
    """The point of the graph: somewhere a player who gets it wrong ends up,
    rather than a board that is one line with nothing off it."""
    found = 0
    for seed in range(8):
        story = _story(seed, chapters=10)
        if any(len(c.exits) > 1 for c in story.chapters.values()):
            found += 1
    assert found >= 4


def test_every_way_round_leads_somewhere_that_exists():
    for seed in range(8):
        story = _story(seed, chapters=10)
        for chapter in story.chapters.values():
            for target in chapter.exits.values():
                assert target in story.chapters


def test_a_composed_board_can_be_played():
    from openglcontext_marble_demo.game import PLAYING, MarbleGame
    level = _story(4, chapters=8).build(4).level()
    game = MarbleGame(level)
    for _ in range(300):
        game.advance(1 / 120.0)
    assert game.state in (PLAYING, 'won', 'lost')


def test_every_composed_board_lays_out_without_running_out_of_room():
    """A branch that will not fit is an error, so this is a real check."""
    for seed in range(12):
        built = _story(seed, chapters=9).build(seed)
        assert built.cells


# -- helpers ----------------------------------------------------------------------

def _kinds(story):
    """Whether each chapter of the main line is a place or asks a question.

    The main line rather than every chapter: a way round something is written
    into the story after the line it hangs off, so reading the mapping in order
    would put it at the end and say the board finishes on a challenge.
    """
    library = fragments.library()
    kinds = []
    at = story.start
    while at is not None:
        kinds.append('place' if not library[story.chapters[at].fragment].rule
                     else 'joiner')
        at = story.chapters[at].exits.get('ok')
    return kinds
