"""Stories: chapters joined by which exit leads where.

A chain is a list, and a list cannot say what happens when a player misses the
turn-off. A story is a **graph**: each chapter says where its ``ok`` exit leads
and, where it has one, where its failure exit leads -- so getting it wrong drops
you somewhere slower rather than ending the run, and the slower way rejoins.

What a test can hold a story to is that the board it builds is one connected
region, that both ways through actually exist, and that the branch does not land
on top of the main line.
"""
import pytest

from openglcontext_marble_demo import pieces, stories


def _linear():
    return stories.Story(name='linear', start='a', chapters={
        'a': stories.Chapter(id='a', fragment='plateau', exits={'ok': 'b'}),
        'b': stories.Chapter(id='b', fragment='ramp_down', exits={'ok': 'c'}),
        'c': stories.Chapter(id='c', fragment='plateau', exits={}),
    })


def _branching():
    """A hairpin whose failure exit goes the long way round and rejoins."""
    return stories.Story(name='branching', start='a', chapters={
        'a': stories.Chapter(id='a', fragment='plateau', exits={'ok': 'turn'}),
        'turn': stories.Chapter(id='turn', fragment='hairpin',
                                exits={'ok': 'quick', 'missed': 'long'}),
        'quick': stories.Chapter(id='quick', fragment='plateau',
                                 exits={'ok': 'end'}),
        'long': stories.Chapter(id='long', fragment='scatter',
                                exits={'ok': 'end'}),
        'end': stories.Chapter(id='end', fragment='plateau', exits={}),
    })


# -- what a story is ------------------------------------------------------------

def test_a_story_names_its_first_chapter():
    assert _linear().start == 'a'


def test_a_chapter_says_where_each_of_its_exits_leads():
    assert _branching().chapters['turn'].exits == {'ok': 'quick', 'missed': 'long'}


def test_a_story_whose_start_is_not_a_chapter_is_refused():
    with pytest.raises(ValueError, match='nowhere'):
        stories.Story(name='broken', start='nowhere', chapters={}).build(1)


def test_a_chapter_leading_to_a_chapter_that_is_not_there_is_refused():
    story = stories.Story(name='broken', start='a', chapters={
        'a': stories.Chapter(id='a', fragment='plateau', exits={'ok': 'missing'})})
    with pytest.raises(ValueError, match='missing'):
        story.build(1)


def test_a_chapter_naming_a_fragment_that_is_not_there_is_refused():
    story = stories.Story(name='broken', start='a', chapters={
        'a': stories.Chapter(id='a', fragment='trampoline', exits={})})
    with pytest.raises(KeyError, match='trampoline'):
        story.build(1)


# -- a linear story is still a story --------------------------------------------

def test_a_linear_story_builds_a_connected_board():
    built = _linear().build(3)
    assert pieces.joined(built.cells, built.start, built.finish)


def test_a_linear_story_lays_every_chapter_down():
    built = _linear().build(3)
    assert set(built.placed) == {'a', 'b', 'c'}


def test_the_finish_is_the_last_chapter_with_nowhere_left_to_go():
    built = _linear().build(3)
    assert built.finish in built.placed['c'].cells


# -- the branch -----------------------------------------------------------------

def test_a_branching_story_lays_both_ways_down():
    built = _branching().build(3)
    assert set(built.placed) == {'a', 'turn', 'quick', 'long', 'end'}


def test_both_ways_reach_the_end():
    """The long way is passable and slow, not a dead end: a player who gets it
    wrong is behind, not finished."""
    built = _branching().build(3)
    for through in ('quick', 'long'):
        assert pieces.joined(built.cells, built.placed[through].entry.cell,
                             built.finish), through


def test_the_branch_does_not_land_on_top_of_the_main_line():
    built = _branching().build(3)
    quick = set(built.placed['quick'].cells)
    long_way = set(built.placed['long'].cells)
    assert not (quick & long_way)


def test_the_long_way_is_longer_than_the_quick_one():
    built = _branching().build(3)
    quick = pieces.distance(built.cells, built.placed['quick'].entry.cell,
                            built.finish)
    slow = pieces.distance(built.cells, built.placed['long'].entry.cell,
                           built.finish)
    assert slow > quick


def test_a_story_is_the_same_board_for_the_same_seed():
    first, again = _branching().build(5), _branching().build(5)
    assert first.cells == again.cells


def test_different_seeds_give_different_boards():
    assert _branching().build(1).cells != _branching().build(2).cells


# -- what comes out -------------------------------------------------------------

def test_a_story_becomes_a_level_the_game_can_play():
    from openglcontext_marble_demo.game import PLAYING, MarbleGame
    level = _branching().build(3).level()
    game = MarbleGame(level)
    for _ in range(240):
        game.advance(1 / 120.0)
    assert game.state in (PLAYING, 'won', 'lost')


def test_a_story_holds_the_slope_budget_across_its_joins():
    built = _branching().build(3)
    for (col, row), height in built.cells.items():
        for dcol, drow in ((1, 0), (0, 1)):
            beside = (col + dcol, row + drow)
            if beside in built.cells:
                assert abs(built.cells[beside] - height) <= pieces.MAX_STEP + 1e-9


def test_a_chapter_can_ask_for_a_variant_and_a_theme():
    story = stories.Story(name='themed', start='a', chapters={
        'a': stories.Chapter(id='a', fragment='plateau', variant='rink',
                             exits={'ok': 'b'}),
        'b': stories.Chapter(id='b', fragment='plateau', theme='foundry',
                             exits={}),
    })
    built = story.build(3)
    assert built.placed['a'].theme == 'ice'
    assert built.placed['b'].theme == 'foundry'


def test_the_rules_of_a_story_are_the_rules_of_its_chapters():
    """What a story asks of a player, in the order it asks it."""
    assert _branching().build(3).rules()
