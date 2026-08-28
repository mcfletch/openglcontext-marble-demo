"""The registries that let a dozen fragments be built at once.

The point of these is a merge property rather than a runtime one: a fragment is
one file that registers itself, so two fragments never touch the same file and
never conflict, and nothing has to be kept in step by hand.

What a test can hold that to is that a module dropped into the directory is
found without anything else being edited, and that the things found declare
enough about themselves to be chosen between.
"""
import pytest

from openglcontext_marble_demo import fragments, mechanisms, pieces

# -- discovery ------------------------------------------------------------------

def test_the_library_finds_what_is_in_its_directory():
    """Discovered rather than listed: a file everybody has to edit is a file
    everybody conflicts in."""
    assert fragments.library()


def test_every_fragment_answers_to_its_own_name():
    for name, entry in fragments.library().items():
        assert entry.name == name


def test_a_fragment_carries_the_tags_a_generator_chooses_by():
    for entry in fragments.library().values():
        assert entry.tags, entry.name
        assert all(isinstance(tag, str) for tag in entry.tags)


def test_a_fragment_says_what_it_asks_of_a_player():
    for entry in fragments.library().values():
        assert isinstance(entry.rule, str), entry.name


def test_asking_for_a_fragment_that_is_not_there_says_so():
    with pytest.raises(KeyError, match='nonesuch'):
        fragments.build('nonesuch', rng=None, entry=None)


# -- variants -------------------------------------------------------------------

def test_every_fragment_offers_at_least_one_variant():
    """A fragment with no variants reads the same way every time it appears."""
    for entry in fragments.library().values():
        assert entry.variants, entry.name


def test_a_variant_can_be_asked_for_by_name():
    import random
    for entry in fragments.library().values():
        for variant in entry.variants:
            piece = fragments.build(entry.name, random.Random(1), _port(),
                                    variant=variant)
            assert piece.cells, '%s/%s' % (entry.name, variant)


def test_asking_for_a_variant_that_is_not_there_says_so():
    import random
    name = sorted(fragments.library())[0]
    with pytest.raises(KeyError, match='nonesuch'):
        fragments.build(name, random.Random(1), _port(), variant='nonesuch')


def test_variants_of_one_fragment_differ_from_each_other():
    """Two variants that build the same thing are one variant."""
    import random
    for entry in fragments.library().values():
        if len(entry.variants) < 2:
            continue
        built = [_signature(fragments.build(entry.name, random.Random(1), _port(),
                                            variant=variant))
                 for variant in entry.variants]
        assert len(set(built)) == len(built), entry.name


# -- what every fragment has to be ----------------------------------------------

def test_every_fragment_builds_a_piece_with_an_ok_exit():
    import random
    for entry in fragments.library().values():
        piece = fragments.build(entry.name, random.Random(3), _port())
        assert 'ok' in piece.exits, entry.name
        assert piece.exits['ok'].cell in piece.cells, entry.name


def test_every_fragment_can_be_crossed_from_its_entry_to_its_ok_exit():
    import random
    for entry in fragments.library().values():
        piece = fragments.build(entry.name, random.Random(3), _port())
        assert pieces.joined(piece.cells, piece.entry.cell,
                             piece.exits['ok'].cell), entry.name


def test_every_fragment_holds_the_slope_budget():
    import random
    for entry in fragments.library().values():
        piece = fragments.build(entry.name, random.Random(3), _port())
        for (col, row), height in piece.cells.items():
            for dcol, drow in ((1, 0), (0, 1)):
                beside = (col + dcol, row + drow)
                if beside in piece.cells:
                    assert abs(piece.cells[beside] - height) <= pieces.MAX_STEP + 1e-9, \
                        entry.name


def test_a_fragment_is_the_same_fragment_for_the_same_seed():
    import random
    for entry in fragments.library().values():
        first = fragments.build(entry.name, random.Random(5), _port())
        again = fragments.build(entry.name, random.Random(5), _port())
        assert first.cells == again.cells, entry.name


# -- mechanisms -----------------------------------------------------------------

def test_the_mechanism_registry_finds_what_is_in_its_directory():
    assert isinstance(mechanisms.registry(), dict)


def test_every_mechanism_is_a_feature_the_file_format_can_write():
    from openglcontext_marble_demo import levelfile
    for name, factory in mechanisms.registry().items():
        assert levelfile.FEATURES.get(name) is factory, name


def test_a_mechanism_registers_under_a_name_of_its_own():
    seen = list(mechanisms.registry())
    assert len(seen) == len(set(seen))


# -- helpers --------------------------------------------------------------------

def _port():
    return pieces.Port(cell=(0, 0), facing=(0, 1), height=0.0, width=3)


def _signature(piece):
    """What makes one built piece different from another.

    The theme counts: a variant that differs only in what it is made of is a
    real variant, because the material is the grip and an icy room is a
    different room to drive in.
    """
    return (piece.theme,
            tuple(sorted(piece.cells.items())),
            tuple(sorted(repr(f) for f in piece.features)))
