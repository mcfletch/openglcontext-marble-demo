"""The seven pieces the library started with, offered as fragments with variants.

These predate the library and live in :mod:`~openglcontext_marble_demo.pieces`;
what is here is their registration and the variants that make each read
differently from one appearance to the next.  Fragments written from now on get
a module each, which is what keeps two of them from ever being in one another's
way -- these seven share this one because they already shared a file.

A variant is a bundle of the three axes: **material** (the theme, which is grip
as much as colour), **layout** (the shape), and **effect** (what it does to a
marble that gets it wrong).
"""
from openglcontext_marble_demo import pieces
from openglcontext_marble_demo.fragments import fragment

__all__ = ['plateau', 'ramp_down', 'kicker', 'spillway', 'hairpin', 'bridge',
           'scatter']

#: What each variant asks of the builder underneath it.
_PLATEAU = {
    'plain': {'theme': 'stone', 'across': 5, 'along': 5},
    'hall': {'theme': 'foundry', 'across': 7, 'along': 6},
    'ledge': {'theme': 'stone', 'across': 5, 'along': 3},
    'rink': {'theme': 'ice', 'across': 7, 'along': 5},
}
_RAMP = {
    'plain': {'theme': 'stone', 'drop': 2.7, 'length': 4},
    'long': {'theme': 'stone', 'drop': 3.6, 'length': 7},
    'steep': {'theme': 'foundry', 'drop': 4.5, 'length': 4},
    'slick': {'theme': 'ice', 'drop': 2.7, 'length': 5},
}
_KICKER = {
    'plain': {'theme': 'stone', 'depth': 2.7},
    'deep': {'theme': 'foundry', 'depth': 4.5},
    'shallow': {'theme': 'stone', 'depth': 1.8},
    'icy': {'theme': 'ice', 'depth': 2.7},
}
_SPILLWAY = {
    'plain': {'theme': 'stone', 'drop': 2.7},
    'long': {'theme': 'stone', 'drop': 4.5},
    'slick': {'theme': 'ice', 'drop': 3.6},
}
_HAIRPIN = {
    'plain': {'theme': 'stone'},
    'foundry': {'theme': 'foundry'},
    'slick': {'theme': 'ice'},
}
_BRIDGE = {
    'plain': {'theme': 'stone', 'length': 4},
    'long': {'theme': 'stone', 'length': 7},
    'slick': {'theme': 'ice', 'length': 4},
}
_SCATTER = {
    'plain': {'theme': 'rubber', 'length': 5},
    'wide': {'theme': 'rubber', 'length': 8},
    'foundry': {'theme': 'foundry', 'length': 6},
}


def _wrap(builder, table):
    """Turn one of the original builders into a fragment that takes a variant."""
    def build(rng, entry, variant='plain', **named):
        settings = dict(table[variant])
        settings.update(named)
        return builder(rng, entry, **settings)
    return build


plateau = fragment('plateau', tags=('place',), rule='',
                   variants=tuple(_PLATEAU))(_wrap(pieces.plateau, _PLATEAU))

ramp_down = fragment('ramp_down', tags=('place', 'speed'), rule='',
                     variants=tuple(_RAMP))(_wrap(pieces.ramp_down, _RAMP))

kicker = fragment('kicker', tags=('speed', 'gate'), cost=4.0,
                  rule='enter fast enough to climb the far side',
                  variants=tuple(_KICKER))(_wrap(pieces.kicker, _KICKER))

spillway = fragment('spillway', tags=('brake', 'hazard'), cost=6.0,
                    rule='hold the descent or run off the open end',
                    variants=tuple(_SPILLWAY))(_wrap(pieces.spillway, _SPILLWAY))

hairpin = fragment('hairpin', tags=('brake', 'aim'), cost=5.0,
                   rule='brake for the right-angle or be carried past it',
                   variants=tuple(_HAIRPIN))(_wrap(pieces.hairpin, _HAIRPIN))

bridge = fragment('bridge', tags=('aim', 'hazard'), cost=6.0,
                  rule='cross a single cell with nothing beside it',
                  variants=tuple(_BRIDGE))(_wrap(pieces.bridge, _BRIDGE))

scatter = fragment('scatter', tags=('luck', 'hazard'), cost=3.0,
                   rule='get through a field of bumpers that will not have you straight',
                   variants=tuple(_SCATTER))(_wrap(pieces.scatter, _SCATTER))
