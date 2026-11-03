"""Rank-preserving exchange interface; no learned MRB selector is released."""
from __future__ import annotations
from .algebra import prepare_basis
from .interfaces import Exchange


def legal_exchanges(view, parent=None):
    parent = frozenset(parent) if parent is not None else None
    current = frozenset(view.positions)
    choices = []
    for col in range(view.k, view.n):
        entering = int(view.permutation[col])
        for row, leaving in enumerate(view.positions):
            if view.systematic[row,col] and current.difference([leaving]).union([entering]) != parent:
                choices.append(Exchange(leaving,entering))
    return tuple(choices)


def refine(view, policy=None, max_exchanges=2):
    if not isinstance(max_exchanges,int) or max_exchanges < 0:
        raise ValueError("max_exchanges must be a nonnegative integer")
    history=[];parent=None
    if policy is None:
        return view,tuple(history)
    for step in range(max_exchanges):
        legal=legal_exchanges(view,parent)
        action=policy(view,legal,step)
        if action is None:
            break
        if not isinstance(action,Exchange) or action not in legal:
            raise ValueError("MRB policy returned an illegal exchange")
        positions=list(view.positions);positions[positions.index(action.outgoing)]=action.incoming
        parent=view.positions
        view=prepare_basis(view.generator,view.received,positions)
        history.append(action)
    return view,tuple(history)
