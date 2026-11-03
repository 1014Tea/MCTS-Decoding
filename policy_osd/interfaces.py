"""Extension contracts. No trained neural network or weights are included."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol
import numpy as np
from .algebra import BasisView, readonly
from .tree import children, dense


@dataclass(frozen=True)
class Exchange:
    outgoing: int
    incoming: int


@dataclass(frozen=True)
class BranchObservation:
    basis: BasisView
    support: tuple[int, ...]
    tep: np.ndarray
    candidate_systematic: np.ndarray
    signed_distance: np.ndarray
    normalized_signed_distance: np.ndarray
    legal_actions: tuple[int, ...]


class BranchPolicy(Protocol):
    def __call__(self, observation: BranchObservation) -> int:
        """Return one of observation.legal_actions; only queried at branches."""
        ...


class MRBPolicy(Protocol):
    def __call__(self, basis: BasisView, legal: tuple[Exchange, ...], step: int) -> Exchange | None:
        """Return a legal exchange, or None to STOP. No transmitted word is supplied."""
        ...


class PruneDecision(Enum):
    KEEP = "keep"
    SKIP_CANDIDATE = "skip_candidate"
    SKIP_SUBTREE = "skip_subtree"


@dataclass(frozen=True)
class Candidate:
    codeword: np.ndarray
    support: tuple[int, ...]
    squared_distance: float


@dataclass(frozen=True)
class Progress:
    basis: BasisView
    current: Candidate
    best: Candidate
    checked: int
    visited: int


class StopRule(Protocol):
    def __call__(self, progress: Progress) -> bool:
        """Run a supplied candidate-level acceptance rule after a metric check."""
        ...


class PruneRule(Protocol):
    def __call__(self, basis: BasisView, support: tuple[int, ...], best: Candidate | None) -> PruneDecision:
        """A subtree may be removed only when the supplied rule justifies it."""
        ...


class ModelBranchPolicy:
    """Adapt a private/external model returning scores for [extend, adjacent]."""
    def __init__(self, predict: Callable[[BranchObservation], object]):
        self.predict = predict

    def __call__(self, observation):
        scores = np.asarray(self.predict(observation), dtype=float)
        if scores.shape != (2,) or not np.isfinite(scores).all():
            raise ValueError("The model adapter expects two finite action scores")
        return max(observation.legal_actions, key=lambda action: (scores[action], -action))


class ReliabilityPolicy:
    """Executable non-neural fallback, not the learned policy of the paper."""
    def __init__(self, order):
        self.order = order

    def __call__(self, observation):
        view = observation.basis
        successors = children(observation.support, view.k, self.order)
        return min(observation.legal_actions, key=lambda action: (
            float(np.dot(np.abs(view.received_systematic[:view.k]), dense(successors[action], view.k))),
            action))


def make_observation(basis, support, candidate, legal_actions):
    permuted = candidate[basis.permutation]
    symbols = 1.0 - 2.0*permuted.astype(float)
    signed = (basis.received_systematic-symbols)**2*symbols
    scale = max(float(np.mean(np.abs(signed))), 1e-6)
    return BranchObservation(basis, support, readonly(dense(support,basis.k)),
        readonly(permuted), readonly(signed), readonly(signed/scale), tuple(legal_actions))
