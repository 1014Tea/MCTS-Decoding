"""Binary linear algebra and basis-aware systematic re-encoding."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def binary_matrix(matrix) -> np.ndarray:
    array = np.asarray(matrix)
    if array.ndim != 2 or min(array.shape) < 1 or not np.isin(array, (0, 1)).all():
        raise ValueError("Expected a nonempty two-dimensional binary matrix")
    return array.astype(np.uint8, copy=True)


def gf2_rank(matrix) -> int:
    array = binary_matrix(matrix)
    rank = 0
    for col in range(array.shape[1]):
        rows = np.flatnonzero(array[rank:, col])
        if not len(rows):
            continue
        row = rank + int(rows[0])
        array[[rank, row]] = array[[row, rank]]
        for other in range(rank + 1, len(array)):
            if array[other, col]:
                array[other] ^= array[rank]
        rank += 1
        if rank == len(array):
            break
    return rank


def gf2_inverse(matrix) -> np.ndarray:
    array = binary_matrix(matrix)
    k, n = array.shape
    if k != n:
        raise ValueError("Only square binary matrices can be inverted")
    aug = np.concatenate([array, np.eye(k, dtype=np.uint8)], axis=1)
    for col in range(k):
        choices = np.flatnonzero(aug[col:, col])
        if not len(choices):
            raise ValueError("Selected columns are not independent over GF(2)")
        pivot = col + int(choices[0])
        aug[[col, pivot]] = aug[[pivot, col]]
        for row in range(k):
            if row != col and aug[row, col]:
                aug[row] ^= aug[col]
    return aug[:, k:]


def validate_received(generator, received):
    generator = binary_matrix(generator)
    k, n = generator.shape
    if k > n or gf2_rank(generator) != k:
        raise ValueError("Generator must have full row rank and k <= n")
    received = np.asarray(received, dtype=np.float64)
    if received.shape != (n,) or not np.isfinite(received).all():
        raise ValueError("Received vector must contain n finite real values")
    return generator, received.copy()


def conventional_basis(generator, received) -> tuple[int, ...]:
    """Greedily retain independent columns in descending reliability order."""
    generator, received = validate_received(generator, received)
    order = np.argsort(-np.abs(received), kind="stable")
    work = generator.copy()
    selected = []
    for col in order:
        rank = len(selected)
        choices = np.flatnonzero(work[rank:, col])
        if not len(choices):
            continue
        pivot = rank + int(choices[0])
        work[[rank, pivot]] = work[[pivot, rank]]
        for row in range(rank + 1, len(work)):
            if work[row, col]:
                work[row] ^= work[rank]
        selected.append(int(col))
        if len(selected) == len(generator):
            break
    return tuple(selected)


def readonly(array):
    result = np.asarray(array).copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class BasisView:
    """All coordinate-dependent objects are rebuilt for the selected basis."""
    generator: np.ndarray
    received: np.ndarray
    positions: tuple[int, ...]
    permutation: np.ndarray
    systematic: np.ndarray
    received_systematic: np.ndarray
    hard_information: np.ndarray

    @property
    def k(self):
        return self.generator.shape[0]

    @property
    def n(self):
        return self.generator.shape[1]

    def encode_tep(self, tep):
        tep = np.asarray(tep)
        if tep.shape != (self.k,) or not np.isin(tep, (0, 1)).all():
            raise ValueError("A TEP must be a binary vector of length k")
        info = self.hard_information ^ tep.astype(np.uint8)
        permuted = (info @ self.systematic) % 2
        original = np.empty(self.n, dtype=np.uint8)
        original[self.permutation] = permuted
        return original


def prepare_basis(generator, received, positions=None) -> BasisView:
    generator, received = validate_received(generator, received)
    k, n = generator.shape
    if positions is None:
        positions = conventional_basis(generator, received)
    positions = tuple(positions)
    if (len(positions) != k or len(set(positions)) != k or
            any(not isinstance(i, (int, np.integer)) or isinstance(i, (bool, np.bool_)) or
                i < 0 or i >= n for i in positions)):
        raise ValueError("A basis must contain k distinct code-position indices")
    positions = tuple(sorted(map(int, positions), key=lambda i: (-abs(received[i]), i)))
    inverse = gf2_inverse(generator[:, positions])
    original_order = (inverse @ generator) % 2
    remaining = sorted(set(range(n)) - set(positions), key=lambda i: (-abs(received[i]), i))
    permutation = np.array(positions + tuple(remaining), dtype=np.int64)
    systematic = original_order[:, permutation]
    if not np.array_equal(systematic[:, :k], np.eye(k, dtype=np.uint8)):
        raise RuntimeError("Systematization failed")
    rp = received[permutation]
    return BasisView(readonly(generator), readonly(received), positions,
                     readonly(permutation), readonly(systematic), readonly(rp),
                     readonly((rp[:k] < 0).astype(np.uint8)))
