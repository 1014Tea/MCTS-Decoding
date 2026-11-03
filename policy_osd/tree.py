"""Deterministic extend/adjacent tree; every order-m TEP appears once."""
from __future__ import annotations
import numpy as np

EXTEND, ADJACENT = 0, 1


def validate_order(k, order):
    if not isinstance(k, int) or k < 1 or not isinstance(order, int) or not 0 <= order <= k:
        raise ValueError("Expected integer k >= 1 and 0 <= order <= k")


def children(support: tuple[int, ...], k: int, order: int):
    validate_order(k, order)
    if (len(support) > order or tuple(sorted(set(support))) != support or
            any(i < 0 or i >= k for i in support)):
        raise ValueError("Invalid TEP support")
    result = {}
    if len(support) < order and (not support or support[-1] != k - 1):
        result[EXTEND] = support + (k - 1,)
    if support and support[-1] > 0 and (len(support) == 1 or support[-1]-1 > support[-2]):
        result[ADJACENT] = support[:-1] + (support[-1]-1,)
    return result


def dense(support, k):
    result = np.zeros(k, dtype=np.uint8)
    result[list(support)] = 1
    return result
