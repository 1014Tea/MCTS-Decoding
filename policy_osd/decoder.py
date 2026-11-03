"""Basis refinement followed by policy-guided depth-first TEP traversal."""
from __future__ import annotations
from dataclasses import dataclass
from math import comb
import numpy as np
from .algebra import prepare_basis, readonly
from .interfaces import (BranchPolicy, Candidate, Exchange, MRBPolicy, Progress,
                         PruneDecision, PruneRule, ReliabilityPolicy, StopRule,
                         make_observation)
from .refinement import refine
from .tree import children, dense, validate_order


@dataclass(frozen=True)
class DecodeResult:
    codeword: np.ndarray | None
    squared_distance: float | None
    checked: int
    visited: int
    pruned: int
    policy_calls: int
    reason: str
    basis: tuple[int,...]
    exchanges: tuple[Exchange,...]
    checked_supports: tuple[tuple[int,...],...]


def decode(generator,received,*,order=2,budget=None,basis=None,
           branch_policy: BranchPolicy | None=None,
           mrb_policy: MRBPolicy | None=None,max_exchanges=2,
           stopping: StopRule | None=None,pruning: PruneRule | None=None,
           trace=False) -> DecodeResult:
    """No reference/transmitted codeword is accepted by the receiver API.

    With no stopping or pruning and a complete budget, the policy changes only
    visiting order for the selected basis. A different basis changes a bounded-
    order candidate set; capped or early-stopped runs can have different outputs.
    """
    view=prepare_basis(generator,received,basis)
    validate_order(view.k,order)
    if budget is None:
        budget=sum(comb(view.k,w) for w in range(order+1))
    if not isinstance(budget,int) or isinstance(budget,bool) or budget<1:
        raise ValueError("budget must be a positive integer")
    view,exchanges=refine(view,mrb_policy,max_exchanges)
    policy=branch_policy if branch_policy is not None else ReliabilityPolicy(order)
    stack=[()];best=None;checked=visited=pruned=calls=0;records=[]
    reason="space_exhausted"
    while stack:
        support=stack.pop();visited+=1
        decision=PruneDecision.KEEP if pruning is None else pruning(view,support,best)
        if not isinstance(decision,PruneDecision):
            raise ValueError("Pruning must return a PruneDecision")
        if decision != PruneDecision.KEEP:
            pruned+=1
        if decision == PruneDecision.SKIP_SUBTREE:
            continue
        codeword=None
        if decision == PruneDecision.KEEP:
            codeword=view.encode_tep(dense(support,view.k))
            difference=view.received-(1.0-2.0*codeword.astype(float))
            distance=float(np.dot(difference,difference))
            candidate=Candidate(readonly(codeword),support,distance)
            checked+=1
            if trace:
                records.append(support)
            if best is None or (distance,tuple(codeword)) < (best.squared_distance,tuple(best.codeword)):
                best=candidate
            if stopping is not None and stopping(Progress(view,candidate,best,checked,visited)):
                reason="stopping_rule"
                break
        successors=children(support,view.k,order)
        if checked>=budget:
            reason="budget_exhausted" if stack or successors else "space_exhausted"
            break
        actions=tuple(successors)
        if len(actions)==2:
            if codeword is None:
                # Candidate-only pruning must not silently remove child states.
                codeword=view.encode_tep(dense(support,view.k))
            observation=make_observation(view,support,codeword,actions)
            preferred=policy(observation);calls+=1
            if preferred not in successors:
                raise ValueError("Branch policy returned an illegal action")
            other=next(action for action in actions if action!=preferred)
            stack.extend([successors[other],successors[preferred]])
        elif actions:
            stack.append(successors[actions[0]])
    if best is None:
        reason="no_candidate"
    return DecodeResult(None if best is None else best.codeword.copy(),
        None if best is None else best.squared_distance,checked,visited,pruned,calls,
        reason,view.positions,exchanges,tuple(records))
