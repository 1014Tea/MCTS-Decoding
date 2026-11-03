"""Illustrate earlier target visitation after one legal basis exchange.

This fixed example was selected offline to demonstrate the interface. It does
not measure average gains, and the exchange rule uses no target codeword.
"""
from dataclasses import asdict
import json
import numpy as np
from policy_osd import ModelBranchPolicy, PruneDecision, decode, prepare_basis
from policy_osd.demo import hamming_generator
from policy_osd.tree import dense


def example_scores(observation):
    # Replace with a model call returning [extend_score, adjacent_score].
    # These fixed scores demonstrate the API; they are not learned weights.
    assert observation.basis.systematic.shape == (4,7)
    assert np.array_equal(observation.basis.systematic[:,:4],np.eye(4,dtype=np.uint8))
    return [0.,1.]


def example_refinement(basis,legal,step):
    # Deliberately simple interface example, not a performance optimization.
    return legal[0] if step==0 and legal else None


def example_stop(progress):
    # Exact observation/codeword agreement; not the PB-OSD approximation.
    return progress.best.squared_distance < 1e-12


def example_prune(basis,support,best):
    # Attach a valid candidate or subtree discarding rule here.
    return PruneDecision.KEEP


def summarize(result, generator, received, transmitted):
    """Evaluate the trace offline; the receiver never sees transmitted."""
    view = prepare_basis(generator, received, result.basis)
    first_position = None
    prefix = []
    for position, support in enumerate(result.checked_supports, start=1):
        candidate = view.encode_tep(dense(support, view.k))
        prefix.append(''.join(map(str, candidate.tolist())))
        if np.array_equal(candidate, transmitted):
            first_position = position
            break
    return dict(
        selected_basis=result.basis,
        exchanges=[asdict(a) for a in result.exchanges],
        target_first_checked_position=first_position,
        checked=result.checked,
        codewords_through_target=prefix,
        decoded=result.codeword.tolist() if result.codeword is not None else None,
        correct=bool(np.array_equal(result.codeword, transmitted)),
    )


def main():
    generator=hamming_generator()
    message=np.array([1,0,1,0],dtype=np.uint8)
    transmitted=message@generator%2
    seed = 2331
    noise_std = .8
    received=1.-2.*transmitted.astype(float)+noise_std*np.random.default_rng(seed).normal(size=7)
    # Keep traversal and stopping identical so the difference comes from the basis.
    shared = dict(order=2, branch_policy=ModelBranchPolicy(example_scores),
                  stopping=example_stop, pruning=example_prune, trace=True)
    conventional = decode(generator, received, **shared)
    refined = decode(generator, received, mrb_policy=example_refinement, **shared)
    print(json.dumps(dict(
        policy='same fixed-score policy for both runs; no trained model',
        sample='offline-selected illustration, not an average-performance result',
        seed=seed, noise_std=noise_std, target=transmitted.tolist(),
        target_position='one-based offline statistic, not an online stopping rule',
        conventional=summarize(conventional, generator, received, transmitted),
        refined=summarize(refined, generator, received, transmitted),
    ), indent=2))


if __name__=='__main__':
    main()
